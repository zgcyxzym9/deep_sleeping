import builtins
import math
import types
import subprocess
from pathlib import Path

import pytest

from procmusic_dataset.config import BatchConfig, GenerationConfig
from procmusic_dataset.generator import ProjectGenerator
from procmusic_dataset.models import ProjectSpec, TrackSpec
from procmusic_dataset.pipeline import DatasetPipeline
from procmusic_dataset.renderers.vst import (
    DawDreamerBackend,
    VSTPreset,
    VSTRenderer,
    VSTRendererConfig,
    VSTTrackRender,
    _try_select_preset,
)
from procmusic_dataset.renderers import vst_presets


class MockVSTBackend:
    name = "mock_vst"

    def render_pitched_track(
        self,
        plugin_path: Path,
        project: ProjectSpec,
        track: TrackSpec,
        sample_rate: int,
        block_size: int,
        attempt: int = 0,
    ) -> list[list[float]]:
        total_samples = int(math.ceil(project.duration_beats * 60.0 / project.bpm * sample_rate))
        audio = [[0.0] * total_samples, [0.0] * total_samples]
        for note in track.notes:
            start = int(note.start_beat * 60.0 / project.bpm * sample_rate)
            length = max(1, int(note.duration_beats * 60.0 / project.bpm * sample_rate))
            frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
            for offset in range(length):
                index = start + offset
                if index >= total_samples:
                    break
                sample = 0.2 * math.sin(2.0 * math.pi * frequency * offset / sample_rate)
                audio[0][index] += sample
                audio[1][index] += sample
        return audio


class MockPresetVSTBackend(MockVSTBackend):
    name = "mock_preset_vst"

    def render_pitched_track(
        self,
        plugin_path: Path,
        project: ProjectSpec,
        track: TrackSpec,
        sample_rate: int,
        block_size: int,
        attempt: int = 0,
    ) -> VSTTrackRender:
        return VSTTrackRender(
            super().render_pitched_track(plugin_path, project, track, sample_rate, block_size),
            {
                "preset_or_program": f"preset_for_{track.role}",
                "patch_selection_status": "preset_or_program_selected",
                "available_preset_count": 128,
                "available_parameter_count": 512,
                "parameter_randomization": [],
            },
        )


class MockSilentVSTBackend:
    name = "mock_silent_vst"

    def render_pitched_track(
        self,
        plugin_path: Path,
        project: ProjectSpec,
        track: TrackSpec,
        sample_rate: int,
        block_size: int,
        attempt: int = 0,
    ) -> VSTTrackRender:
        total_samples = int(math.ceil(project.duration_beats * 60.0 / project.bpm * sample_rate))
        return VSTTrackRender(
            [[0.0] * total_samples, [0.0] * total_samples],
            {
                "preset_or_program": "silent_patch",
                "patch_selection_status": "preset_or_program_selected",
                "available_preset_count": 1,
                "available_parameter_count": 0,
                "parameter_randomization": [],
            },
        )


class MockRetryVSTBackend(MockVSTBackend):
    name = "mock_retry_vst"

    def render_pitched_track(
        self,
        plugin_path: Path,
        project: ProjectSpec,
        track: TrackSpec,
        sample_rate: int,
        block_size: int,
        attempt: int = 0,
    ) -> VSTTrackRender:
        if attempt == 0:
            total_samples = int(math.ceil(project.duration_beats * 60.0 / project.bpm * sample_rate))
            audio = [[0.0] * total_samples, [0.0] * total_samples]
        else:
            audio = super().render_pitched_track(plugin_path, project, track, sample_rate, block_size, attempt=attempt)
        return VSTTrackRender(
            audio,
            {
                "preset_or_program": f"attempt_{attempt}",
                "patch_selection_status": "preset_or_program_selected",
                "available_preset_count": 2,
                "available_parameter_count": 0,
                "parameter_randomization": [],
            },
        )


def test_vst_renderer_reports_missing_dawdreamer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plugin = tmp_path / "instrument.dll"
    plugin.write_bytes(b"fake")
    project = ProjectGenerator(
        GenerationConfig(min_tracks=1, max_tracks=1, min_bars=1, max_bars=1, allow_drums=False)
    ).generate(0, 12)
    pitched_track = next(track for track in project.tracks if track.role != "drums")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "dawdreamer":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(RuntimeError, match="DawDreamer is required"):
        DawDreamerBackend().render_pitched_track(plugin, project, pitched_track, 8_000, 128)


def test_vst_renderer_rejects_32_bit_plugin_on_64_bit_python(tmp_path: Path):
    plugin = tmp_path / "instrument.dll"
    plugin.write_bytes(_fake_pe(machine=0x14C))

    with pytest.raises(RuntimeError, match="32-bit VST plugin"):
        VSTRendererConfig(plugin_path=plugin).validate()


def test_dawdreamer_backend_reuses_processor_only_within_retry_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plugin = tmp_path / "instrument.vst3"
    plugin.write_bytes(b"fake")
    project = ProjectGenerator(
        GenerationConfig(min_tracks=1, max_tracks=1, min_bars=1, max_bars=1, allow_drums=False)
    ).generate(0, 33)
    track = next(item for item in project.tracks if item.role != "drums")
    calls = {"make_plugin_processor": 0, "set_patch": 0, "clear_midi": 0}

    class Processor:
        def __init__(self):
            self.notes = []

        def get_patch(self):
            return [0.0]

        def set_patch(self, patch):
            calls["set_patch"] += 1

        def clear_midi(self):
            calls["clear_midi"] += 1
            self.notes.clear()

        def add_midi_note(self, pitch, velocity, start, duration):
            self.notes.append((pitch, velocity, start, duration))

    class Engine:
        def __init__(self, sample_rate, block_size):
            self.processor = None

        def make_plugin_processor(self, name, path):
            calls["make_plugin_processor"] += 1
            self.processor = Processor()
            return self.processor

        def load_graph(self, graph):
            return None

        def render(self, duration):
            return None

        def get_audio(self):
            return [[0.1, 0.0], [0.1, 0.0]]

    monkeypatch.setattr(
        "procmusic_dataset.renderers.vst_backend.select_plugin_sound",
        lambda plugin_path, processor, project, track, attempt: {
            "preset_or_program": "fake",
            "patch_selection_status": "preset_or_program_selected",
            "available_preset_count": 1,
            "available_parameter_count": 1,
            "parameter_randomization": [],
        },
    )
    backend = DawDreamerBackend()
    monkeypatch.setattr(backend, "_dawdreamer", lambda: types.SimpleNamespace(RenderEngine=Engine))

    backend.render_pitched_track(plugin, project, track, 8_000, 128, attempt=0)
    backend.render_pitched_track(plugin, project, track, 8_000, 128, attempt=1)

    assert calls["make_plugin_processor"] == 1
    assert calls["set_patch"] == 2
    assert calls["clear_midi"] == 4

    backend.render_pitched_track(plugin, project, track, 8_000, 128, attempt=0)

    assert calls["make_plugin_processor"] == 2


def test_vst_renderer_mock_backend_generates_dataset_audio_and_metadata(tmp_path: Path):
    plugin = tmp_path / "instrument.dll"
    plugin.write_bytes(b"fake")
    output_dir = tmp_path / "dataset"
    generation = GenerationConfig(
        min_tracks=2,
        max_tracks=2,
        min_bars=1,
        max_bars=1,
        sample_rate=8_000,
        allow_drums=False,
    )
    config = BatchConfig(output_dir=output_dir, count=1, seed=22, generation=generation)
    renderer = VSTRenderer(
        VSTRendererConfig(plugin_path=plugin, sample_rate=8_000, block_size=128),
        backend=MockVSTBackend(),
    )

    DatasetPipeline(config, renderer).run()

    project_dir = output_dir / "project_000000"
    assert (project_dir / "metadata.json").exists()
    assert (project_dir / "mixture.wav").stat().st_size > 44
    stems = sorted((project_dir / "stems").glob("*.wav"))
    assert len(stems) == 2
    assert all(stem.stat().st_size > 44 for stem in stems)

    metadata = (project_dir / "metadata.json").read_text(encoding="utf-8")
    assert "mock_vst" in metadata
    assert "default_or_unavailable" in metadata
    assert str(project_dir).replace("\\", "/") in metadata


def test_vst_renderer_records_backend_preset_metadata(tmp_path: Path):
    plugin = tmp_path / "instrument.dll"
    plugin.write_bytes(b"fake")
    output_dir = tmp_path / "dataset"
    generation = GenerationConfig(
        min_tracks=1,
        max_tracks=1,
        min_bars=1,
        max_bars=1,
        sample_rate=8_000,
        allow_drums=False,
    )
    renderer = VSTRenderer(
        VSTRendererConfig(plugin_path=plugin, sample_rate=8_000, block_size=128),
        backend=MockPresetVSTBackend(),
    )

    DatasetPipeline(BatchConfig(output_dir=output_dir, count=1, seed=24, generation=generation), renderer).run()

    metadata = (output_dir / "project_000000" / "metadata.json").read_text(encoding="utf-8")
    assert "mock_preset_vst" in metadata
    assert "preset_or_program_selected" in metadata
    assert "preset_for_" in metadata
    assert "available_preset_count" in metadata


def test_vst_renderer_falls_back_when_vst_track_is_silent(tmp_path: Path):
    plugin = tmp_path / "instrument.dll"
    plugin.write_bytes(b"fake")
    output_dir = tmp_path / "dataset"
    generation = GenerationConfig(
        min_tracks=1,
        max_tracks=1,
        min_bars=1,
        max_bars=1,
        sample_rate=8_000,
        allow_drums=False,
    )
    renderer = VSTRenderer(
        VSTRendererConfig(plugin_path=plugin, sample_rate=8_000, block_size=128),
        backend=MockSilentVSTBackend(),
    )

    DatasetPipeline(BatchConfig(output_dir=output_dir, count=1, seed=25, generation=generation), renderer).run()

    project_dir = output_dir / "project_000000"
    assert (project_dir / "metadata.json").exists()
    assert (project_dir / "mixture.wav").stat().st_size > 44
    metadata = (project_dir / "metadata.json").read_text(encoding="utf-8")
    assert "internal_pitched_synth_after_silent_vst" in metadata


def test_vst_renderer_retries_silent_track_with_new_attempt(tmp_path: Path):
    plugin = tmp_path / "instrument.dll"
    plugin.write_bytes(b"fake")
    output_dir = tmp_path / "dataset"
    generation = GenerationConfig(
        min_tracks=1,
        max_tracks=1,
        min_bars=1,
        max_bars=1,
        sample_rate=8_000,
        allow_drums=False,
    )
    renderer = VSTRenderer(
        VSTRendererConfig(plugin_path=plugin, sample_rate=8_000, block_size=128),
        backend=MockRetryVSTBackend(),
    )

    DatasetPipeline(BatchConfig(output_dir=output_dir, count=1, seed=26, generation=generation), renderer).run()

    metadata = (output_dir / "project_000000" / "metadata.json").read_text(encoding="utf-8")
    assert "attempt_1" in metadata
    assert '"silent": true' in metadata
    assert '"silent": false' in metadata
    assert "internal_pitched_synth_after_silent_vst_retries" not in metadata


def test_vst_retry_reuses_silence_check_for_successful_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plugin = tmp_path / "instrument.dll"
    plugin.write_bytes(b"fake")
    project = ProjectGenerator(
        GenerationConfig(min_tracks=1, max_tracks=1, min_bars=1, max_bars=1, sample_rate=8_000, allow_drums=False)
    ).generate(0, 27)
    track = next(item for item in project.tracks if item.role != "drums")
    calls = {"is_silent": 0}

    def counted_is_silent(audio):
        calls["is_silent"] += 1
        return False

    monkeypatch.setattr("procmusic_dataset.renderers.vst_retry.is_silent", counted_is_silent)
    from procmusic_dataset.renderers.vst_retry import render_pitched_with_retries

    render_pitched_with_retries(MockVSTBackend(), plugin, project, track, 8_000, 128, 1000)

    assert calls["is_silent"] == 1


def test_vst_preset_selection_loads_fxp_file(tmp_path: Path):
    preset = tmp_path / "Basses" / "Deep Bass.fxp"
    preset.parent.mkdir()
    preset.write_bytes(b"fake preset")

    class Processor:
        loaded = None
        value = 0.0

        def load_preset(self, path: str) -> None:
            self.loaded = path
            self.value = 0.5

        def get_plugin_parameters_description(self):
            return [{"name": "A Osc 1 Type"}]

        def get_parameter(self, index):
            return self.value

    processor = Processor()
    project = ProjectGenerator(
        GenerationConfig(min_tracks=1, max_tracks=1, min_bars=1, max_bars=1, allow_drums=False)
    ).generate(0, 31)
    track = next(item for item in project.tracks if item.role != "drums")

    details = _try_select_preset(
        processor,
        track,
        [VSTPreset("Deep Bass", preset, "Basses")],
        __import__("random").Random(1),
    )

    assert processor.loaded == str(preset.resolve())
    assert details["patch_selection_status"] == "preset_or_program_selected"
    assert details["preset_or_program"] == "Deep Bass"


def test_vst_filesystem_preset_scan_is_cached():
    vst_presets._filesystem_plugin_presets_cached.cache_clear()

    list(vst_presets.filesystem_plugin_presets(Path("C:/Program Files/Common Files/VST3/Surge XT.vst3")))
    before = vst_presets._filesystem_plugin_presets_cached.cache_info()
    list(vst_presets.filesystem_plugin_presets(Path("C:/Program Files/Common Files/VST3/Surge XT.vst3")))
    after = vst_presets._filesystem_plugin_presets_cached.cache_info()

    assert after.hits == before.hits + 1


def test_vst_parameter_descriptions_are_cached_by_plugin_path(tmp_path: Path):
    vst_presets._PLUGIN_PARAMETER_CACHE.clear()
    plugin = tmp_path / "instrument.vst3"
    plugin.write_bytes(b"fake")

    class Processor:
        calls = 0

        def get_plugin_parameters_description(self):
            self.calls += 1
            return [{"name": "A Osc 1 Type"}]

    first = Processor()
    second = Processor()

    assert vst_presets.plugin_parameters(first, plugin) == [{"index": 0, "name": "A Osc 1 Type"}]
    assert vst_presets.plugin_parameters(second, plugin) == [{"index": 0, "name": "A Osc 1 Type"}]
    assert first.calls == 1
    assert second.calls == 0


def test_vst_renderer_writes_optional_mp3_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plugin = tmp_path / "instrument.dll"
    plugin.write_bytes(b"fake")
    output_dir = tmp_path / "dataset"
    generation = GenerationConfig(
        min_tracks=1,
        max_tracks=1,
        min_bars=1,
        max_bars=1,
        sample_rate=8_000,
        allow_drums=False,
    )

    monkeypatch.setattr("procmusic_dataset.renderers.vst_audio.shutil.which", lambda name: "ffmpeg")

    def fake_run(args, check):
        Path(args[-1]).write_bytes(b"fake mp3 preview data" * 4)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("procmusic_dataset.renderers.vst_audio.subprocess.run", fake_run)

    renderer = VSTRenderer(
        VSTRendererConfig(plugin_path=plugin, sample_rate=8_000, block_size=128, write_preview_mp3=True),
        backend=MockVSTBackend(),
    )
    DatasetPipeline(BatchConfig(output_dir=output_dir, count=1, seed=23, generation=generation), renderer).run()

    project_dir = output_dir / "project_000000"
    assert (project_dir / "mixture_preview.mp3").stat().st_size > 0
    metadata = (project_dir / "metadata.json").read_text(encoding="utf-8")
    assert "mixture_preview.mp3" in metadata


def _fake_pe(machine: int) -> bytes:
    data = bytearray(256)
    data[0:2] = b"MZ"
    data[0x3C:0x40] = (0x80).to_bytes(4, "little")
    data[0x80:0x84] = b"PE\0\0"
    data[0x84:0x86] = machine.to_bytes(2, "little")
    return bytes(data)
