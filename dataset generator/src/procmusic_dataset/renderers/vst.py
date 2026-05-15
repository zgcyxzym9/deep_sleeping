from __future__ import annotations

import math
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from procmusic_dataset.midi import write_midi, write_track_midi
from procmusic_dataset.models import ProjectSpec, RenderResult, RenderedTrack, TrackSpec, normalize_path
from procmusic_dataset.renderers.base import Renderer
from procmusic_dataset.renderers.vst_audio import (
    PendingStem,
    add_to_mixture,
    measure,
    normalize_project_loudness,
    project_duration_seconds,
    validate_audio,
    validate_output_file,
    write_mp3_preview,
    write_wav,
)
from procmusic_dataset.renderers.vst_backend import (
    DawDreamerBackend,
    VSTBackend,
    VSTTrackRender,
    validate_plugin_architecture,
)
from procmusic_dataset.renderers.vst_drums import render_drum_track
from procmusic_dataset.renderers.vst_presets import VSTPreset, try_select_preset as _try_select_preset
from procmusic_dataset.renderers.vst_retry import render_pitched_with_retries


DEFAULT_VST_PLUGIN = Path("C:/Program Files/Common Files/VST3/Surge Synth Team/Surge XT.vst3")
LOGGER = logging.getLogger("procmusic_dataset")


@dataclass(frozen=True)
class VSTRendererConfig:
    plugin_path: Path = DEFAULT_VST_PLUGIN
    sample_rate: int = 44_100
    block_size: int = 512
    write_preview_mp3: bool = False

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not self.plugin_path.exists():
            raise FileNotFoundError(f"VST plugin does not exist: {self.plugin_path}")
        validate_plugin_architecture(self.plugin_path)


class VSTRenderer(Renderer):
    """Render dataset audio by hosting a VST plugin from Python."""

    def __init__(self, config: VSTRendererConfig | None = None, backend: VSTBackend | None = None) -> None:
        self.config = config or VSTRendererConfig()
        self.backend = backend or DawDreamerBackend()

    def render(self, project: ProjectSpec, output_dir: Path) -> RenderResult:
        started = time.perf_counter()
        self.config.validate()
        stems_dir = output_dir / "stems"
        midi_dir = output_dir / "midi"
        stems_dir.mkdir(parents=True, exist_ok=True)
        midi_dir.mkdir(parents=True, exist_ok=True)

        project_midi = midi_dir / "project.mid"
        write_midi(project, project_midi)

        total_samples = int(math.ceil(project_duration_seconds(project) * self.config.sample_rate))
        mixture = np.zeros((2, total_samples), dtype=np.float32)
        pending_stems: list[PendingStem] = []

        for track in project.tracks:
            track_started = time.perf_counter()
            stem = self._render_track(project, track, midi_dir, total_samples)
            validate_audio(stem.track, stem.audio)
            add_to_mixture(mixture, stem.audio)
            pending_stems.append(stem)
            LOGGER.debug(
                "rendered track project=%s track=%s role=%s seconds=%.3f",
                project.project_id,
                track.name,
                track.role,
                time.perf_counter() - track_started,
            )

        pending_stems, mixture, normalization = normalize_project_loudness(project, pending_stems, mixture)
        mixture_path = output_dir / "mixture.wav"
        write_wav(mixture_path, mixture, self.config.sample_rate)
        validate_output_file(mixture_path)

        rendered_tracks = self._write_stems(stems_dir, pending_stems)
        notes = self._render_notes(normalization)
        if self.config.write_preview_mp3:
            preview_path = output_dir / "mixture_preview.mp3"
            write_mp3_preview(mixture_path, preview_path)
            notes.append(f"preview_mp3_path={normalize_path(preview_path)}")

        LOGGER.debug("rendered project=%s seconds=%.3f", project.project_id, time.perf_counter() - started)
        return RenderResult(
            mixture_path=normalize_path(mixture_path),
            midi_path=normalize_path(project_midi),
            stems=rendered_tracks,
            status="rendered",
            notes=notes,
        )

    def _render_track(self, project: ProjectSpec, track: TrackSpec, midi_dir: Path, total_samples: int) -> PendingStem:
        track_midi = midi_dir / f"{track.name}.mid"
        if track.role == "drums":
            write_track_midi(project, track, track_midi)
            audio = render_drum_track(project, track, total_samples, self.config.sample_rate)
            details = _drum_details(project, track, track_midi)
            return PendingStem(track, audio, details)

        resolved = render_pitched_with_retries(
            self.backend,
            self.config.plugin_path,
            project,
            track,
            self.config.sample_rate,
            self.config.block_size,
            total_samples,
        )
        write_track_midi(project, resolved.track, track_midi)
        details = {
            "backend": self.backend.name,
            "plugin_path": normalize_path(self.config.plugin_path),
            "parameter_randomization": _track_parameter_randomization(project, resolved.track),
            "midi_path": normalize_path(track_midi),
            **resolved.details,
        }
        return PendingStem(resolved.track, resolved.audio, details)

    def _write_stems(self, stems_dir: Path, stems: list[PendingStem]) -> list[RenderedTrack]:
        rendered_tracks = []
        for stem in stems:
            started = time.perf_counter()
            stem_path = stems_dir / f"{stem.track.name}.wav"
            write_wav(stem_path, stem.audio, self.config.sample_rate)
            peak, rms, dbfs = measure(stem.audio)
            LOGGER.debug("wrote stem path=%s seconds=%.3f", stem_path, time.perf_counter() - started)
            rendered_tracks.append(
                RenderedTrack(stem.track.track_id, normalize_path(stem_path), peak, rms, dbfs, stem.details)
            )
        return rendered_tracks

    def _render_notes(self, normalization: dict) -> list[str]:
        return [
            f"VST renderer backend={self.backend.name}",
            f"plugin_path={normalize_path(self.config.plugin_path)}",
            "FL Studio UI automation is not used by this renderer.",
            (
                "mixture_loudness_normalization="
                f"target_dbfs={normalization['target_dbfs']},"
                f"input_dbfs={normalization['input_dbfs']},"
                f"applied_gain_db={normalization['applied_gain_db']}"
            ),
        ]


def _drum_details(project: ProjectSpec, track: TrackSpec, track_midi: Path) -> dict:
    return {
        "backend": "internal_drum_synth",
        "plugin_path": None,
        "preset_or_program": "deterministic_reference_drum_kit",
        "patch_selection_status": "internal_fallback",
        "parameter_randomization": _track_parameter_randomization(project, track),
        "midi_path": normalize_path(track_midi),
    }


def _track_parameter_randomization(project: ProjectSpec, track: TrackSpec) -> dict:
    rng = random.Random(f"{project.seed}:{track.track_id}:vst-params")
    return {
        "gain_db": track.gain_db,
        "pan": track.pan,
        "macro_1": round(rng.random(), 6),
        "macro_2": round(rng.random(), 6),
        "effects": [effect.name for effect in track.effects],
    }
