from __future__ import annotations

import platform
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from procmusic_dataset.models import ProjectSpec, TrackSpec
from procmusic_dataset.renderers.vst_audio import StereoAudio, coerce_stereo, project_duration_seconds
from procmusic_dataset.renderers.vst_presets import select_plugin_sound


LOGGER = logging.getLogger("procmusic_dataset")


@dataclass(frozen=True)
class VSTTrackRender:
    audio: StereoAudio
    details: dict


@dataclass
class _DawDreamerContext:
    engine: Any
    processor: Any
    initial_patch: object | None


class VSTBackend(Protocol):
    name: str

    def render_pitched_track(
        self,
        plugin_path: Path,
        project: ProjectSpec,
        track: TrackSpec,
        sample_rate: int,
        block_size: int,
        attempt: int = 0,
    ) -> VSTTrackRender | StereoAudio:
        """Return rendered audio and optional plugin selection metadata."""


class DawDreamerBackend:
    name = "dawdreamer"

    def __init__(self) -> None:
        self._daw = None
        self._contexts: dict[tuple[str, int, int, str, str], _DawDreamerContext] = {}

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
            self._contexts.clear()
        context = self._context(plugin_path, project, track, sample_rate, block_size)
        engine = context.engine
        processor = context.processor
        self._prepare_processor(context)
        plugin_details = select_plugin_sound(plugin_path, processor, project, track, attempt)
        for note in track.notes:
            start_seconds = note.start_beat * 60.0 / project.bpm
            duration = max(0.001, note.duration_beats * 60.0 / project.bpm)
            processor.add_midi_note(int(note.pitch), int(note.velocity), start_seconds, duration)
        LOGGER.debug(
            "dawdreamer render start project=%s track=%s attempt=%d preset=%s status=%s duration=%.3f",
            project.project_id,
            track.name,
            attempt,
            plugin_details.get("preset_or_program"),
            plugin_details.get("patch_selection_status"),
            project_duration_seconds(project),
        )
        engine.render(project_duration_seconds(project))
        audio = coerce_stereo(engine.get_audio())
        clear_midi = getattr(processor, "clear_midi", None)
        if callable(clear_midi):
            clear_midi()
        return VSTTrackRender(audio, plugin_details)

    def _context(
        self, plugin_path: Path, project: ProjectSpec, track: TrackSpec, sample_rate: int, block_size: int
    ) -> _DawDreamerContext:
        key = (str(plugin_path.resolve()), sample_rate, block_size, project.project_id, track.track_id)
        context = self._contexts.get(key)
        if context is not None:
            return context
        daw = self._dawdreamer()
        engine = daw.RenderEngine(sample_rate, block_size)
        processor = engine.make_plugin_processor("vst_processor", str(plugin_path))
        engine.load_graph([(processor, [])])
        context = _DawDreamerContext(engine, processor, self._get_patch(processor))
        self._contexts[key] = context
        return context

    def _dawdreamer(self):
        if self._daw is not None:
            return self._daw
        try:
            import dawdreamer as daw
        except ImportError as exc:
            raise RuntimeError(
                "DawDreamer is required for --renderer vst. Install the optional VST dependencies with "
                "`pip install -e .[vst]` or install `dawdreamer` in this environment."
            ) from exc
        self._daw = daw
        return daw

    def _prepare_processor(self, context: _DawDreamerContext) -> None:
        processor = context.processor
        clear_midi = getattr(processor, "clear_midi", None)
        if callable(clear_midi):
            clear_midi()
        if context.initial_patch is not None:
            set_patch = getattr(processor, "set_patch", None)
            if callable(set_patch):
                set_patch(context.initial_patch)

    def _get_patch(self, processor: object) -> object | None:
        get_patch = getattr(processor, "get_patch", None)
        if not callable(get_patch):
            return None
        try:
            return get_patch()
        except Exception:
            return None


def unpack_backend_result(result: VSTTrackRender | StereoAudio) -> tuple[StereoAudio, dict]:
    if isinstance(result, VSTTrackRender):
        return result.audio, result.details
    return result, {
        "preset_or_program": None,
        "patch_selection_status": "default_or_unavailable",
        "available_preset_count": 0,
        "available_parameter_count": 0,
    }


def validate_plugin_architecture(plugin_path: Path) -> None:
    machine = windows_pe_machine(plugin_path)
    if machine is None:
        return
    python_is_64bit = platform.architecture()[0] == "64bit"
    if machine == 0x14C and python_is_64bit:
        raise RuntimeError(
            f"Cannot load 32-bit VST plugin from 64-bit Python/DawDreamer: {plugin_path}. "
            "Use a 64-bit VST plugin, or run the renderer from a 32-bit Python environment with compatible DawDreamer."
        )
    if machine == 0x8664 and not python_is_64bit:
        raise RuntimeError(
            f"Cannot load 64-bit VST plugin from 32-bit Python/DawDreamer: {plugin_path}. "
            "Use a matching 64-bit Python environment or a 32-bit plugin."
        )


def windows_pe_machine(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"MZ":
                return None
            handle.seek(0x3C)
            pe_offset = int.from_bytes(handle.read(4), "little")
            handle.seek(pe_offset)
            if handle.read(4) != b"PE\0\0":
                return None
            return int.from_bytes(handle.read(2), "little")
    except OSError:
        return None
