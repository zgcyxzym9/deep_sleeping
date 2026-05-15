from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from procmusic_dataset.midi import write_midi, write_track_midi
from procmusic_dataset.models import ProjectSpec, RenderResult, RenderedTrack, TrackSpec, normalize_path
from procmusic_dataset.renderers.base import Renderer


DEFAULT_FL_ROOT = Path("D:/fl")
GENERATOR_PRESET_DIR = Path("Data/Patches/Channel presets")
PLUGIN_PRESET_DIR = Path("Data/Patches/Plugin presets")


@dataclass(frozen=True)
class FLStudioConfig:
    root: Path = DEFAULT_FL_ROOT
    executable: Path | None = None

    @property
    def resolved_executable(self) -> Path:
        if self.executable is not None:
            return self.executable
        return self.root / "FL64.exe"

    @property
    def channel_preset_dir(self) -> Path:
        return self.root / GENERATOR_PRESET_DIR

    @property
    def plugin_preset_dir(self) -> Path:
        return self.root / PLUGIN_PRESET_DIR

    def validate(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"FL Studio root does not exist: {self.root}")
        if not self.resolved_executable.exists():
            raise FileNotFoundError(f"FL Studio executable does not exist: {self.resolved_executable}")
        if not self.channel_preset_dir.exists():
            raise FileNotFoundError(f"FL Studio channel preset directory does not exist: {self.channel_preset_dir}")
        if not self.plugin_preset_dir.exists():
            raise FileNotFoundError(f"FL Studio plugin preset directory does not exist: {self.plugin_preset_dir}")


@dataclass(frozen=True)
class FLPreset:
    plugin: str
    name: str
    path: str


@dataclass(frozen=True)
class FLTrackPlan:
    track_id: str
    track_name: str
    role: str
    instrument_category: str
    midi_path: str
    expected_stem_path: str
    preset: FLPreset
    effects: list[dict]


@dataclass(frozen=True)
class FLRenderPlan:
    status: str
    fl_root: str
    fl_executable: str
    project_id: str
    bpm: int
    seed: int
    project_midi_path: str
    expected_mixture_path: str
    tracks: list[FLTrackPlan]
    manual_steps: list[str]


class FLPresetCatalog:
    def __init__(self, presets: list[FLPreset]) -> None:
        self.presets = presets

    @classmethod
    def scan(cls, config: FLStudioConfig) -> "FLPresetCatalog":
        config.validate()
        presets: list[FLPreset] = []
        for path in sorted(config.channel_preset_dir.rglob("*.fst")):
            if not _is_usable_preset(path):
                continue
            presets.append(FLPreset(plugin=_plugin_from_channel_preset(config.channel_preset_dir, path), name=path.stem, path=normalize_path(path)))
        for path in sorted(config.plugin_preset_dir.rglob("*.fst")):
            if not _is_usable_preset(path):
                continue
            presets.append(FLPreset(plugin=_plugin_from_plugin_preset(config.plugin_preset_dir, path), name=path.stem, path=normalize_path(path)))
        if not presets:
            raise FileNotFoundError(f"No .fst channel presets found under {config.channel_preset_dir}")
        return cls(presets)

    def choose_for_track(self, track: TrackSpec, rng: random.Random) -> FLPreset:
        candidates = _ranked_candidates(self.presets, track)
        if not candidates:
            candidates = self.presets
        return rng.choice(candidates)


class FLStudioRenderer(Renderer):
    """Prepare deterministic FL Studio render plans.

    This class deliberately stops at a verified boundary: selecting installed FL
    channel presets and exporting MIDI/render plans. Creating .flp files and
    exporting audio require a verified automation method on the target FL Studio
    installation.
    """

    def __init__(self, config: FLStudioConfig | None = None) -> None:
        self.config = config or FLStudioConfig()
        self.catalog = FLPresetCatalog.scan(self.config)

    def render(self, project: ProjectSpec, output_dir: Path) -> RenderResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        midi_dir = output_dir / "flstudio_midi"
        expected_stem_dir = output_dir / "stems"
        expected_stem_dir.mkdir(parents=True, exist_ok=True)

        project_midi = midi_dir / "project.mid"
        write_midi(project, project_midi)

        rng = random.Random(project.seed)
        track_plans: list[FLTrackPlan] = []
        rendered_tracks: list[RenderedTrack] = []
        for track in project.tracks:
            track_midi = midi_dir / f"{track.name}.mid"
            write_track_midi(project, track, track_midi)
            preset = self.catalog.choose_for_track(track, rng)
            stem_path = expected_stem_dir / f"{track.name}.wav"
            track_plans.append(
                FLTrackPlan(
                    track_id=track.track_id,
                    track_name=track.name,
                    role=track.role,
                    instrument_category=track.instrument_category,
                    midi_path=normalize_path(track_midi),
                    expected_stem_path=normalize_path(stem_path),
                    preset=preset,
                    effects=[asdict(effect) for effect in track.effects],
                )
            )
            rendered_tracks.append(RenderedTrack(track.track_id, normalize_path(stem_path), 0.0, 0.0, -120.0))

        mixture_path = output_dir / "mixture.wav"
        plan = FLRenderPlan(
            status="prepared",
            fl_root=normalize_path(self.config.root),
            fl_executable=normalize_path(self.config.resolved_executable),
            project_id=project.project_id,
            bpm=project.bpm,
            seed=project.seed,
            project_midi_path=normalize_path(project_midi),
            expected_mixture_path=normalize_path(mixture_path),
            tracks=track_plans,
            manual_steps=[
                "Open FL Studio with the configured executable.",
                "For each track, create a channel using the selected .fst preset.",
                "Import the matching per-track MIDI file into that channel.",
                "Route each channel to its own mixer insert and export split mixer tracks.",
                "Place exported stems at the expected paths or implement an automation adapter that does this.",
            ],
        )
        plan_path = output_dir / "flstudio_render_plan.json"
        _write_plan(plan_path, plan)
        return RenderResult(
            mixture_path=normalize_path(mixture_path),
            midi_path=normalize_path(project_midi),
            stems=rendered_tracks,
            status="prepared",
            render_plan_path=normalize_path(plan_path),
            notes=["FL Studio plan prepared; audio export is not automated yet."],
        )


def _write_plan(path: Path, plan: FLRenderPlan) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(plan), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _plugin_from_channel_preset(base: Path, path: Path) -> str:
    relative = path.relative_to(base)
    return relative.parts[0] if len(relative.parts) > 1 else path.parent.name


def _plugin_from_plugin_preset(base: Path, path: Path) -> str:
    relative = path.relative_to(base)
    if len(relative.parts) >= 3 and relative.parts[0].lower() in {"generators", "effects"}:
        return relative.parts[1]
    return relative.parts[0] if len(relative.parts) > 1 else path.parent.name


def _is_usable_preset(path: Path) -> bool:
    text = " ".join(path.parts).lower()
    blocked = ("empty", "default", "template", "initialized", "init")
    return not any(token in text for token in blocked)


def _preset_matches_track(preset: FLPreset, track: TrackSpec) -> bool:
    text = f"{preset.plugin} {preset.name}".lower()
    if track.role == "drums":
        return any(token in text for token in ("drum", "fpc", "kick", "snare", "hat", "perc"))
    if track.role == "bass":
        return "bass" in text or preset.plugin.lower() in {"boobass", "transistor bass", "sytrus", "gms", "3x osc"}
    if track.role == "harmony":
        return any(token in text for token in ("pad", "key", "piano", "string", "chord", "organ", "ep"))
    if track.role == "melody":
        return any(token in text for token in ("lead", "pluck", "bell", "arp", "saw", "synth"))
    return True


def _ranked_candidates(presets: list[FLPreset], track: TrackSpec) -> list[FLPreset]:
    scored = [(score, preset) for preset in presets if (score := _preset_score(preset, track)) > 0]
    if not scored:
        return []
    best = max(score for score, _preset in scored)
    return [preset for score, preset in scored if score == best]


def _preset_score(preset: FLPreset, track: TrackSpec) -> int:
    plugin = preset.plugin.lower()
    text = f"{preset.plugin} {preset.name}".lower()
    score = 0
    if track.role == "drums":
        score += 5 if plugin in {"fpc", "drumaxx"} else 0
        score += 3 if plugin in {"bassdrum", "drumpad", "fruity drumsynth live"} else 0
        score += 1 if any(token in text for token in ("drum", "kick", "snare", "hat", "perc")) else 0
    elif track.role == "bass":
        score += 5 if plugin in {"boobass", "transistor bass"} else 0
        score += 3 if plugin in {"sytrus", "gms", "3x osc", "harmless", "harmor"} else 0
        score += 2 if "bass" in text else 0
    elif track.role == "harmony":
        score += 4 if plugin in {"fl keys", "sytrus", "gms", "harmless", "harmor", "sakura"} else 0
        score += 2 if any(token in text for token in ("pad", "key", "piano", "string", "chord", "organ", "ep")) else 0
    elif track.role == "melody":
        score += 4 if plugin in {"sytrus", "gms", "harmless", "harmor", "sawer", "poizone", "3x osc"} else 0
        score += 2 if any(token in text for token in ("lead", "pluck", "bell", "arp", "saw", "synth")) else 0
    else:
        score = 1 if _preset_matches_track(preset, track) else 0
    return score
