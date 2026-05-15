from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    start_beat: float
    duration_beats: float
    velocity: int


@dataclass(frozen=True)
class EffectSpec:
    name: str
    parameters: JsonDict


@dataclass(frozen=True)
class SynthSpec:
    engine: str
    patch_name: str
    parameters: JsonDict


@dataclass(frozen=True)
class AutomationSpec:
    target: str
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    name: str
    instrument_category: str
    role: str
    midi_channel: int
    gain_db: float
    pan: float
    synth: SynthSpec
    effects: list[EffectSpec]
    notes: list[NoteEvent]
    automation: list[AutomationSpec] = field(default_factory=list)
    sidechain_from: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArrangementSection:
    name: str
    start_beat: float
    duration_beats: float


@dataclass(frozen=True)
class ProjectSpec:
    project_id: str
    seed: int
    bpm: int
    key: str
    scale: str
    duration_beats: float
    arrangement: list[ArrangementSection]
    tracks: list[TrackSpec]

    @property
    def source_count(self) -> int:
        return len(self.tracks)


@dataclass(frozen=True)
class RenderedTrack:
    track_id: str
    stem_path: str
    peak: float
    rms: float
    loudness_dbfs: float
    details: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class RenderResult:
    mixture_path: str
    midi_path: str | None
    stems: list[RenderedTrack]
    status: str = "rendered"
    render_plan_path: str | None = None
    notes: list[str] = field(default_factory=list)


def to_json_dict(value: Any) -> JsonDict:
    return asdict(value)


def project_metadata(project: ProjectSpec, render: RenderResult | None = None) -> JsonDict:
    data = to_json_dict(project)
    data["source_count"] = project.source_count
    if render is not None:
        data["render"] = to_json_dict(render)
    return data


def normalize_path(path: Path) -> str:
    return path.as_posix()
