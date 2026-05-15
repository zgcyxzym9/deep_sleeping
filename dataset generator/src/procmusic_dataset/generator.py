from __future__ import annotations

import random
from dataclasses import dataclass

from .config import GenerationConfig
from .models import (
    ArrangementSection,
    AutomationSpec,
    EffectSpec,
    NoteEvent,
    ProjectSpec,
    SynthSpec,
    TrackSpec,
)


KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "minor_pentatonic": [0, 3, 5, 7, 10],
}
ROOT_MIDI = {
    "C": 60,
    "Db": 61,
    "D": 62,
    "Eb": 63,
    "E": 64,
    "F": 65,
    "Gb": 66,
    "G": 67,
    "Ab": 68,
    "A": 69,
    "Bb": 70,
    "B": 71,
}


@dataclass(frozen=True)
class InstrumentTemplate:
    category: str
    role: str
    engine: str
    patch_prefixes: tuple[str, ...]


INSTRUMENTS = (
    InstrumentTemplate("synth_lead", "melody", "subtractive", ("saw_lead", "square_lead", "sync_lead")),
    InstrumentTemplate("synth_pad", "harmony", "wavetable", ("warm_pad", "glass_pad", "motion_pad")),
    InstrumentTemplate("synth_bass", "bass", "fm", ("fm_bass", "acid_bass", "round_bass")),
    InstrumentTemplate("electric_piano", "harmony", "sampled", ("soft_ep", "bell_ep", "chorus_ep")),
    InstrumentTemplate("pluck", "melody", "physical_model", ("nylon_pluck", "karplus", "muted_pluck")),
    InstrumentTemplate("string_ensemble", "harmony", "sampled", ("small_strings", "slow_strings")),
    InstrumentTemplate("drum_kit", "drums", "drum_sampler", ("tight_kit", "synthetic_kit", "noisy_kit")),
)


class ProjectGenerator:
    """Creates deterministic abstract music projects from integer seeds."""

    def __init__(self, config: GenerationConfig) -> None:
        config.validate()
        self.config = config

    def generate(self, project_index: int, seed: int) -> ProjectSpec:
        rng = random.Random(seed)
        bpm = rng.randint(self.config.min_bpm, self.config.max_bpm)
        key = rng.choice(KEYS)
        scale = rng.choice(list(SCALES))
        bars = rng.randint(self.config.min_bars, self.config.max_bars)
        duration_beats = float(bars * 4)
        track_count = rng.randint(self.config.min_tracks, self.config.max_tracks)
        arrangement = self._arrangement(bars, rng)

        templates = self._choose_templates(track_count, rng)
        tracks = [
            self._track(index, template, key, scale, duration_beats, rng)
            for index, template in enumerate(templates)
        ]
        # tracks = self._add_sidechains(tracks)

        return ProjectSpec(
            project_id=f"project_{project_index:06d}",
            seed=seed,
            bpm=bpm,
            key=key,
            scale=scale,
            duration_beats=duration_beats,
            arrangement=arrangement,
            tracks=tracks,
        )

    def _choose_templates(self, count: int, rng: random.Random) -> list[InstrumentTemplate]:
        """
        This function selects the type of instruments present in the sample, but does not
        specify what specific instruments are chosen.
        """
        pool = list(INSTRUMENTS)
        if not self.config.allow_drums:
            pool = [item for item in pool if item.role != "drums"]
        if not self.config.allow_synths:
            pool = [item for item in pool if not item.category.startswith("synth")]
        if not self.config.allow_acoustic:
            pool = [item for item in pool if item.engine not in {"sampled", "physical_model"}]
        if not pool:
            raise ValueError("instrument constraints removed every instrument template")

        selected = []
        if self.config.allow_drums and any(item.role == "drums" for item in pool) and rng.random() < 0.75:
            selected.append(next(item for item in pool if item.role == "drums"))
        remaining = [item for item in pool if item.category not in {template.category for template in selected}]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, count - len(selected))])
        rng.shuffle(selected)
        return selected

    def _arrangement(self, bars: int, rng: random.Random) -> list[ArrangementSection]:
        names = ["intro", "A", "B", "break", "outro"]
        sections: list[ArrangementSection] = []
        cursor = 0
        while cursor < bars:
            remaining = bars - cursor
            length = min(remaining, rng.choice([2, 4, 8]))
            sections.append(ArrangementSection(rng.choice(names), float(cursor * 4), float(length * 4)))
            cursor += length
        return sections

    def _track(
        self,
        index: int,
        template: InstrumentTemplate,
        key: str,
        scale: str,
        duration_beats: float,
        rng: random.Random,
    ) -> TrackSpec:
        patch = f"{rng.choice(template.patch_prefixes)}_{rng.randrange(10_000):04d}"
        notes = self._drum_notes(duration_beats, rng) if template.role == "drums" else self._pitched_notes(
            template.role, key, scale, duration_beats, rng
        )
        synth = SynthSpec(template.engine, patch, self._synth_parameters(template.engine, rng))
        effects = self._effects(template.role, rng)
        automation = self._automation(duration_beats, rng)
        return TrackSpec(
            track_id=f"trk_{index:03d}",
            name=f"{index:03d}_{template.category}",
            instrument_category=template.category,
            role=template.role,
            midi_channel=index % 16,
            gain_db=round(rng.uniform(-12.0, -3.0), 2),
            pan=round(rng.uniform(-0.75, 0.75), 3),
            synth=synth,
            effects=effects,
            notes=notes,
            automation=automation,
        )

    def _pitched_notes(
        self, role: str, key: str, scale: str, duration_beats: float, rng: random.Random
    ) -> list[NoteEvent]:
        """
        Generates notes for pitched instruments.
        """
        pitches = _role_scale_pitches(role, key, scale, rng)
        step = {"bass": 1.0, "harmony": 2.0, "melody": 0.5}.get(role, 0.5)
        duration = {"bass": 0.75, "harmony": 1.75, "melody": 0.45}.get(role, 0.45)
        notes: list[NoteEvent] = []
        beat = 0.0
        while beat < duration_beats:
            if rng.random() < 0.78:
                pitch = rng.choice(pitches)
                notes.append(NoteEvent(pitch, round(beat, 3), min(duration, duration_beats - beat), rng.randint(50, 118)))
            beat += step
        return notes

    def _drum_notes(self, duration_beats: float, rng: random.Random) -> list[NoteEvent]:
        """
        Generates notes for drums.
        """
        notes: list[NoteEvent] = []
        beat = 0.0
        while beat < duration_beats:
            if beat % 4 in (0, 2) or rng.random() < 0.18:
                notes.append(NoteEvent(36, round(beat, 3), 0.12, rng.randint(85, 122)))
            if beat % 4 in (1, 3) or rng.random() < 0.12:
                notes.append(NoteEvent(38, round(beat, 3), 0.10, rng.randint(75, 118)))
            notes.append(NoteEvent(42 if rng.random() < 0.8 else 46, round(beat, 3), 0.05, rng.randint(45, 105)))
            beat += 0.5
        return notes

    def _synth_parameters(self, engine: str, rng: random.Random) -> dict[str, float | str]:
        common: dict[str, float | str] = {
            "attack": round(rng.uniform(0.001, 1.2), 4),
            "decay": round(rng.uniform(0.03, 1.5), 4),
            "sustain": round(rng.uniform(0.15, 1.0), 4),
            "release": round(rng.uniform(0.02, 2.5), 4),
            "filter_cutoff_hz": round(rng.uniform(250, 14_000), 2),
            "filter_resonance": round(rng.uniform(0.05, 0.9), 4),
            "lfo_rate_hz": round(rng.uniform(0.05, 12.0), 4),
            "lfo_depth": round(rng.uniform(0.0, 1.0), 4),
        }
        if engine == "fm":
            common.update({"fm_ratio": round(rng.uniform(0.25, 8.0), 4), "fm_index": round(rng.uniform(0.0, 12.0), 4)})
        elif engine == "wavetable":
            common.update({"wavetable_position": round(rng.random(), 4), "unison_detune": round(rng.uniform(0, 0.18), 4)})
        elif engine == "subtractive":
            common.update({"osc_mix": round(rng.random(), 4), "pulse_width": round(rng.uniform(0.1, 0.9), 4)})
        return common

    def _effects(self, role: str, rng: random.Random) -> list[EffectSpec]:
        effects = [
            EffectSpec("eq", {"low_gain_db": round(rng.uniform(-6, 4), 2), "high_gain_db": round(rng.uniform(-4, 6), 2)}),
            EffectSpec("compressor", {"threshold_db": round(rng.uniform(-28, -8), 2), "ratio": round(rng.uniform(1.2, 5.0), 2)}),
        ]
        if role != "drums" and rng.random() < 0.65:
            effects.append(EffectSpec("chorus", {"rate_hz": round(rng.uniform(0.1, 2.5), 3), "mix": round(rng.uniform(0.05, 0.45), 3)}))
        if rng.random() < 0.8:
            effects.append(EffectSpec("reverb", {"room_size": round(rng.uniform(0.05, 0.9), 3), "mix": round(rng.uniform(0.02, 0.35), 3)}))
        if rng.random() < 0.45:
            effects.append(EffectSpec("delay", {"time_beats": rng.choice([0.25, 0.5, 0.75]), "feedback": round(rng.uniform(0.05, 0.55), 3)}))
        if rng.random() < 0.35:
            effects.append(EffectSpec("distortion", {"drive": round(rng.uniform(0.02, 0.8), 3)}))
        return effects

    def _automation(self, duration_beats: float, rng: random.Random) -> list[AutomationSpec]:
        if rng.random() > 0.6:
            return []
        return [
            AutomationSpec(
                "filter_cutoff_hz",
                [(0.0, round(rng.uniform(300, 4000), 2)), (duration_beats, round(rng.uniform(1000, 14_000), 2))],
            )
        ]

    def _add_sidechains(self, tracks: list[TrackSpec]) -> list[TrackSpec]:
        drum_ids = [track.track_id for track in tracks if track.role == "drums"]
        if not drum_ids:
            return tracks
        updated = []
        for track in tracks:
            if track.role in {"bass", "harmony"}:
                updated.append(
                    TrackSpec(
                        **{**track.__dict__, "sidechain_from": drum_ids[:1]},
                    )
                )
            else:
                updated.append(track)
        return updated


def _role_scale_pitches(role: str, key: str, scale: str, rng: random.Random) -> list[int]:
    root = ROOT_MIDI[key]
    degrees = SCALES[scale]
    ranges = {
        "bass": (34, 55),
        "harmony": (45, 76),
        "melody": (50, 84),
    }
    preferred_centers = {
        "bass": (40, 43, 45, 47, 50),
        "harmony": (55, 57, 60, 62, 64, 67),
        "melody": (60, 62, 64, 67, 69, 72),
    }
    max_span = {"bass": 12, "harmony": 16, "melody": 18}.get(role, 16)
    low, high = ranges.get(role, (45, 84))
    center = rng.choice(preferred_centers.get(role, (57, 60, 62, 64, 67)))
    window_low = max(low, center - max_span // 2)
    window_high = min(high, center + max_span // 2)
    pitches = []
    for octave in range(-3, 4):
        octave_root = root + octave * 12
        for degree in degrees:
            pitch = octave_root + degree
            if window_low <= pitch <= window_high:
                pitches.append(pitch)
    if pitches:
        return pitches
    return [max(low, min(high, center))]
