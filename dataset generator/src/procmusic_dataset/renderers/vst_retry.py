from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from procmusic_dataset.models import NoteEvent, ProjectSpec, TrackSpec
from procmusic_dataset.renderers.vst_audio import (
    StereoAudio,
    apply_track_controls,
    fit_stereo_length,
    is_silent,
)
from procmusic_dataset.renderers.vst_backend import VSTBackend, unpack_backend_result


@dataclass(frozen=True)
class ResolvedPitchedTrack:
    track: TrackSpec
    audio: StereoAudio
    details: dict


def render_pitched_with_retries(
    backend: VSTBackend,
    plugin_path: Path,
    project: ProjectSpec,
    track: TrackSpec,
    sample_rate: int,
    block_size: int,
    total_samples: int,
) -> ResolvedPitchedTrack:
    attempts = []
    for attempt in range(3):
        attempt_track = retune_track_for_attempt(project, track, attempt)
        backend_result = backend.render_pitched_track(
            plugin_path,
            project,
            attempt_track,
            sample_rate,
            block_size,
            attempt=attempt,
        )
        audio, plugin_details = unpack_backend_result(backend_result)
        audio = fit_stereo_length(audio, total_samples)
        silent = is_silent(audio)
        if not silent:
            audio = apply_track_controls(project, attempt_track, audio, sample_rate)
        attempts.append(
            {
                "attempt": attempt,
                "midi_regenerated": attempt > 0,
                "silent": silent,
                "preset_or_program": plugin_details.get("preset_or_program"),
                "patch_selection_status": plugin_details.get("patch_selection_status"),
            }
        )
        if not silent:
            return ResolvedPitchedTrack(
                attempt_track,
                audio,
                {
                    **plugin_details,
                    "render_attempts": attempts,
                    "midi_regenerated": attempt > 0,
                    "rendered_midi_notes": [note.__dict__ for note in attempt_track.notes],
                },
            )

    fallback_track = retune_track_for_attempt(project, track, 3)
    audio = render_fallback_pitched_track(project, fallback_track, total_samples, sample_rate)
    audio = apply_track_controls(project, fallback_track, audio, sample_rate)
    return ResolvedPitchedTrack(
        fallback_track,
        audio,
        {
            "preset_or_program": None,
            "patch_selection_status": "internal_fallback_after_silent_retries",
            "available_preset_count": 0,
            "available_parameter_count": 0,
            "parameter_randomization": [],
            "render_fallback": "internal_pitched_synth_after_silent_vst_retries",
            "render_attempts": attempts,
            "midi_regenerated": True,
            "rendered_midi_notes": [note.__dict__ for note in fallback_track.notes],
        },
    )


def retune_track_for_attempt(project: ProjectSpec, track: TrackSpec, attempt: int) -> TrackSpec:
    if attempt == 0 and track.notes:
        return track
    rng = random.Random(f"{project.seed}:{track.track_id}:retry-midi:{attempt}")
    notes = track.notes if track.notes and attempt == 0 else regenerate_role_notes(track.role, project.duration_beats, rng)
    if not notes:
        notes = regenerate_role_notes(track.role, project.duration_beats, rng)
    if attempt <= 1:
        return TrackSpec(**{**track.__dict__, "notes": notes})

    low, high = safe_pitch_range(track.role)
    pitches = [note.pitch for note in notes]
    if pitches:
        current_center = sum(pitches) / len(pitches)
        target_center = {"bass": 43, "harmony": 60, "melody": 67}.get(track.role, 60)
        shift = round((target_center - current_center) / 12.0) * 12
        notes = [
            NoteEvent(max(low, min(high, note.pitch + shift)), note.start_beat, note.duration_beats, note.velocity)
            for note in notes
        ]
    return TrackSpec(**{**track.__dict__, "notes": notes})


def regenerate_role_notes(role: str, duration_beats: float, rng: random.Random) -> list[NoteEvent]:
    low, high = safe_pitch_range(role)
    center = {"bass": 43, "harmony": 60, "melody": 67}.get(role, 60)
    span = {"bass": 10, "harmony": 14, "melody": 16}.get(role, 14)
    scale = [0, 2, 3, 5, 7, 10] if rng.random() < 0.5 else [0, 2, 4, 5, 7, 9]
    candidates = []
    for pitch in range(max(low, center - span // 2), min(high, center + span // 2) + 1):
        if (pitch - center) % 12 in scale:
            candidates.append(pitch)
    if not candidates:
        candidates = [center]
    step = {"bass": 1.0, "harmony": 2.0, "melody": 0.5}.get(role, 1.0)
    duration = {"bass": 0.75, "harmony": 1.75, "melody": 0.45}.get(role, 0.75)
    notes = []
    beat = 0.0
    while beat < duration_beats:
        if rng.random() < 0.9:
            notes.append(NoteEvent(rng.choice(candidates), round(beat, 3), min(duration, duration_beats - beat), rng.randint(70, 112)))
        beat += step
    if not notes:
        notes.append(NoteEvent(candidates[len(candidates) // 2], 0.0, min(duration, duration_beats), 96))
    return notes


def safe_pitch_range(role: str) -> tuple[int, int]:
    return {
        "bass": (34, 55),
        "harmony": (45, 76),
        "melody": (50, 84),
    }.get(role, (45, 84))


def render_fallback_pitched_track(
    project: ProjectSpec, track: TrackSpec, total_samples: int, sample_rate: int
) -> StereoAudio:
    audio = [[0.0] * total_samples, [0.0] * total_samples]
    rng = random.Random(f"{project.seed}:{track.track_id}:fallback-pitched")
    waveform = rng.choice(("sine", "triangle", "saw"))
    amp = 0.16
    for note in track.notes:
        start = int(note.start_beat * 60.0 / project.bpm * sample_rate)
        length = max(1, int(note.duration_beats * 60.0 / project.bpm * sample_rate))
        frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
        velocity = note.velocity / 127.0
        for offset in range(length):
            index = start + offset
            if index >= total_samples:
                break
            phase = (frequency * offset / sample_rate) % 1.0
            if waveform == "triangle":
                sample = 4.0 * abs(phase - 0.5) - 1.0
            elif waveform == "saw":
                sample = 2.0 * phase - 1.0
            else:
                sample = math.sin(2.0 * math.pi * phase)
            attack = min(1.0, offset / max(1, int(sample_rate * 0.01)))
            release = min(1.0, (length - offset) / max(1, int(sample_rate * 0.03)))
            env = max(0.0, min(attack, release))
            value = sample * env * amp * velocity
            audio[0][index] += value
            audio[1][index] += value
    return audio
