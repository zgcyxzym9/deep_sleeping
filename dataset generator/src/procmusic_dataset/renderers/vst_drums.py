from __future__ import annotations

import math

from procmusic_dataset.models import ProjectSpec, TrackSpec
from procmusic_dataset.renderers.vst_audio import StereoAudio


def render_drum_track(project: ProjectSpec, track: TrackSpec, total_samples: int, sample_rate: int) -> StereoAudio:
    audio = [[0.0] * total_samples, [0.0] * total_samples]
    amp = 10 ** (track.gain_db / 20.0) * 0.35
    for note in track.notes:
        start = int(note.start_beat * 60.0 / project.bpm * sample_rate)
        length = max(1, int(note.duration_beats * 60.0 / project.bpm * sample_rate))
        velocity = note.velocity / 127.0
        for offset in range(length):
            index = start + offset
            if index >= total_samples:
                break
            env = math.exp(-6.0 * offset / length)
            if note.pitch == 36:
                sample = math.sin(2 * math.pi * (55 + 80 * env) * offset / sample_rate)
            elif note.pitch == 38:
                sample = math.sin(2 * math.pi * 180 * offset / sample_rate) * 0.4 + noise(index) * 0.6
            else:
                sample = noise(index) * 0.8
            value = sample * env * amp * velocity
            audio[0][index] += value
            audio[1][index] += value
    return audio


def noise(index: int) -> float:
    value = (index * 1103515245 + 12345) & 0x7FFFFFFF
    return (value / 0x3FFFFFFF) - 1.0
