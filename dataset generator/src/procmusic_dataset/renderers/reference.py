from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from procmusic_dataset.midi import write_midi
from procmusic_dataset.models import ProjectSpec, RenderResult, RenderedTrack, TrackSpec, normalize_path
from procmusic_dataset.renderers.base import Renderer


class ReferenceRenderer(Renderer):
    """Small deterministic renderer for pipeline verification and baseline data."""

    def __init__(self, sample_rate: int = 44_100) -> None:
        self.sample_rate = sample_rate

    def render(self, project: ProjectSpec, output_dir: Path) -> RenderResult:
        stems_dir = output_dir / "stems"
        midi_dir = output_dir / "midi"
        stems_dir.mkdir(parents=True, exist_ok=True)
        midi_dir.mkdir(parents=True, exist_ok=True)

        total_samples = int(math.ceil(project.duration_beats * 60.0 / project.bpm * self.sample_rate))
        mix = [0.0] * total_samples
        rendered_tracks: list[RenderedTrack] = []

        for track in project.tracks:
            audio = self._render_track(project, track, total_samples)
            for idx, sample in enumerate(audio):
                mix[idx] += sample
            stem_path = stems_dir / f"{track.name}.wav"
            self._write_wav(stem_path, audio)
            peak, rms, dbfs = _measure(audio)
            rendered_tracks.append(RenderedTrack(track.track_id, normalize_path(stem_path), peak, rms, dbfs))

        max_abs = max(1.0, max(abs(value) for value in mix))
        if max_abs > 0.98:
            mix = [value * 0.98 / max_abs for value in mix]
        mixture_path = output_dir / "mixture.wav"
        midi_path = midi_dir / "project.mid"
        self._write_wav(mixture_path, mix)
        write_midi(project, midi_path)
        return RenderResult(normalize_path(mixture_path), normalize_path(midi_path), rendered_tracks)

    def _render_track(self, project: ProjectSpec, track: TrackSpec, total_samples: int) -> list[float]:
        audio = [0.0] * total_samples
        amp = 10 ** (track.gain_db / 20.0) * 0.25
        pan_gain = 1.0 - 0.25 * abs(track.pan)
        for note in track.notes:
            start = int(note.start_beat * 60.0 / project.bpm * self.sample_rate)
            length = max(1, int(note.duration_beats * 60.0 / project.bpm * self.sample_rate))
            velocity = note.velocity / 127.0
            if track.role == "drums":
                self._add_drum(audio, start, length, note.pitch, amp * velocity * pan_gain)
            else:
                freq = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
                self._add_tone(audio, start, length, freq, amp * velocity * pan_gain, track.synth.engine)
        return audio

    def _add_tone(self, audio: list[float], start: int, length: int, freq: float, amp: float, engine: str) -> None:
        for offset in range(length):
            idx = start + offset
            if idx >= len(audio):
                break
            t = offset / self.sample_rate
            env = min(1.0, offset / max(1, int(0.01 * self.sample_rate))) * max(0.0, 1.0 - offset / length)
            phase = 2.0 * math.pi * freq * t
            if engine == "fm":
                sample = math.sin(phase + 1.5 * math.sin(phase * 2.01))
            elif engine == "wavetable":
                sample = 0.65 * math.sin(phase) + 0.25 * math.sin(phase * 2.0) + 0.1 * math.sin(phase * 3.0)
            elif engine == "subtractive":
                sample = 1.0 if math.sin(phase) >= 0 else -1.0
            else:
                sample = math.sin(phase)
            audio[idx] += sample * env * amp

    def _add_drum(self, audio: list[float], start: int, length: int, pitch: int, amp: float) -> None:
        decay = max(1, length)
        for offset in range(decay):
            idx = start + offset
            if idx >= len(audio):
                break
            env = math.exp(-6.0 * offset / decay)
            if pitch == 36:
                sample = math.sin(2 * math.pi * (55 + 80 * env) * offset / self.sample_rate)
            elif pitch == 38:
                sample = math.sin(2 * math.pi * 180 * offset / self.sample_rate) * 0.4 + _noise(idx) * 0.6
            else:
                sample = _noise(idx) * 0.8
            audio[idx] += sample * env * amp

    def _write_wav(self, path: Path, audio: list[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            frames = bytearray()
            for sample in audio:
                clipped = max(-1.0, min(1.0, sample))
                frames += struct.pack("<h", int(clipped * 32767))
            handle.writeframes(bytes(frames))


def _noise(index: int) -> float:
    value = (index * 1103515245 + 12345) & 0x7FFFFFFF
    return (value / 0x3FFFFFFF) - 1.0


def _measure(audio: list[float]) -> tuple[float, float, float]:
    if not audio:
        return 0.0, 0.0, -120.0
    peak = max(abs(value) for value in audio)
    rms = math.sqrt(sum(value * value for value in audio) / len(audio))
    dbfs = 20 * math.log10(max(rms, 1e-6))
    return round(peak, 6), round(rms, 6), round(dbfs, 3)
