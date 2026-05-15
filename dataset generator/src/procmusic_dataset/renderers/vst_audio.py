from __future__ import annotations

import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from procmusic_dataset.models import ProjectSpec, TrackSpec


StereoAudio = Union[np.ndarray, list[list[float]]]


@dataclass(frozen=True)
class PendingStem:
    track: TrackSpec
    audio: StereoAudio
    details: dict


def project_duration_seconds(project: ProjectSpec) -> float:
    return project.duration_beats * 60.0 / project.bpm


def coerce_stereo(audio: object) -> StereoAudio:
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        raise RuntimeError("DawDreamer returned empty audio")
    if array.ndim == 1:
        return np.stack([array, array.copy()])
    if array.shape[0] == 1:
        return np.repeat(array[:1], 2, axis=0)
    if array.shape[0] >= 2:
        return array[:2].copy()
    if array.shape[1] >= 2:
        return array[:, :2].T.copy()
    return np.repeat(array.T[:1], 2, axis=0)


def fit_stereo_length(audio: StereoAudio, total_samples: int) -> StereoAudio:
    array = coerce_stereo(audio)
    if array.shape[1] < total_samples:
        return np.pad(array, ((0, 0), (0, total_samples - array.shape[1])))
    return array[:, :total_samples].copy()


def add_to_mixture(mixture: StereoAudio, audio: StereoAudio) -> None:
    mixture += coerce_stereo(audio)


def normalize_project_loudness(
    project: ProjectSpec,
    pending_stems: list[PendingStem],
    mixture: StereoAudio,
) -> tuple[list[PendingStem], StereoAudio, dict]:
    import random

    rng = random.Random(f"{project.seed}:mixture-loudness")
    target_dbfs = round(rng.uniform(-19.0, -17.0), 3)
    input_rms = rms(mixture)
    input_dbfs = 20 * math.log10(max(input_rms, 1e-6))
    target_rms = 10 ** (target_dbfs / 20.0)
    gain = target_rms / max(input_rms, 1e-6)
    peak = max(1e-9, float(np.max(np.abs(coerce_stereo(mixture)))))
    gain = min(gain, 0.98 / peak)
    scaled_mixture = scale_audio(mixture, gain)
    scaled_stems = [PendingStem(stem.track, scale_audio(stem.audio, gain), stem.details) for stem in pending_stems]
    return scaled_stems, scaled_mixture, {
        "target_dbfs": target_dbfs,
        "input_dbfs": round(input_dbfs, 3),
        "applied_gain_db": round(20 * math.log10(max(gain, 1e-9)), 3),
    }


def apply_track_controls(project: ProjectSpec, track: TrackSpec, audio: StereoAudio, sample_rate: int) -> StereoAudio:
    import random

    rng = random.Random(f"{project.seed}:{track.track_id}:vst-controls")
    gain = 10 ** (track.gain_db / 20.0)
    pan = max(-1.0, min(1.0, track.pan))
    left_gain = gain * (1.0 - max(0.0, pan) * 0.5)
    right_gain = gain * (1.0 + min(0.0, pan) * 0.5)
    processed = coerce_stereo(audio).copy()
    processed[0] *= left_gain
    processed[1] *= right_gain

    if any(effect.name == "distortion" for effect in track.effects):
        drive = 1.0 + effect_value(track, "distortion", "drive", 0.2) * 8.0
        processed = np.tanh(processed * drive) / math.tanh(drive)
    if any(effect.name == "delay" for effect in track.effects):
        delay_beats = effect_value(track, "delay", "time_beats", 0.5)
        feedback = min(0.75, effect_value(track, "delay", "feedback", 0.25))
        delay_samples = max(1, int(delay_beats * 60.0 / project.bpm * sample_rate))
        processed = [delay(channel, delay_samples, feedback, 0.22) for channel in processed]
    if any(effect.name == "reverb" for effect in track.effects):
        mix = min(0.45, effect_value(track, "reverb", "mix", 0.18))
        processed = [simple_reverb(channel, sample_rate, mix, rng) for channel in processed]
    return normalize_if_needed(processed, ceiling=0.99)


def effect_value(track: TrackSpec, effect_name: str, parameter: str, default: float) -> float:
    for effect in track.effects:
        if effect.name == effect_name and parameter in effect.parameters:
            return float(effect.parameters[parameter])
    return default


def delay(channel: list[float], delay_samples: int, feedback: float, mix: float) -> list[float]:
    output = np.asarray(channel, dtype=np.float32).copy()
    for index in range(delay_samples, len(output)):
        output[index] += output[index - delay_samples] * feedback * mix
    return output


def simple_reverb(channel: list[float], sample_rate: int, mix: float, rng) -> list[float]:
    channel = np.asarray(channel, dtype=np.float32)
    output = channel.copy()
    delays = [int(sample_rate * value) for value in (0.031, 0.047, 0.071)]
    gains = [rng.uniform(0.08, 0.18), rng.uniform(0.04, 0.12), rng.uniform(0.03, 0.09)]
    for delay_samples, gain in zip(delays, gains):
        for index in range(delay_samples, len(output)):
            output[index] += channel[index - delay_samples] * gain * mix
    return output


def normalize_if_needed(audio: StereoAudio, ceiling: float = 0.98) -> StereoAudio:
    array = coerce_stereo(audio)
    peak = max(1e-9, float(np.max(np.abs(array))))
    if peak <= ceiling:
        return array
    scale = ceiling / peak
    return array * scale


def scale_audio(audio: StereoAudio, gain: float) -> StereoAudio:
    return coerce_stereo(audio) * gain


def is_silent(audio: StereoAudio) -> bool:
    try:
        array = coerce_stereo(audio)
    except RuntimeError:
        return True
    return array.size == 0 or float(np.max(np.abs(array))) <= 1e-8


def validate_audio(track: TrackSpec, audio: StereoAudio) -> None:
    array = coerce_stereo(audio)
    if array.shape[0] != 2 or array.shape[1] == 0:
        raise RuntimeError(f"track {track.name} rendered empty audio")
    if not bool(np.all(np.isfinite(array))):
        raise RuntimeError(f"track {track.name} rendered non-finite audio")
    if float(np.max(np.abs(array))) <= 1e-8:
        raise RuntimeError(f"track {track.name} rendered silence")


def write_wav(path: Path, audio: StereoAudio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = coerce_stereo(audio)
    pcm = (np.clip(array, -1.0, 1.0).T * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    validate_output_file(path)


def validate_output_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 44:
        raise RuntimeError(f"rendered output is missing or empty: {path}")


def write_mp3_preview(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--write-preview-mp3 requires ffmpeg to be available on PATH")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(mp3_path),
        ],
        check=True,
    )
    validate_output_file(mp3_path)


def measure(audio: StereoAudio) -> tuple[float, float, float]:
    array = coerce_stereo(audio)
    peak = float(np.max(np.abs(array)))
    audio_rms = rms(audio)
    dbfs = 20 * math.log10(max(audio_rms, 1e-6))
    return round(peak, 6), round(audio_rms, 6), round(dbfs, 3)


def rms(audio: StereoAudio) -> float:
    array = coerce_stereo(audio)
    return float(np.sqrt(np.mean(array * array)))
