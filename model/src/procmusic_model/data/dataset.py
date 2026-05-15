from __future__ import annotations

import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


JsonDict = dict[str, Any]


@dataclass
class SeparationExample:
    mixture: torch.Tensor
    sources: torch.Tensor
    source_count: int
    metadata: JsonDict


@dataclass
class SeparationBatch:
    mixture: torch.Tensor
    sources: torch.Tensor
    source_mask: torch.Tensor
    source_count: torch.Tensor
    metadata: list[JsonDict]

    def to(self, device: torch.device | str) -> "SeparationBatch":
        return SeparationBatch(
            mixture=self.mixture.to(device),
            sources=self.sources.to(device),
            source_mask=self.source_mask.to(device),
            source_count=self.source_count.to(device),
            metadata=self.metadata,
        )


class MusicSeparationDataset(torch.utils.data.Dataset[SeparationExample]):
    """Reads generated procedural music projects without depending on the generator package."""

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 44_100,
        segment_seconds: float | None = 24.0,
        random_crop: bool = True,
        mono: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.sample_rate = sample_rate
        self.segment_samples = None if segment_seconds is None else int(round(segment_seconds * sample_rate))
        self.random_crop = random_crop
        self.mono = mono
        self.projects = self._discover_projects()
        if not self.projects:
            raise FileNotFoundError(f"no rendered projects found under {self.root}")

    def __len__(self) -> int:
        return len(self.projects)

    def __getitem__(self, index: int) -> SeparationExample:
        metadata_path = self.projects[index]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mixture = _load_wav(self._resolve_artifact(metadata["render"]["mixture_path"]), self.sample_rate)
        stems = [
            _load_wav(self._resolve_artifact(stem["stem_path"]), self.sample_rate)
            for stem in metadata["render"]["stems"]
        ]
        if self.mono:
            mixture = mixture.mean(dim=0, keepdim=True)
            stems = [stem.mean(dim=0, keepdim=True) for stem in stems]
        mixture, sources = self._align_and_crop(mixture, stems)
        return SeparationExample(
            mixture=mixture,
            sources=sources,
            source_count=sources.shape[0],
            metadata=metadata,
        )

    def _discover_projects(self) -> list[Path]:
        manifest = self.root / "manifest.jsonl"
        fallback = sorted(self.root.glob("project_*/metadata.json"))
        if not manifest.exists():
            return fallback

        paths: list[Path] = []
        seen: set[str] = set()
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                project_id = record.get("project_id")
                if record.get("status") == "rendered" and project_id and project_id not in seen:
                    candidate = self.root / project_id / "metadata.json"
                    if candidate.exists():
                        paths.append(candidate)
                        seen.add(project_id)
        except (OSError, json.JSONDecodeError):
            return fallback
        return paths

    def _resolve_artifact(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return path
        candidates = [
            self.root / path,
            self.root.parent / path,
            self.root / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"artifact path not found: {value}")

    def _align_and_crop(self, mixture: torch.Tensor, stems: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        channels = mixture.shape[0]
        stems = [_match_channels(stem, channels) for stem in stems]
        total = min([mixture.shape[-1], *[stem.shape[-1] for stem in stems]])
        mixture = mixture[..., :total]
        stems = [stem[..., :total] for stem in stems]
        if self.segment_samples is not None:
            target = self.segment_samples
            if total < target:
                mixture = _pad_audio(mixture, target)
                stems = [_pad_audio(stem, target) for stem in stems]
            elif total > target:
                start = random.randint(0, total - target) if self.random_crop else 0
                mixture = mixture[..., start : start + target]
                stems = [stem[..., start : start + target] for stem in stems]
        return mixture.contiguous(), torch.stack(stems, dim=0).contiguous()


def collate_separation_batch(examples: list[SeparationExample]) -> SeparationBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    max_sources = max(example.source_count for example in examples)
    channels = max(example.mixture.shape[0] for example in examples)
    samples = max(example.mixture.shape[-1] for example in examples)
    batch_size = len(examples)

    mixture = torch.zeros(batch_size, channels, samples)
    sources = torch.zeros(batch_size, max_sources, channels, samples)
    source_mask = torch.zeros(batch_size, max_sources, dtype=torch.bool)
    source_count = torch.zeros(batch_size, dtype=torch.long)
    metadata: list[JsonDict] = []

    for idx, example in enumerate(examples):
        mix = _match_channels(example.mixture, channels)
        mixture[idx, :, : mix.shape[-1]] = mix
        src = torch.stack([_match_channels(item, channels) for item in example.sources], dim=0)
        sources[idx, : src.shape[0], :, : src.shape[-1]] = src
        source_mask[idx, : src.shape[0]] = True
        source_count[idx] = src.shape[0]
        metadata.append(example.metadata)

    return SeparationBatch(mixture, sources, source_mask, source_count, metadata)


def _load_wav(path: Path, expected_sample_rate: int) -> torch.Tensor:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_rate != expected_sample_rate:
        raise ValueError(f"{path} sample rate {sample_rate} != expected {expected_sample_rate}")
    if sample_width != 2:
        raise ValueError(f"{path} uses {sample_width * 8}-bit samples; only 16-bit PCM is supported")
    audio = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
    return audio.view(-1, channels).transpose(0, 1).contiguous()


def _pad_audio(audio: torch.Tensor, target_samples: int) -> torch.Tensor:
    if audio.shape[-1] >= target_samples:
        return audio
    padded = torch.zeros(*audio.shape[:-1], target_samples, dtype=audio.dtype)
    padded[..., : audio.shape[-1]] = audio
    return padded


def _match_channels(audio: torch.Tensor, channels: int) -> torch.Tensor:
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1:
        return audio.expand(channels, -1)
    if channels == 1:
        return audio.mean(dim=0, keepdim=True)
    raise ValueError(f"cannot convert {audio.shape[0]} channels to {channels}")


