from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GenerationConfig:
    min_tracks: int = 3
    max_tracks: int = 8
    min_bpm: int = 70
    max_bpm: int = 170
    min_bars: int = 8
    max_bars: int = 24
    sample_rate: int = 44_100
    bit_depth: int = 16
    allow_drums: bool = True
    allow_synths: bool = True
    allow_acoustic: bool = True

    def validate(self) -> None:
        if self.min_tracks < 1 or self.max_tracks < self.min_tracks:
            raise ValueError("track range must be positive and ordered")
        if self.min_bpm < 30 or self.max_bpm < self.min_bpm:
            raise ValueError("BPM range is invalid")
        if self.min_bars < 1 or self.max_bars < self.min_bars:
            raise ValueError("bar range must be positive and ordered")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")


@dataclass(frozen=True)
class BatchConfig:
    output_dir: Path
    count: int
    seed: int
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    overwrite: bool = False

    def validate(self) -> None:
        if self.count < 1:
            raise ValueError("count must be at least 1")
        self.generation.validate()
