from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints


@dataclass
class DatasetConfig:
    root: str = "../dataset generator/generated_vst"
    sample_rate: int = 44_100
    segment_seconds: float = 24.0
    random_crop: bool = True
    mono: bool = False


@dataclass
class ModelConfig:
    sample_rate: int = 44_100
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    hidden_dim: int = 128
    encoder_channels: int = 64
    decoder_channels: int = 32
    max_steps: int = 8
    stop_threshold: float = 0.5
    refine_channels: int = 32


@dataclass
class TrainingConfig:
    output_dir: str = "runs/default"
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 5
    max_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    amp: bool = True
    log_every: int = 10
    checkpoint_every: int = 100
    resume: str | None = None


@dataclass
class LossConfig:
    waveform_l1: float = 1.0
    spectral: float = 1.0
    si_sdr: float = 0.5
    stop: float = 0.2
    residual_energy: float = 0.05
    source_impurity: float = 0.1
    spectral_chunk_size: int = 8


@dataclass
class DiscriminatorConfig:
    enabled: bool = False
    checkpoint: str | None = None
    channels: int = 32
    hidden_dim: int = 128
    negative_gain_min_db: float = -4.0
    negative_gain_max_db: float = 4.0
    max_relative_gain_db: float = 8.0
    three_stem_probability: float = 0.15
    target_rms: float = 0.05
    label_chunk_size: int = 16


@dataclass
class ExperimentConfig:
    seed: int = 1234
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)


T = TypeVar("T")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _from_dict(ExperimentConfig, data)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
        handle.write("\n")


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.json", help="Path to JSON experiment config.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset.root.")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir.")
    parser.add_argument("--sample-rate", type=int, default=None, help="Override dataset.sample_rate and model.sample_rate.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if getattr(args, "dataset_root", None):
        config.dataset.root = args.dataset_root
    if getattr(args, "output_dir", None):
        config.training.output_dir = args.output_dir
    if getattr(args, "sample_rate", None):
        config.dataset.sample_rate = args.sample_rate
        config.model.sample_rate = args.sample_rate
    return config


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name in data:
            value = data[item.name]
        elif item.default is not MISSING:
            value = item.default
        elif item.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            value = item.default_factory()  # type: ignore[misc]
        else:
            raise KeyError(f"missing required config field {item.name}")
        target_type = type_hints[item.name]
        if is_dataclass(target_type) and isinstance(value, dict):
            kwargs[item.name] = _from_dict(target_type, value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)

