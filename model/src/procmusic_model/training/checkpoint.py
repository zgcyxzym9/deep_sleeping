from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from procmusic_model.config import ExperimentConfig, config_to_dict


def save_checkpoint(
    path: str | Path,
    config: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config_to_dict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    return checkpoint
