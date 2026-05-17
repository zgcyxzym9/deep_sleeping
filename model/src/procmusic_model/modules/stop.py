from __future__ import annotations

import torch
from torch import nn


class StopPredictor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.net(summary).squeeze(-1)
