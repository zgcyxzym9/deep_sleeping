from __future__ import annotations

import torch
from torch import nn


class AudioEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, channels: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden_dim, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def summary(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.pool(encoded).flatten(1)
