from __future__ import annotations

import torch
from torch import nn


class RestorationRefiner(nn.Module):
    def __init__(self, channels: int = 1, hidden_channels: int = 32) -> None:
        super().__init__()
        in_channels = channels * 3
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, rough_source: torch.Tensor, residual: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([rough_source, residual, mixture], dim=1))
        return rough_source + 0.1 * torch.tanh(delta)
