from __future__ import annotations

import torch
from torch import nn


class MaskDecoder(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        hidden_dim: int,
        decoder_channels: int,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(feature_channels, decoder_channels, kernel_size=1)
        self.condition = nn.Linear(hidden_dim, decoder_channels)
        self.net = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, output_channels, kernel_size=1),
        )

    def forward(self, residual_features: torch.Tensor, source_query: torch.Tensor) -> torch.Tensor:
        conditioned = self.input(residual_features) + self.condition(source_query).unsqueeze(-1).unsqueeze(-1)
        return torch.sigmoid(self.net(conditioned))
