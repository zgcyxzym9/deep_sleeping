from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        decoder_channels: int,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(hidden_dim, decoder_channels, kernel_size=1)
        self.net = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, output_channels * 2, kernel_size=1),
        )

    def forward(self, encoded: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        mask = self.net(self.input(encoded))
        if mask.shape[-2:] != output_size:
            mask = F.interpolate(mask, size=output_size, mode="bilinear", align_corners=False)
        return torch.tanh(mask)
