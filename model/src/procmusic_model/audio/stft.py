from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class STFTConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024


class STFTFrontend(nn.Module):
    def __init__(self, config: STFTConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("window", torch.hann_window(config.win_length), persistent=False)

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = audio.shape
        flat = audio.reshape(batch * channels, samples)
        spec = torch.stft(
            flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            return_complex=True,
        )
        return spec.reshape(batch, channels, spec.shape[-2], spec.shape[-1])

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        batch, channels, freqs, frames = spec.shape
        flat = spec.reshape(batch * channels, freqs, frames)
        audio = torch.istft(
            flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            length=length,
        )
        return audio.reshape(batch, channels, length)

    def features(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.stft(audio)
        mag = torch.log1p(spec.abs())
        features = torch.cat([spec.real, spec.imag, mag], dim=1)
        return features, spec
