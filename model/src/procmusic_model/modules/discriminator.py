from __future__ import annotations

import torch
from torch import nn

from procmusic_model.audio import STFTConfig, STFTFrontend
from procmusic_model.config import ModelConfig


class SingleSourceDiscriminator(nn.Module):
    """Predicts how much non-primary source energy is present in an audio segment."""

    def __init__(
        self,
        model_config: ModelConfig,
        audio_channels: int,
        channels: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.frontend = STFTFrontend(STFTConfig(model_config.n_fft, model_config.hop_length, model_config.win_length))
        feature_channels = audio_channels * 3
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden_dim, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        features, _ = self.frontend.features(audio)
        encoded = self.net(features)
        pooled = self.pool(encoded).flatten(1)
        return self.classifier(pooled).squeeze(-1)


def weighted_stft_power(audio: torch.Tensor, model_config: ModelConfig, eps: float = 1e-8) -> torch.Tensor:
    if audio.ndim != 3:
        raise ValueError(f"expected audio shape (batch, channels, samples), got {tuple(audio.shape)}")
    batch, channels, samples = audio.shape
    window = torch.hann_window(model_config.win_length, device=audio.device, dtype=audio.dtype)
    flat = audio.reshape(batch * channels, samples)
    spec = torch.stft(
        flat,
        n_fft=model_config.n_fft,
        hop_length=model_config.hop_length,
        win_length=model_config.win_length,
        window=window,
        return_complex=True,
    )
    spec = spec.reshape(batch, channels, spec.shape[-2], spec.shape[-1])
    freqs = torch.linspace(0.0, model_config.sample_rate / 2.0, spec.shape[-2], device=audio.device, dtype=audio.dtype)
    weights = a_weighting_power_weights(freqs, eps).view(1, 1, -1, 1)
    return (spec.abs().pow(2) * weights).mean(dim=(1, 2, 3))


def a_weighting_power_weights(freqs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    f2 = freqs.clamp_min(eps).pow(2)
    ra_num = (12_200.0**2) * f2.pow(2)
    ra_den = (f2 + 20.6**2) * torch.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12_200.0**2)
    a_db = (20.0 * torch.log10(ra_num / ra_den.clamp_min(eps)) + 2.0).clamp_min(-80.0)
    weights = 10.0 ** (a_db / 10.0)
    return weights / weights.max().clamp_min(eps)


def normalize_rms(audio: torch.Tensor, target_rms: float, eps: float = 1e-8) -> torch.Tensor:
    rms = audio.pow(2).mean(dim=tuple(range(1, audio.ndim)), keepdim=True).sqrt()
    return audio * (target_rms / rms.clamp_min(eps))
