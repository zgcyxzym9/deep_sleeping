from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from procmusic_model.audio import STFTConfig, STFTFrontend
from procmusic_model.config import ModelConfig
from procmusic_model.modules import AudioEncoder, MaskDecoder, RestorationRefiner, SourceDiscovery


@dataclass
class SeparatorOutput:
    estimated_sources: torch.Tensor
    stop_logits: torch.Tensor
    final_residual: torch.Tensor
    residuals: torch.Tensor | None
    loss_terms: dict[str, torch.Tensor]


class OpenSetSeparator(nn.Module):
    def __init__(self, config: ModelConfig, channels: int = 1) -> None:
        super().__init__()
        self.config = config
        self.channels = channels
        self.frontend = STFTFrontend(STFTConfig(config.n_fft, config.hop_length, config.win_length))
        feature_channels = channels * 3
        self.encoder = AudioEncoder(feature_channels, config.encoder_channels, config.hidden_dim)
        self.discovery = SourceDiscovery(config.hidden_dim, config.max_steps)
        self.decoder = MaskDecoder(
            feature_channels,
            config.hidden_dim,
            config.decoder_channels,
            output_channels=channels,
        )
        self.refiner = RestorationRefiner(channels=channels, hidden_channels=config.refine_channels)

    def forward_train(self, batch, return_residuals: bool = False) -> SeparatorOutput:
        return self._run(batch.mixture, stop_threshold=None, return_residuals=return_residuals)

    @torch.no_grad()
    def separate(self, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self._run(mixture, stop_threshold=self.config.stop_threshold, return_residuals=True)
        stop_prob = torch.sigmoid(output.stop_logits)
        counts = []
        for row in stop_prob:
            stop_indices = (row > self.config.stop_threshold).nonzero(as_tuple=False)
            counts.append(int(stop_indices[0].item()) if len(stop_indices) else self.config.max_steps)
        return {
            "sources": output.estimated_sources,
            "stop_logits": output.stop_logits,
            "stop_prob": stop_prob,
            "predicted_source_count": torch.tensor(counts, device=mixture.device),
            "residuals": output.residuals,
        }

    def _run(
        self,
        mixture: torch.Tensor,
        stop_threshold: float | None,
        return_residuals: bool,
    ) -> SeparatorOutput:
        features, _ = self.frontend.features(mixture)
        mix_embedding = self.encoder(features)
        queries, stop_logits = self.discovery(mix_embedding)

        residual = mixture
        estimates = []
        residuals = [residual] if return_residuals else None
        for step in range(self.config.max_steps):
            residual_features, residual_spec = self.frontend.features(residual)
            mask = self.decoder(residual_features, queries[:, step])
            rough_spec = residual_spec * mask
            rough = self.frontend.istft(rough_spec, mixture.shape[-1])
            refined = self.refiner(rough, residual, mixture)
            estimates.append(refined)
            residual = residual - refined
            if residuals is not None:
                residuals.append(residual)

            if stop_threshold is not None:
                should_stop = torch.sigmoid(stop_logits[:, step + 1]) > stop_threshold
                if bool(should_stop.all()):
                    break

        estimated = torch.stack(estimates, dim=1)
        residual_tensor = torch.stack(residuals, dim=1) if residuals is not None else None
        return SeparatorOutput(estimated, stop_logits, residual, residual_tensor, {})
