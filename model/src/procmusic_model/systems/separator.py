from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from procmusic_model.audio import STFTConfig, STFTFrontend
from procmusic_model.config import ModelConfig
from procmusic_model.modules import AudioEncoder, MaskDecoder, RestorationRefiner, StopPredictor


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
        self.stop_predictor = StopPredictor(config.hidden_dim)
        self.decoder = MaskDecoder(
            config.hidden_dim,
            config.decoder_channels,
            output_channels=channels,
        )
        self.refiner = RestorationRefiner(channels=channels, hidden_channels=config.refine_channels)

    def forward_train(self, batch, return_residuals: bool = False) -> SeparatorOutput:
        return self._run(batch.mixture, stop_threshold=None, return_residuals=return_residuals, force_max_steps=True)

    @torch.no_grad()
    def separate(self, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
        if mixture.shape[0] != 1:
            raise ValueError("separate expects batch size 1 so stopped samples do not require padded blank sources")
        output = self._run(
            mixture,
            stop_threshold=self.config.stop_threshold,
            return_residuals=True,
            force_max_steps=False,
        )
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
        force_max_steps: bool,
    ) -> SeparatorOutput:
        residual = mixture
        estimates = []
        stop_logits = []
        residuals = [residual] if return_residuals else None
        for step in range(self.config.max_steps):
            residual_features, residual_spec = self.frontend.features(residual)
            encoded = self.encoder(residual_features)
            stop_logit = self.stop_predictor(self.encoder.summary(encoded))
            stop_logits.append(stop_logit)

            if stop_threshold is not None:
                should_stop = torch.sigmoid(stop_logit) > stop_threshold
                if bool(should_stop.item()):
                    break

            mask = self.decoder(encoded, output_size=residual_spec.shape[-2:])
            complex_mask = torch.complex(mask[:, 0::2], mask[:, 1::2])
            rough_spec = residual_spec * complex_mask
            rough = self.frontend.istft(rough_spec, mixture.shape[-1])
            refined = self.refiner(rough, residual, mixture)
            estimates.append(refined)
            residual = residual - refined
            if residuals is not None:
                residuals.append(residual)

        if force_max_steps or len(stop_logits) == self.config.max_steps:
            residual_features, _ = self.frontend.features(residual)
            encoded = self.encoder(residual_features)
            stop_logits.append(self.stop_predictor(self.encoder.summary(encoded)))

        if estimates:
            estimated = torch.stack(estimates, dim=1)
        else:
            estimated = mixture.new_empty(mixture.shape[0], 0, mixture.shape[1], mixture.shape[-1])
        stop_tensor = torch.stack(stop_logits, dim=1)
        residual_tensor = torch.stack(residuals, dim=1) if residuals is not None else None
        return SeparatorOutput(estimated, stop_tensor, residual, residual_tensor, {})
