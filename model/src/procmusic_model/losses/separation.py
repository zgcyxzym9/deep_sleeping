from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from procmusic_model.config import LossConfig, ModelConfig
from procmusic_model.modules import normalize_rms


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(dim=-1, keepdim=True) * target / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    noise = estimate - projection
    ratio = projection.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps).mean()


@dataclass
class LossOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    matched_targets: torch.Tensor


class SeparationLoss(nn.Module):
    def __init__(
        self,
        loss_config: LossConfig,
        model_config: ModelConfig,
        discriminator: nn.Module | None = None,
        discriminator_target_rms: float = 0.05,
    ) -> None:
        super().__init__()
        self.loss_config = loss_config
        self.model_config = model_config
        self.discriminator = discriminator
        self.discriminator_target_rms = discriminator_target_rms

    def forward(self, output, batch) -> LossOutput:
        from .matching import greedy_match

        estimates = output.estimated_sources
        matched, _ = greedy_match(estimates, batch.sources, batch.source_mask)
        valid_steps = _valid_estimate_mask(batch.source_count, estimates.shape[1]).to(estimates.device)

        waveform_l1 = _masked_mean((estimates - matched).abs().mean(dim=(2, 3)), valid_steps)
        spectral = _spectral_l1(estimates, matched, valid_steps, self.model_config, self.loss_config.spectral_chunk_size)
        si_sdr_loss = -_masked_si_sdr(estimates, matched, valid_steps)
        stop_loss = _stop_loss(output.stop_logits, batch.source_count)
        residual_energy = output.final_residual.pow(2).mean()
        source_impurity = _source_impurity_loss(estimates, valid_steps, self.discriminator, self.discriminator_target_rms)

        terms = {
            "waveform_l1": waveform_l1,
            "spectral": spectral,
            "si_sdr": si_sdr_loss,
            "stop": stop_loss,
            "residual_energy": residual_energy,
        }
        if self.discriminator is not None:
            terms["source_impurity"] = source_impurity
        total = sum(getattr(self.loss_config, name) * terms[name] for name in ("waveform_l1", "spectral", "si_sdr", "stop", "residual_energy"))
        if self.discriminator is not None:
            total = total + self.loss_config.source_impurity * source_impurity
        terms["total"] = total
        return LossOutput(total=total, terms=terms, matched_targets=matched)


def _valid_estimate_mask(source_count: torch.Tensor, steps: int) -> torch.Tensor:
    indices = torch.arange(steps, device=source_count.device).unsqueeze(0)
    return indices < source_count.unsqueeze(1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _masked_si_sdr(estimates: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = []
    for bidx in range(estimates.shape[0]):
        for step in range(estimates.shape[1]):
            if bool(mask[bidx, step]):
                values.append(si_sdr(estimates[bidx, step : step + 1], targets[bidx, step : step + 1]))
    if not values:
        return estimates.new_tensor(0.0)
    return torch.stack(values).mean()


def _spectral_l1(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    config: ModelConfig,
    chunk_size: int,
) -> torch.Tensor:
    batch, steps, channels, samples = estimates.shape
    flat_est = estimates.reshape(batch * steps * channels, samples)
    flat_tgt = targets.reshape(batch * steps * channels, samples)
    window = torch.hann_window(config.win_length, device=estimates.device)
    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, flat_est.shape[0], chunk_size):
        end = min(start + chunk_size, flat_est.shape[0])
        est_spec = torch.stft(flat_est[start:end], config.n_fft, config.hop_length, config.win_length, window, return_complex=True).abs()
        tgt_spec = torch.stft(flat_tgt[start:end], config.n_fft, config.hop_length, config.win_length, window, return_complex=True).abs()
        chunks.append((torch.log1p(est_spec) - torch.log1p(tgt_spec)).abs().mean(dim=(1, 2)))
    values = torch.cat(chunks, dim=0)
    values = values.reshape(batch, steps, channels).mean(dim=2)
    return _masked_mean(values, mask)


def _source_impurity_loss(
    estimates: torch.Tensor,
    mask: torch.Tensor,
    discriminator: nn.Module | None,
    target_rms: float,
) -> torch.Tensor:
    if discriminator is None:
        return estimates.new_tensor(0.0)
    valid = estimates[mask]
    if valid.numel() == 0:
        return estimates.new_tensor(0.0)
    logits = discriminator(normalize_rms(valid, target_rms))
    targets = torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, targets)


def _stop_loss(stop_logits: torch.Tensor, source_count: torch.Tensor) -> torch.Tensor:
    batch, steps_plus_stop = stop_logits.shape
    targets = torch.zeros_like(stop_logits)
    mask = torch.zeros_like(stop_logits, dtype=torch.bool)
    for bidx, count in enumerate(source_count.tolist()):
        stop_index = min(int(count), steps_plus_stop - 1)
        targets[bidx, stop_index] = 1.0
        mask[bidx, : stop_index + 1] = True
    values = F.binary_cross_entropy_with_logits(stop_logits, targets, reduction="none")
    return values[mask].mean()
