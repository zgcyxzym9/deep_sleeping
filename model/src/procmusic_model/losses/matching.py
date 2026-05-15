from __future__ import annotations

import torch

from .separation import si_sdr


def greedy_match(estimates: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    """Greedy per-batch matching by highest SI-SDR.

    Returns a target tensor ordered like estimates. Unmatched estimate slots remain zero.
    """

    batch, steps, channels, samples = estimates.shape
    matched = torch.zeros_like(estimates)
    pairs: list[list[tuple[int, int]]] = []
    for bidx in range(batch):
        available = [idx for idx, valid in enumerate(target_mask[bidx].tolist()) if valid]
        batch_pairs: list[tuple[int, int]] = []
        for step in range(steps):
            if not available:
                break
            scores = torch.stack([si_sdr(estimates[bidx, step : step + 1], targets[bidx, tgt : tgt + 1]) for tgt in available])
            best_pos = int(torch.argmax(scores).item())
            target_idx = available.pop(best_pos)
            matched[bidx, step] = targets[bidx, target_idx]
            batch_pairs.append((step, target_idx))
        pairs.append(batch_pairs)
    return matched, pairs
