from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from procmusic_model.config import add_common_args, apply_overrides, load_config
from procmusic_model.data import MusicSeparationDataset, collate_separation_batch
from procmusic_model.losses import greedy_match, si_sdr
from procmusic_model.systems import OpenSetSeparator
from procmusic_model.training import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--json-out", default="eval.json")
    parser.add_argument("--csv-out", default="eval.csv")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    metrics, rows = evaluate(config, args.checkpoint, device=args.device)
    Path(args.json_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with Path(args.csv_out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()) if rows else ["project_id"])
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(config, checkpoint_path: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MusicSeparationDataset(config.dataset.root, config.dataset.sample_rate, config.dataset.segment_seconds, False, config.dataset.mono)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_separation_batch)
    channels = dataset[0].mixture.shape[0]
    model = OpenSetSeparator(config.model, channels=channels).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    rows = []
    category_scores: dict[str, list[float]] = defaultdict(list)
    for batch in loader:
        batch = batch.to(device)
        result = model.separate(batch.mixture)
        estimates = result["sources"]
        matched, pairs = greedy_match(estimates, batch.sources, batch.source_mask)
        valid_scores = []
        for step, target_idx in pairs[0]:
            score = float(si_sdr(estimates[:, step], matched[:, step]).cpu())
            valid_scores.append(score)
            category = batch.metadata[0]["tracks"][target_idx]["instrument_category"]
            category_scores[category].append(score)
        pred_count = int(result["predicted_source_count"][0].cpu())
        true_count = int(batch.source_count[0].cpu())
        stop_probs = torch.sigmoid(result["stop_logits"])[0].cpu()
        rows.append(
            {
                "project_id": batch.metadata[0]["project_id"],
                "si_sdr": sum(valid_scores) / max(1, len(valid_scores)),
                "source_count": true_count,
                "predicted_source_count": pred_count,
                "source_count_error": abs(pred_count - true_count),
                "residual_rms": float(result["residuals"][0, -1].pow(2).mean().sqrt().cpu()),
                "stop_at_true_count_prob": float(stop_probs[min(true_count, len(stop_probs) - 1)]),
            }
        )

    metrics = {
        "average_si_sdr": _mean(row["si_sdr"] for row in rows),
        "average_source_count_error": _mean(row["source_count_error"] for row in rows),
        "average_residual_rms": _mean(row["residual_rms"] for row in rows),
        "per_instrument_category_si_sdr": {key: _mean(values) for key, values in category_scores.items()},
    }
    return metrics, rows


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / max(1, len(values)))


if __name__ == "__main__":
    main()
