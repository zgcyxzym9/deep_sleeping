from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from procmusic_model.config import add_common_args, apply_overrides, load_config
from procmusic_model.data import MusicSeparationDataset, collate_separation_batch
from procmusic_model.modules import SingleSourceDiscriminator
from procmusic_model.training import load_checkpoint
from procmusic_model.training.train_discriminator import _make_discriminator_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--examples", type=int, default=16)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    evaluate_discriminator(config, args.checkpoint, args.batches, args.examples, args.device)


@torch.no_grad()
def evaluate_discriminator(config, checkpoint_path: str, batches: int, examples: int, device: str | None = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MusicSeparationDataset(
        config.dataset.root,
        sample_rate=config.dataset.sample_rate,
        segment_seconds=config.dataset.segment_seconds,
        random_crop=False,
        mono=config.dataset.mono,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_separation_batch,
    )

    channels = dataset[0].mixture.shape[0]
    model = SingleSourceDiscriminator(
        config.model,
        audio_channels=channels,
        channels=config.discriminator.channels,
        hidden_dim=config.discriminator.hidden_dim,
    ).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    rows: list[tuple[float, float, float, str, float | None]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        batch = batch.to(device)
        stems = batch.sources[batch.source_mask]
        audio, targets, sample_counts = _make_discriminator_batch(stems, config.discriminator, config.model)
        logits = model(audio)
        predictions = torch.sigmoid(logits)
        for index in range(audio.shape[0]):
            target = float(targets[index].detach().cpu())
            prediction = float(predictions[index].detach().cpu())
            kind = "positive" if target == 0.0 else "negative"
            sample_count = float(sample_counts[index].detach().cpu())
            rows.append((target, prediction, abs(prediction - target), kind, sample_count))

    if not rows:
        print("no discriminator eval examples generated")
        return

    print("idx kind      target_impurity  predicted_impurity  abs_error  stems")
    print("--- --------- ---------------- ------------------- ---------- -----")
    for index, (target, prediction, error, kind, sample_count) in enumerate(rows[:examples]):
        stems_text = f"{sample_count:.0f}"
        print(f"{index:3d} {kind:<9} {target:16.4f} {prediction:19.4f} {error:10.4f} {stems_text:>5}")

    targets_tensor = torch.tensor([row[0] for row in rows])
    predictions_tensor = torch.tensor([row[1] for row in rows])
    errors_tensor = (predictions_tensor - targets_tensor).abs()
    negative_mask = targets_tensor > 0
    positive_mask = ~negative_mask
    print("")
    print(f"examples={len(rows)}")
    print(f"mae/all={float(errors_tensor.mean()):.4f}")
    if bool(positive_mask.any()):
        print(f"predicted_positive_mean={float(predictions_tensor[positive_mask].mean()):.4f}")
    if bool(negative_mask.any()):
        print(f"target_negative_mean={float(targets_tensor[negative_mask].mean()):.4f}")
        print(f"predicted_negative_mean={float(predictions_tensor[negative_mask].mean()):.4f}")
        print(f"mae/negative={float(errors_tensor[negative_mask].mean()):.4f}")


if __name__ == "__main__":
    main()
