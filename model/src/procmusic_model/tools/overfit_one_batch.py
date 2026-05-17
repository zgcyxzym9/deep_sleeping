from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from procmusic_model.config import add_common_args, apply_overrides, load_config
from procmusic_model.data import MusicSeparationDataset, collate_separation_batch
from procmusic_model.losses import SeparationLoss
from procmusic_model.systems import OpenSetSeparator
from procmusic_model.training.checkpoint import save_checkpoint
from procmusic_model.training.train import _load_discriminator, _log, _make_writer


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint", default=None, help="Path to save the overfit checkpoint.")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MusicSeparationDataset(config.dataset.root, config.dataset.sample_rate, config.dataset.segment_seconds, False, config.dataset.mono)
    batch = next(iter(DataLoader(dataset, batch_size=config.training.batch_size, collate_fn=collate_separation_batch))).to(device)
    _print_batch_summary(batch.metadata)
    model = OpenSetSeparator(config.model, channels=batch.mixture.shape[1]).to(device)
    discriminator = _load_discriminator(config, batch.mixture.shape[1], device)
    criterion = SeparationLoss(
        config.loss,
        config.model,
        discriminator,
        config.discriminator.target_rms,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    writer = _make_writer()
    checkpoint_path = Path(args.checkpoint or Path(config.training.output_dir) / "checkpoints" / "overfit_one_batch.pt")

    try:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_train(batch)
            loss_output = criterion(output, batch)
            loss_output.total.backward()
            optimizer.step()
            _log(writer, loss_output.terms, output, batch, step, config.dataset.sample_rate)
            print(f"step={step} loss={float(loss_output.total.detach().cpu()):.6f}")
        save_checkpoint(checkpoint_path, config, model, optimizer, epoch=0, step=args.steps)
        print(f"saved checkpoint: {checkpoint_path}")
    finally:
        writer.close()


def _print_batch_summary(metadata: list[dict]) -> None:
    print("overfit batch:")
    for index, item in enumerate(metadata):
        tracks = item.get("tracks", [])
        render = item.get("render", {})
        stems = render.get("stems", [])
        print(f"  sample={index} project_id={item.get('project_id')} source_count={len(stems)}")
        print(f"    mixture={render.get('mixture_path')}")
        for track_index, track in enumerate(tracks):
            stem_path = stems[track_index].get("stem_path") if track_index < len(stems) else None
            name = track.get("name")
            role = track.get("role")
            category = track.get("instrument_category")
            print(f"    source={track_index} name={name} role={role} category={category} stem={stem_path}")


if __name__ == "__main__":
    main()
