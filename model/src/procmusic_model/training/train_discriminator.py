from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from procmusic_model.config import add_common_args, apply_overrides, load_config, save_config
from procmusic_model.data import MusicSeparationDataset, collate_separation_batch
from procmusic_model.modules import SingleSourceDiscriminator, normalize_rms, weighted_stft_power
from procmusic_model.training.checkpoint import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--steps", type=int, default=None, help="Override training.max_steps.")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    train_discriminator(config, steps=args.steps, device=args.device)


def train_discriminator(config, steps: int | None = None, device: str | None = None) -> None:
    torch.manual_seed(config.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _make_run_dir()
    save_config(config, run_dir / "config.json")
    writer = SummaryWriter(str(run_dir / "tb"))
    print(f"discriminator run dir: {run_dir}")

    dataset = MusicSeparationDataset(
        config.dataset.root,
        sample_rate=config.dataset.sample_rate,
        segment_seconds=config.dataset.segment_seconds,
        random_crop=config.dataset.random_crop,
        mono=config.dataset.mono,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    use_amp = config.training.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_steps = steps if steps is not None else config.training.max_steps
    global_step = 0

    try:
        while global_step < max_steps:
            for batch in loader:
                batch = batch.to(device)
                stems = batch.sources[batch.source_mask]
                audio, impurity_targets, negative_counts = _make_discriminator_batch(stems, config.discriminator, config.model)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(audio)
                    loss = F.binary_cross_entropy_with_logits(logits, impurity_targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                if global_step % config.training.log_every == 0:
                    with torch.no_grad():
                        _log(writer, logits, impurity_targets, negative_counts, loss, global_step)
                    print(f"step={global_step} loss={float(loss.detach().cpu()):.6f}")
                if global_step > 0 and global_step % config.training.checkpoint_every == 0:
                    save_checkpoint(run_dir / "checkpoints" / f"step_{global_step:08d}.pt", config, model, optimizer, 0, global_step)
                global_step += 1
                if global_step >= max_steps:
                    break
    finally:
        save_checkpoint(run_dir / "checkpoints" / "last.pt", config, model, optimizer, 0, global_step)
        writer.close()


def _make_discriminator_batch(stems: torch.Tensor, config, model_config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if stems.shape[0] < 2:
        raise ValueError("discriminator training requires at least two valid stems in a batch")
    with torch.no_grad():
        positives = normalize_rms(stems, config.target_rms)
        negatives = []
        selected_counts = []
        scaled_segments = []
        for _ in range(stems.shape[0]):
            count = 3 if stems.shape[0] >= 3 and torch.rand((), device=stems.device) < config.three_stem_probability else 2
            indices = torch.randperm(stems.shape[0], device=stems.device)[:count]
            gains_db = _sample_gains_db(count, config, stems.device)
            gains = 10.0 ** (gains_db.view(count, 1, 1) / 20.0)
            scaled = stems[indices] * gains
            negatives.append(normalize_rms(scaled.sum(dim=0, keepdim=True), config.target_rms).squeeze(0))
            scaled_segments.append(scaled)
            selected_counts.append(count)

        scores = _weighted_stft_power_chunked(torch.cat(scaled_segments, dim=0), model_config, config.label_chunk_size)
        impurities = []
        offset = 0
        for count in selected_counts:
            item_scores = scores[offset : offset + count]
            impurities.append(1.0 - item_scores.max() / item_scores.sum().clamp_min(1e-8))
            offset += count

        negatives_tensor = torch.stack(negatives, dim=0)
        negative_impurities_tensor = torch.stack(impurities, dim=0)
        audio = torch.cat([positives, negatives_tensor], dim=0)
        impurity_targets = torch.cat([torch.zeros(positives.shape[0], device=stems.device), negative_impurities_tensor])
        sample_counts = torch.cat(
            [
                torch.ones(positives.shape[0], device=stems.device, dtype=torch.float32),
                torch.tensor(selected_counts, device=stems.device, dtype=torch.float32),
            ]
        )
        permutation = torch.randperm(audio.shape[0], device=stems.device)
        return audio[permutation], impurity_targets[permutation], sample_counts[permutation]


def _sample_gains_db(count: int, config, device: torch.device) -> torch.Tensor:
    return torch.empty(count, device=device).uniform_(config.negative_gain_min_db, config.negative_gain_max_db)


def _weighted_stft_power_chunked(audio: torch.Tensor, model_config, chunk_size: int) -> torch.Tensor:
    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, audio.shape[0], chunk_size):
        end = min(start + chunk_size, audio.shape[0])
        chunks.append(weighted_stft_power(audio[start:end], model_config))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _log(writer: SummaryWriter, logits: torch.Tensor, targets: torch.Tensor, negative_counts: torch.Tensor, loss: torch.Tensor, step: int) -> None:
    predicted_impurity = torch.sigmoid(logits.detach())
    impurity_mae = (predicted_impurity - targets).abs().mean()
    negative_mask = targets > 0
    writer.add_scalar("loss/impurity_bce", float(loss.detach().cpu()), step)
    writer.add_scalar("metric/impurity_mae", float(impurity_mae.detach().cpu()), step)
    writer.add_scalar("metric/predicted_impurity", float(predicted_impurity.mean().cpu()), step)
    writer.add_scalar("metric/target_impurity", float(targets.detach().mean().cpu()), step)
    if bool(negative_mask.any()):
        writer.add_scalar("debug/negative_stem_count", float(negative_counts[negative_mask].float().mean().detach().cpu()), step)


def _make_run_dir() -> Path:
    model_root = Path(__file__).resolve().parents[3]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return model_root / "runs" / "vst" / "discriminator" / timestamp


if __name__ == "__main__":
    main()
