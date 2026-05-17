from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from procmusic_model.config import add_common_args, apply_overrides, load_config, save_config
from procmusic_model.data import MusicSeparationDataset, collate_separation_batch
from procmusic_model.losses import SeparationLoss, si_sdr
from procmusic_model.modules import SingleSourceDiscriminator
from procmusic_model.systems import OpenSetSeparator
from procmusic_model.training.checkpoint import load_checkpoint, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps.")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    train(config, device=args.device)


def train(config, device: str | None = None) -> None:
    torch.manual_seed(config.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.training.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")

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
    model = OpenSetSeparator(config.model, channels=channels).to(device)
    discriminator = _load_discriminator(config, channels, device)
    criterion = SeparationLoss(
        config.loss,
        config.model,
        discriminator,
        config.discriminator.target_rms,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    use_amp = config.training.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = _make_writer(output_dir)

    start_epoch = 0
    global_step = 0
    if config.training.resume:
        checkpoint = load_checkpoint(config.training.resume, model, optimizer)
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", 0))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.training.epochs):
        print(f"starting epoch {epoch}")
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model.forward_train(batch)
                loss_output = criterion(output, batch)
                loss = loss_output.total / config.training.grad_accum_steps

            scaler.scale(loss).backward()
            if (global_step + 1) % config.training.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if global_step % config.training.log_every == 0:
                with torch.no_grad():
                    _log(writer, loss_output.terms, output, batch, global_step, config.dataset.sample_rate)
            if global_step > 0 and global_step % config.training.checkpoint_every == 0:
                save_checkpoint(output_dir / "checkpoints" / f"step_{global_step:08d}.pt", config, model, optimizer, epoch, global_step)
            global_step += 1
            if global_step >= config.training.max_steps:
                print(f"step {global_step}")
                save_checkpoint(output_dir / "checkpoints" / "last.pt", config, model, optimizer, epoch, global_step)
                writer.close()
                return

    save_checkpoint(output_dir / "checkpoints" / "last.pt", config, model, optimizer, config.training.epochs, global_step)
    writer.close()


def _to_log_audio(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 2:
        return audio.mean(dim=0)
    return audio


@torch.no_grad()
def _log(writer: SummaryWriter, terms: dict[str, torch.Tensor], output, batch, step: int, sample_rate: int) -> None:
    for name, value in terms.items():
        writer.add_scalar(f"loss/{name}", float(value.detach().cpu()), step)
    stop_prob = torch.sigmoid(output.stop_logits)
    pred_count = _predicted_source_count(stop_prob, threshold=0.5, max_steps=output.estimated_sources.shape[1]).float().mean()
    writer.add_scalar("debug/predicted_source_count", float(pred_count.detach().cpu()), step)
    writer.add_scalar("debug/mixture_rms", float(batch.mixture.pow(2).mean().sqrt().detach().cpu()), step)
    if output.estimated_sources.numel() and batch.sources.numel():
        writer.add_scalar("metric/first_step_si_sdr", float(si_sdr(output.estimated_sources[:, 0], batch.sources[:, 0]).detach().cpu()), step)


def _predicted_source_count(stop_prob: torch.Tensor, threshold: float = 0.5, max_steps: int | None = None) -> torch.Tensor:
    has_stop = stop_prob > threshold
    first_stop = has_stop.float().argmax(dim=1)
    fallback = stop_prob.shape[1] - 1 if max_steps is None else max_steps
    return torch.where(has_stop.any(dim=1), first_stop, torch.full_like(first_stop, fallback))


def _load_discriminator(config, channels: int, device: str | torch.device) -> torch.nn.Module | None:
    if not config.discriminator.enabled:
        return None
    if not config.discriminator.checkpoint:
        raise ValueError("discriminator.enabled=true requires discriminator.checkpoint")
    discriminator = SingleSourceDiscriminator(
        config.model,
        audio_channels=channels,
        channels=config.discriminator.channels,
        hidden_dim=config.discriminator.hidden_dim,
    ).to(device)
    checkpoint = torch.load(config.discriminator.checkpoint, map_location="cpu", weights_only=False)
    discriminator.load_state_dict(checkpoint["model"])
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    return discriminator


def _make_writer(output_dir: Path | None = None):
    tb_dir = _timestamped_tb_dir()
    try:
        print(f"TensorBoard logdir: {tb_dir}")
        return SummaryWriter(str(tb_dir))
    except Exception as exc:
        fallback_dir = output_dir or tb_dir
        print(f"TensorBoard writer unavailable at {tb_dir}: {exc}")
        print("falling back to JSONL scalar logging")
        return JsonlSummaryWriter(fallback_dir / "scalars.jsonl")


def _timestamped_tb_dir() -> Path:
    model_root = Path(__file__).resolve().parents[3]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return model_root / "runs" / "vst" / "tb" / timestamp


class JsonlSummaryWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": global_step, "tag": tag, "value": float(scalar_value)}, sort_keys=True) + "\n")

    def add_audio(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


if __name__ == "__main__":
    main()
