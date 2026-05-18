from __future__ import annotations

import argparse
import wave
from pathlib import Path

import torch

from procmusic_model.config import load_config
from procmusic_model.data.dataset import _load_wav
from procmusic_model.systems import OpenSetSeparator
from procmusic_model.training import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mixture = _load_wav(Path(args.mixture), config.dataset.sample_rate).unsqueeze(0).to(device)
    model = OpenSetSeparator(config.model, channels=mixture.shape[1]).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    result = model.separate(mixture)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = int(result["predicted_source_count"][0].cpu())
    source_count = min(count, result["sources"].shape[1])
    for idx in range(source_count):
        _write_wav(out_dir / f"source_{idx:02d}.wav", result["sources"][0, idx].detach().cpu(), config.dataset.sample_rate)
    _write_wav(out_dir / "residual.wav", result["residual"][0].detach().cpu(), config.dataset.sample_rate)
    _print_separation_summary(mixture, result)
    print(f"predicted_source_count={count}")


def _write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.clamp(-1.0, 1.0)
    pcm = (audio.transpose(0, 1).contiguous().view(-1) * 32767.0).short().numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(audio.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def _print_separation_summary(mixture: torch.Tensor, result: dict[str, torch.Tensor]) -> None:
    sources = result["sources"]
    residual = result["residual"]
    mixture_rms = _rms(mixture)
    residual_rms = _rms(residual)
    print(f"mixture_rms={mixture_rms:.6f}")
    print(f"residual_rms={residual_rms:.6f}")
    print(f"residual_to_mixture_rms={residual_rms / max(mixture_rms, 1e-8):.3f}")
    print(f"stop_prob={result['stop_prob'][0].detach().cpu().tolist()}")
    for idx in range(sources.shape[1]):
        source = sources[:, idx]
        print(
            f"source_{idx:02d}_rms={_rms(source):.6f} "
            f"source_{idx:02d}_mixture_corr={_corr(source, mixture):.3f}"
        )


def _rms(audio: torch.Tensor) -> float:
    return float(audio.detach().pow(2).mean().sqrt().cpu())


def _corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.detach().flatten()
    b = b.detach().flatten()
    a = a - a.mean()
    b = b - b.mean()
    value = (a * b).mean() / (a.pow(2).mean().sqrt() * b.pow(2).mean().sqrt()).clamp_min(eps)
    return float(value.cpu())


if __name__ == "__main__":
    main()
