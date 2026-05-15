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
    for idx in range(max(1, count)):
        _write_wav(out_dir / f"source_{idx:02d}.wav", result["sources"][0, idx].detach().cpu(), config.dataset.sample_rate)
    _write_wav(out_dir / "residual.wav", result["residuals"][0, -1].detach().cpu(), config.dataset.sample_rate)
    print(f"predicted_source_count={count}")


def _write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.clamp(-1.0, 1.0)
    pcm = (audio.transpose(0, 1).contiguous().view(-1) * 32767.0).short().numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(audio.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


if __name__ == "__main__":
    main()
