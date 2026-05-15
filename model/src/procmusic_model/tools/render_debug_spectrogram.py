from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from procmusic_model.data.dataset import _load_wav


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--out-dir", default="runs/debug_spectrogram")
    args = parser.parse_args()

    audio = _load_wav(Path(args.wav), args.sample_rate)
    spec = torch.stft(audio.mean(dim=0), n_fft=1024, hop_length=256, return_complex=True).abs().log1p()
    writer = SummaryWriter(args.out_dir)
    writer.add_image("spectrogram", spec.unsqueeze(0), 0)
    writer.add_audio("audio", audio, 0, sample_rate=args.sample_rate)
    writer.close()
    print(f"wrote TensorBoard debug data to {args.out_dir}")


if __name__ == "__main__":
    main()
