from __future__ import annotations

import unittest

import torch

from procmusic_model.audio import STFTConfig, STFTFrontend


class STFTTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        frontend = STFTFrontend(STFTConfig(n_fft=256, hop_length=64, win_length=256))
        audio = torch.randn(2, 1, 4096)
        restored = frontend.istft(frontend.stft(audio), length=audio.shape[-1])
        self.assertLess((audio - restored).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
