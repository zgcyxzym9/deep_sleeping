from __future__ import annotations

import unittest

import torch

from procmusic_model.losses import greedy_match


class MatchingTest(unittest.TestCase):
    def test_greedy_matching_handles_permutation(self) -> None:
        target_a = torch.sin(torch.linspace(0, 10, 1024)).view(1, 1, 1024)
        target_b = torch.cos(torch.linspace(0, 10, 1024)).view(1, 1, 1024)
        targets = torch.stack([target_a, target_b], dim=1)
        estimates = torch.stack([target_b, target_a], dim=1)
        mask = torch.tensor([[True, True]])
        matched, pairs = greedy_match(estimates, targets, mask)
        self.assertEqual(pairs[0], [(0, 1), (1, 0)])
        self.assertLess((matched - estimates).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
