from __future__ import annotations

import torch
from torch import nn


class SourceDiscovery(nn.Module):
    """Autoregressive source-query controller with a STOP logit at each step."""

    def __init__(self, hidden_dim: int, max_steps: int) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.start = nn.Parameter(torch.zeros(hidden_dim))
        self.step_embed = nn.Embedding(max_steps + 1, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.query = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.stop = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, mix_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, hidden_dim = mix_embedding.shape
        state = mix_embedding
        previous = self.start.unsqueeze(0).expand(batch, hidden_dim)
        queries = []
        stop_logits = []
        for step in range(self.max_steps):
            step_input = previous + self.step_embed.weight[step].unsqueeze(0)
            state = self.cell(step_input, state)
            query = self.query(state)
            queries.append(query)
            stop_logits.append(self.stop(state).squeeze(-1))
            previous = query
        final_state = self.cell(previous + self.step_embed.weight[self.max_steps].unsqueeze(0), state)
        stop_logits.append(self.stop(final_state).squeeze(-1))
        return torch.stack(queries, dim=1), torch.stack(stop_logits, dim=1)
