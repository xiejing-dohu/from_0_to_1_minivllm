from __future__ import annotations

import torch
from torch import nn


class Sampler(nn.Module):
    """Temperature sampling using the exponential-race multinomial trick."""

    def forward(
        self, logits: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        if logits.ndim < 2:
            raise ValueError("logits must have shape [..., vocab_size]")
        if temperature.ndim != 1 or temperature.shape[0] != logits.shape[0]:
            raise ValueError("temperature must have shape [batch]")
        if torch.any(temperature <= 1e-10):
            raise ValueError("all temperatures must be greater than 1e-10")

        # Do not mutate logits: callers may reuse them for debugging or scoring.
        scaled_logits = logits / temperature.unsqueeze(-1)
        probabilities = torch.softmax(scaled_logits, dim=-1)
        noise = torch.empty_like(probabilities).exponential_(1).clamp_min_(1e-10)
        return (probabilities / noise).argmax(dim=-1)
