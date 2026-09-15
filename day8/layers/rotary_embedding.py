from __future__ import annotations
import torch
from torch import nn


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    if x.ndim not in (3, 4) or rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("x must be [T,H,D] or [B,S,H,D] with a valid even rotary_dim")
    expected = x.shape[:-2] + (rotary_dim // 2,)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(f"cos/sin must have shape {expected}")
    rotary, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return torch.cat((rotated, tail), dim=-1)


class QKRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.numel():
            raise ValueError("the last dimension must equal head_dim")
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps) * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rotary_dim: int | None = None, max_position: int = 4096, base: float = 1_000_000.0):
        super().__init__()
        rotary_dim = head_dim if rotary_dim is None else rotary_dim
        if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_dim:
            raise ValueError("rotary_dim must be positive, even, and no larger than head_dim")
        self.head_dim, self.rotary_dim = head_dim, rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("positions must use int32 or int64")
        if query.shape[:-2] != key.shape[:-2] or query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError("query/key token dimensions and head_dim must match")
        token_shape = query.shape[:-2]
        if positions.shape != token_shape:
            if query.ndim == 4 and positions.ndim == 1 and positions.numel() == query.shape[1]:
                positions = positions.unsqueeze(0).expand(query.shape[0], -1)
            else:
                raise ValueError("positions must match packed or batched token dimensions")
        if positions.numel() and (int(positions.min()) < 0 or int(positions.max()) >= self.cos_cache.shape[0]):
            raise ValueError("position is outside the precomputed RoPE cache")
        cos = self.cos_cache[positions].to(dtype=query.dtype)
        sin = self.sin_cache[positions].to(dtype=query.dtype)
        return apply_rotary(query, cos, sin, self.rotary_dim), apply_rotary(key, cos, sin, self.rotary_dim)
