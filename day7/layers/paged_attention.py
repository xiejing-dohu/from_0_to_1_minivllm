from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _validate(query, k_cache, v_cache, block_tables, context_lens):
    if query.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("expected query [batch,q_heads,D] and equal K/V cache [blocks,block,kv_heads,D]")
    if query.shape[0] != block_tables.shape[0] or query.shape[0] != context_lens.numel():
        raise ValueError("batch dimensions must match")
    if query.shape[-1] != k_cache.shape[-1] or query.shape[1] % k_cache.shape[2] != 0:
        raise ValueError("head_dim must match and q_heads must be divisible by kv_heads")
    if block_tables.ndim != 2 or context_lens.ndim != 1:
        raise ValueError("block_tables must be 2-D and context_lens must be 1-D")
    if block_tables.dtype != torch.long or context_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_tables must be int64; context_lens must be int32/int64")
    tensors = (query, k_cache, v_cache, block_tables, context_lens)
    if any(t.device != query.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if query.dtype != k_cache.dtype or query.dtype != v_cache.dtype:
        raise ValueError("query and cache dtypes must match")
    capacity = block_tables.shape[1] * k_cache.shape[1]
    if bool(torch.any(context_lens <= 0)) or bool(torch.any(context_lens > capacity)):
        raise ValueError("each context length must be in the block-table capacity")
    for batch in range(query.shape[0]):
        used = (int(context_lens[batch].item()) + k_cache.shape[1] - 1) // k_cache.shape[1]
        physical = block_tables[batch, :used]
        if bool(torch.any((physical < 0) | (physical >= k_cache.shape[0]))):
            raise ValueError("a used block-table entry points outside the physical cache")


def paged_attention_reference(query, k_cache, v_cache, block_tables, context_lens, scale=None):
    _validate(query, k_cache, v_cache, block_tables, context_lens)
    scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    repeat = query.shape[1] // k_cache.shape[2]
    outputs = []
    block_size = k_cache.shape[1]
    for batch in range(query.shape[0]):
        length = int(context_lens[batch].item())
        positions = torch.arange(length, device=query.device)
        physical = block_tables[batch, positions // block_size]
        offsets = positions % block_size
        keys = k_cache[physical, offsets].repeat_interleave(repeat, dim=1).transpose(0, 1)
        values = v_cache[physical, offsets].repeat_interleave(repeat, dim=1).transpose(0, 1)
        scores = torch.sum(query[batch].float()[:, None, :] * keys.float(), dim=-1) * scale
        output = torch.sum(torch.softmax(scores, dim=-1)[..., None] * values.float(), dim=1)
        outputs.append(output.to(query.dtype))
    return torch.stack(outputs)


if triton is not None:
    @triton.jit
    def _paged_decode_kernel(out, query, kc, vc, tables, lens, scale,
                             q_heads: tl.constexpr, kv_heads: tl.constexpr,
                             head_dim: tl.constexpr, block_size: tl.constexpr,
                             max_blocks: tl.constexpr, block_d: tl.constexpr,
                             block_n: tl.constexpr):
        batch, q_head = tl.program_id(0), tl.program_id(1)
        kv_head = q_head // (q_heads // kv_heads)
        length = tl.load(lens + batch)
        od = tl.arange(0, block_d)
        q = tl.load(query + (batch * q_heads + q_head) * head_dim + od,
                    mask=od < head_dim, other=0.0).to(tl.float32)
        m, denom = -float("inf"), 0.0
        acc = tl.zeros([block_d], tl.float32)
        for start in tl.range(0, max_blocks * block_size, block_n):
            pos = start + tl.arange(0, block_n)
            logical = pos // block_size
            valid = (pos < length) & (logical < max_blocks)
            physical = tl.load(tables + batch * max_blocks + logical, mask=valid, other=0).to(tl.int64)
            offset = pos % block_size
            base = ((physical * block_size + offset) * kv_heads + kv_head) * head_dim
            keys = tl.load(kc + base[None, :] + od[:, None],
                           mask=(od[:, None] < head_dim) & valid[None, :], other=0.0).to(tl.float32)
            scores = tl.sum(q[:, None] * keys, axis=0) * scale
            scores = tl.where(valid, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            new_m = tl.maximum(m, tile_max)
            alpha = tl.exp(m - new_m)
            weights = tl.where(valid, tl.exp(scores - new_m), 0.0)
            values = tl.load(vc + base[None, :] + od[:, None],
                             mask=(od[:, None] < head_dim) & valid[None, :], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(values * weights[None, :], axis=1)
            denom = denom * alpha + tl.sum(weights, axis=0)
            m = new_m
        tl.store(out + (batch * q_heads + q_head) * head_dim + od,
                 (acc / denom).to(out.dtype.element_ty), mask=od < head_dim)


def paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens, scale=None, backend="auto"):
    _validate(query, k_cache, v_cache, block_tables, context_lens)
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    use_triton = backend == "triton" or (backend == "auto" and query.is_cuda)
    if not use_triton:
        return paged_attention_reference(query, k_cache, v_cache, block_tables, context_lens, scale)
    if triton is None or not query.is_cuda:
        raise ValueError("the Triton backend requires Triton and CUDA tensors")
    query, k_cache, v_cache = query.contiguous(), k_cache.contiguous(), v_cache.contiguous()
    block_tables, context_lens = block_tables.contiguous(), context_lens.contiguous()
    output = torch.empty_like(query)
    head_dim, block_size = query.shape[-1], k_cache.shape[1]
    actual_scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    block_d = max(16, triton.next_power_of_2(head_dim))
    _paged_decode_kernel[(query.shape[0], query.shape[1])](
        output, query, k_cache, v_cache, block_tables, context_lens, actual_scale,
        q_heads=query.shape[1], kv_heads=k_cache.shape[2], head_dim=head_dim,
        block_size=block_size, max_blocks=block_tables.shape[1], block_d=block_d,
        block_n=32, num_warps=4,
    )
    return output
