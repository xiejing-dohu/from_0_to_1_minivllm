from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _flash_prefill_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, cu_ptr, scale,
        num_q_heads: tl.constexpr, num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr, block_d: tl.constexpr,
        block_m: tl.constexpr, block_n: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        q_head = tl.program_id(1)
        seq = tl.program_id(2)
        seq_start = tl.load(cu_ptr + seq)
        seq_end = tl.load(cu_ptr + seq + 1)
        seq_len = seq_end - seq_start
        offs_m = query_block * block_m + tl.arange(0, block_m)
        offs_d = tl.arange(0, block_d)
        valid_m = offs_m < seq_len
        kv_head = q_head // (num_q_heads // num_kv_heads)

        q_base = (seq_start + offs_m[:, None]) * num_q_heads * head_dim
        q = tl.load(
            q_ptr + q_base + q_head * head_dim + offs_d[None, :],
            mask=valid_m[:, None] & (offs_d[None, :] < head_dim), other=0.0,
        )
        row_max = tl.full([block_m], -float("inf"), tl.float32)
        row_sum = tl.zeros([block_m], tl.float32)
        acc = tl.zeros([block_m, block_d], tl.float32)

        for start_n in tl.range(0, seq_len, block_n):
            offs_n = start_n + tl.arange(0, block_n)
            valid_n = offs_n < seq_len
            k_base = (seq_start + offs_n[None, :]) * num_kv_heads * head_dim
            k = tl.load(
                k_ptr + k_base + kv_head * head_dim + offs_d[:, None],
                mask=(offs_d[:, None] < head_dim) & valid_n[None, :], other=0.0,
            )
            scores = tl.dot(q, k) * scale
            allowed = valid_m[:, None] & valid_n[None, :] & (offs_n[None, :] <= offs_m[:, None])
            scores = tl.where(allowed, scores, -float("inf"))
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            alpha = tl.exp(row_max - new_max)
            probs = tl.exp(scores - new_max[:, None])
            probs = tl.where(allowed, probs, 0.0)
            acc *= alpha[:, None]
            v_base = (seq_start + offs_n[:, None]) * num_kv_heads * head_dim
            value = tl.load(
                v_ptr + v_base + kv_head * head_dim + offs_d[None, :],
                mask=valid_n[:, None] & (offs_d[None, :] < head_dim), other=0.0,
            )
            acc += tl.dot(probs.to(value.dtype), value)
            row_sum = row_sum * alpha + tl.sum(probs, axis=1)
            row_max = new_max

        out_base = (seq_start + offs_m[:, None]) * num_q_heads * head_dim
        tl.store(
            out_ptr + out_base + q_head * head_dim + offs_d[None, :],
            (acc / row_sum[:, None]).to(out_ptr.dtype.element_ty),
            mask=valid_m[:, None] & (offs_d[None, :] < head_dim),
        )


def _validate_cu(name: str, cu: torch.Tensor, total: int) -> None:
    if cu.ndim != 1 or cu.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional cumulative-length tensor")
    if cu.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64")
    if int(cu[0].item()) != 0 or int(cu[-1].item()) != total:
        raise ValueError(f"{name} must start at 0 and end at the packed token count")
    if bool(torch.any(cu[1:] <= cu[:-1])):
        raise ValueError(f"{name} must be strictly increasing")


def _validate(q, k, v, cu_q, cu_k) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("q, k, v must have packed [tokens, heads, head_dim] shapes")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K/V head_dim must match")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads for GQA")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if cu_q.device != q.device or cu_k.device != q.device:
        raise ValueError("cumulative lengths must be on the same device as q/k/v")
    _validate_cu("cu_seqlens_q", cu_q, q.shape[0])
    _validate_cu("cu_seqlens_k", cu_k, k.shape[0])
    if cu_q.numel() != cu_k.numel():
        raise ValueError("Q and K cumulative lengths must describe the same batch")


def _sequence_parts(q, k, v, qs, qe, ks, ke):
    q_part = q[qs:qe].transpose(0, 1)
    repeat = q.shape[1] // k.shape[1]
    k_part = k[ks:ke].transpose(0, 1).repeat_interleave(repeat, dim=0)
    v_part = v[ks:ke].transpose(0, 1).repeat_interleave(repeat, dim=0)
    q_len, k_len = qe - qs, ke - ks
    if k_len < q_len:
        raise ValueError("each Prefill K sequence must be at least as long as its Q sequence")
    prefix = k_len - q_len
    mask = torch.arange(k_len, device=q.device)[None, :] <= (
        prefix + torch.arange(q_len, device=q.device)[:, None]
    )
    return q_part, k_part, v_part, mask


def prefill_attention_reference(q, k, v, cu_seqlens_q, cu_seqlens_k, scale=None):
    """Clear packed causal-attention baseline, including prefix and GQA cases."""
    _validate(q, k, v, cu_seqlens_q, cu_seqlens_k)
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    outputs = []
    for index in range(cu_seqlens_q.numel() - 1):
        qs, qe = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        ks, ke = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
        q_part, k_part, v_part, mask = _sequence_parts(q, k, v, qs, qe, ks, ke)
        scores = torch.matmul(q_part.float(), k_part.float().transpose(-1, -2)) * scale
        scores.masked_fill_(~mask, float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), v_part.float())
        outputs.append(out.to(q.dtype).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def prefill_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, scale=None, backend="auto"):
    """Packed Prefill using the Triton FlashAttention kernel or PyTorch SDPA."""
    _validate(q, k, v, cu_seqlens_q, cu_seqlens_k)
    if backend not in {"auto", "triton", "sdpa"}:
        raise ValueError("backend must be 'auto', 'triton', or 'sdpa'")
    same_lengths = torch.equal(cu_seqlens_q, cu_seqlens_k)
    use_triton = backend == "triton" or (
        backend == "auto" and q.is_cuda and same_lengths
    )
    if use_triton:
        if triton is None:
            raise RuntimeError("Triton is not installed")
        if not q.is_cuda:
            raise ValueError("the Triton backend requires CUDA tensors")
        if not same_lengths:
            raise ValueError("the Triton Prefill kernel requires equal Q/K sequence lengths")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cu_seqlens_q = cu_seqlens_q.contiguous()
        output = torch.empty_like(q)
        lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_len = int(lengths.max().item())
        block_d = triton.next_power_of_2(q.shape[-1])
        if block_d < 16:
            block_d = 16
        block_m = 32 if q.shape[-1] <= 128 else 16
        block_n = block_m
        grid = (triton.cdiv(max_len, block_m), q.shape[1], cu_seqlens_q.numel() - 1)
        actual_scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        _flash_prefill_kernel[grid](
            q, k, v, output, cu_seqlens_q, actual_scale,
            num_q_heads=q.shape[1], num_kv_heads=k.shape[1], head_dim=q.shape[-1],
            block_d=block_d, block_m=block_m, block_n=block_n,
            num_warps=4, num_stages=2,
        )
        return output

    outputs = []
    for index in range(cu_seqlens_q.numel() - 1):
        qs, qe = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        ks, ke = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
        q_part, k_part, v_part, mask = _sequence_parts(q, k, v, qs, qe, ks, ke)
        out = F.scaled_dot_product_attention(
            q_part.unsqueeze(0), k_part.unsqueeze(0), v_part.unsqueeze(0),
            attn_mask=mask, dropout_p=0.0, scale=scale,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)
