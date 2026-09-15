from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installations can still use the reference path.
    triton = None
    tl = None


def _validate_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int | None,
) -> tuple[int, int, int, int]:
    if key.ndim != 3:
        raise ValueError("key and value must have shape [num_tokens, num_kv_heads, head_dim]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if k_cache.ndim != 4 or k_cache.shape != v_cache.shape:
        raise ValueError(
            "K/V caches must have the same [num_blocks, block_size, num_kv_heads, head_dim] shape"
        )
    if key.shape[1:] != k_cache.shape[2:]:
        raise ValueError("input KV head dimensions must match the cache")
    if key.device != value.device or key.device != k_cache.device or key.device != v_cache.device:
        raise ValueError("key, value, and both caches must be on the same device")
    if key.dtype != value.dtype or key.dtype != k_cache.dtype or key.dtype != v_cache.dtype:
        raise ValueError("key, value, and both caches must have the same dtype")
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("K/V caches must be contiguous")
    if slot_mapping.ndim != 1 or slot_mapping.numel() != key.shape[0]:
        raise ValueError("slot_mapping must contain one entry per input token")
    if slot_mapping.dtype != torch.long:
        raise TypeError("slot_mapping must use torch.long (int64)")
    if slot_mapping.device != key.device:
        raise ValueError("slot_mapping must be on the same device as key/value")

    num_blocks, actual_block_size, num_kv_heads, head_dim = k_cache.shape
    if block_size is not None and block_size != actual_block_size:
        raise ValueError("block_size does not match the cache shape")
    capacity = num_blocks * actual_block_size
    if slot_mapping.numel():
        minimum = int(slot_mapping.min().item())
        maximum = int(slot_mapping.max().item())
        if minimum < -1 or maximum >= capacity:
            raise ValueError(f"slot_mapping values must be -1 or in [0, {capacity})")
        valid_slots = slot_mapping[slot_mapping >= 0]
        if valid_slots.unique().numel() != valid_slots.numel():
            raise ValueError("slot_mapping cannot contain duplicate writable slots")
    return actual_block_size, num_kv_heads, head_dim, capacity


def store_kv_cache_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int | None = None,
) -> None:
    """Reference paged-cache write implemented with PyTorch indexing.

    ``slot_mapping=-1`` means that the corresponding token is intentionally not
    cached. All other values are physical slots, not positions inside a sequence.
    """
    _, num_kv_heads, head_dim, capacity = _validate_inputs(
        key, value, k_cache, v_cache, slot_mapping, block_size
    )
    valid = slot_mapping >= 0
    if not bool(valid.any()):
        return
    slots = slot_mapping[valid]
    k_cache.view(capacity, num_kv_heads, head_dim)[slots] = key[valid]
    v_cache.view(capacity, num_kv_heads, head_dim)[slots] = value[valid]


if triton is not None:

    @triton.jit
    def _store_kv_cache_kernel(
        key_ptr,
        value_ptr,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        slot = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
        writable = slot >= 0
        safe_slot = tl.where(writable, slot, 0)

        block_id = safe_slot // block_size
        block_offset = safe_slot % block_size
        offsets_d = tl.arange(0, block_d)
        mask = writable & (offsets_d < head_dim)

        input_base = (token_idx * num_kv_heads + head_idx) * head_dim
        cache_base = (
            ((block_id * block_size + block_offset) * num_kv_heads + head_idx)
            * head_dim
        )
        key = tl.load(key_ptr + input_base + offsets_d, mask=mask, other=0.0)
        value = tl.load(value_ptr + input_base + offsets_d, mask=mask, other=0.0)
        tl.store(k_cache_ptr + cache_base + offsets_d, key, mask=mask)
        tl.store(v_cache_ptr + cache_base + offsets_d, value, mask=mask)


def store_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int | None = None,
    backend: str = "auto",
) -> None:
    """Write new K/V vectors into a paged cache using PyTorch or Triton."""
    actual_block_size, num_kv_heads, head_dim, _ = _validate_inputs(
        key, value, k_cache, v_cache, slot_mapping, block_size
    )
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    use_triton = backend == "triton" or (backend == "auto" and key.is_cuda)
    if not use_triton:
        store_kv_cache_torch(key, value, k_cache, v_cache, slot_mapping, actual_block_size)
        return
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not key.is_cuda:
        raise ValueError("the Triton backend requires CUDA tensors")

    key = key.contiguous()
    value = value.contiguous()
    block_d = triton.next_power_of_2(head_dim)
    _store_kv_cache_kernel[(key.shape[0], num_kv_heads)](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=actual_block_size,
        block_d=block_d,
    )
