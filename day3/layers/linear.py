from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def _tp_info(tp_rank: int | None, tp_size: int | None) -> tuple[int, int]:
    """Use an initialized process group, or explicit values for local tests."""

    if tp_rank is None:
        tp_rank = dist.get_rank() if dist.is_initialized() else 0
    if tp_size is None:
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("expected 0 <= tp_rank < tp_size")
    return tp_rank, tp_size


def _copy_parameter(param: nn.Parameter, value: torch.Tensor) -> None:
    if param.shape != value.shape:
        raise ValueError(f"shape mismatch: parameter {param.shape}, value {value.shape}")
    with torch.no_grad():
        param.copy_(value)


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_rank, self.tp_size = _tp_info(tp_rank, tp_size)
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.bias_loader  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, *args
    ) -> None:
        raise NotImplementedError

    def bias_loader(
        self, param: nn.Parameter, loaded_bias: torch.Tensor, *args
    ) -> None:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Every rank stores the complete linear layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__(
            input_size,
            output_size,
            bias,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def weight_loader(self, param, loaded_weight, *args) -> None:
        _copy_parameter(param, loaded_weight)

    def bias_loader(self, param, loaded_bias, *args) -> None:
        _copy_parameter(param, loaded_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Shard W along output features; input x is replicated on every rank."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        rank, size = _tp_info(tp_rank, tp_size)
        if output_size % size != 0:
            raise ValueError("output_size must be divisible by tp_size")
        self.global_output_size = output_size
        super().__init__(
            input_size,
            output_size // size,
            bias,
            tp_rank=rank,
            tp_size=size,
        )

    def weight_loader(self, param, loaded_weight, *args) -> None:
        shard_size = self.global_output_size // self.tp_size
        shard = loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size)
        _copy_parameter(param, shard)

    def bias_loader(self, param, loaded_bias, *args) -> None:
        shard_size = self.global_output_size // self.tp_size
        shard = loaded_bias.narrow(0, self.tp_rank * shard_size, shard_size)
        _copy_parameter(param, shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: [..., global_output_size / tp_size]. No collective here.
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Store several column-parallel projections in one local parameter."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        if not output_sizes:
            raise ValueError("output_sizes cannot be empty")
        rank, size = _tp_info(tp_rank, tp_size)
        if any(output_size % size != 0 for output_size in output_sizes):
            raise ValueError("every merged output size must be divisible by tp_size")
        self.output_sizes = list(output_sizes)
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            tp_rank=rank,
            tp_size=size,
        )

    def _target_slice(self, loaded_weight_id: int) -> tuple[int, int]:
        if not 0 <= loaded_weight_id < len(self.output_sizes):
            raise IndexError("loaded_weight_id is outside output_sizes")
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        size = self.output_sizes[loaded_weight_id] // self.tp_size
        return offset, size

    def weight_loader(self, param, loaded_weight, loaded_weight_id: int) -> None:
        offset, shard_size = self._target_slice(loaded_weight_id)
        source = loaded_weight.narrow(
            0, self.tp_rank * shard_size, shard_size
        )
        target = param.data.narrow(0, offset, shard_size)
        _copy_parameter(target, source)

    def bias_loader(self, param, loaded_bias, loaded_weight_id: int) -> None:
        offset, shard_size = self._target_slice(loaded_weight_id)
        source = loaded_bias.narrow(0, self.tp_rank * shard_size, shard_size)
        target = param.data.narrow(0, offset, shard_size)
        _copy_parameter(target, source)


class QKVColumnParallelLinear(ColumnParallelLinear):
    """A merged Q/K/V projection whose shards always contain complete heads."""

    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        rank, size = _tp_info(tp_rank, tp_size)
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % size != 0 or num_kv_heads % size != 0:
            raise ValueError("Q and KV head counts must be divisible by tp_size")
        self.head_size = head_size
        self.local_num_heads = num_heads // size
        self.local_num_kv_heads = num_kv_heads // size
        self.q_size = head_size * num_heads
        self.kv_size = head_size * num_kv_heads
        total_output_size = self.q_size + 2 * self.kv_size
        super().__init__(
            input_size,
            total_output_size,
            bias,
            tp_rank=rank,
            tp_size=size,
        )

    def _qkv_slice(self, weight_id: str) -> tuple[int, int, int]:
        if weight_id == "q":
            local_offset = 0
            global_size = self.q_size
        elif weight_id == "k":
            local_offset = self.q_size // self.tp_size
            global_size = self.kv_size
        elif weight_id == "v":
            local_offset = (self.q_size + self.kv_size) // self.tp_size
            global_size = self.kv_size
        else:
            raise ValueError("weight_id must be 'q', 'k', or 'v'")
        shard_size = global_size // self.tp_size
        return local_offset, self.tp_rank * shard_size, shard_size

    def weight_loader(self, param, loaded_weight, weight_id: str) -> None:
        target_offset, source_offset, shard_size = self._qkv_slice(weight_id)
        target = param.data.narrow(0, target_offset, shard_size)
        source = loaded_weight.narrow(0, source_offset, shard_size)
        _copy_parameter(target, source)

    def bias_loader(self, param, loaded_bias, weight_id: str) -> None:
        target_offset, source_offset, shard_size = self._qkv_slice(weight_id)
        target = param.data.narrow(0, target_offset, shard_size)
        source = loaded_bias.narrow(0, source_offset, shard_size)
        _copy_parameter(target, source)


class RowParallelLinear(LinearBase):
    """Shard W and x along input features, then sum local partial outputs."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        rank, size = _tp_info(tp_rank, tp_size)
        if input_size % size != 0:
            raise ValueError("input_size must be divisible by tp_size")
        self.global_input_size = input_size
        super().__init__(
            input_size // size,
            output_size,
            bias,
            tp_rank=rank,
            tp_size=size,
        )

    def weight_loader(self, param, loaded_weight, *args) -> None:
        shard_size = self.global_input_size // self.tp_size
        shard = loaded_weight.narrow(1, self.tp_rank * shard_size, shard_size)
        _copy_parameter(param, shard)

    def bias_loader(self, param, loaded_bias, *args) -> None:
        # Bias is not sharded. It is added once after all_reduce.
        _copy_parameter(param, loaded_bias)

    def forward_local(self, x: torch.Tensor) -> torch.Tensor:
        """Return this rank's partial matrix product without bias/collective."""

        return F.linear(x, self.weight, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward_local(x)
        if self.tp_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "tp_size > 1 requires torch.distributed for forward(); "
                    "use forward_local() only for a virtual-shard test"
                )
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            output = output + self.bias
        return output
