from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def _tp_info(tp_rank: int | None, tp_size: int | None) -> tuple[int, int]:
    if tp_rank is None:
        tp_rank = dist.get_rank() if dist.is_initialized() else 0
    if tp_size is None:
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("expected 0 <= tp_rank < tp_size")
    return tp_rank, tp_size


class VocabParallelEmbedding(nn.Module):
    """Shard embedding rows (vocabulary ids) across tensor-parallel ranks."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_rank, self.tp_size = _tp_info(tp_rank, tp_size)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padded_num_embeddings = (
            (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        )
        self.num_embeddings_per_partition = (
            self.padded_num_embeddings // self.tp_size
        )
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        expected = (self.num_embeddings, self.embedding_dim)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"expected full embedding weight {expected}, got {tuple(loaded_weight.shape)}"
            )
        actual_start = min(self.vocab_start_index, self.num_embeddings)
        actual_end = min(self.vocab_end_index, self.num_embeddings)
        actual_size = max(0, actual_end - actual_start)
        with torch.no_grad():
            if actual_size:
                param[:actual_size].copy_(
                    loaded_weight.narrow(0, actual_start, actual_size)
                )
            if actual_size < self.num_embeddings_per_partition:
                param[actual_size:].zero_()

    def forward_local(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.num_embeddings):
            raise ValueError("input_ids contain ids outside the original vocabulary")

        mask = (input_ids >= self.vocab_start_index) & (
            input_ids < self.vocab_end_index
        )
        local_ids = input_ids - self.vocab_start_index
        local_ids = local_ids.masked_fill(~mask, 0)
        output = F.embedding(local_ids, self.weight)
        # Invalid local ids were temporarily mapped to row 0; remove them again.
        return output * mask.unsqueeze(-1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.forward_local(input_ids)
        if self.tp_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "tp_size > 1 requires torch.distributed for forward(); "
                    "sum forward_local() outputs in virtual-rank tests"
                )
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


class ParallelLMHead(VocabParallelEmbedding):
    """Project hidden states to local vocabulary logits and gather on rank 0."""

    def tie_weights(self, embedding: VocabParallelEmbedding) -> None:
        if (
            embedding.tp_rank != self.tp_rank
            or embedding.tp_size != self.tp_size
            or embedding.weight.shape != self.weight.shape
        ):
            raise ValueError("embedding and LM head partitions are incompatible")
        self.weight = embedding.weight

    def select_prefill_hidden(
        self, hidden_states: torch.Tensor, cu_seqlens_q: torch.Tensor
    ) -> torch.Tensor:
        if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
            raise ValueError("cu_seqlens_q must be a 1D tensor of length batch+1")
        last_indices = cu_seqlens_q[1:].to(torch.long) - 1
        if torch.any(last_indices < 0):
            raise ValueError("every Prefill sequence must contain a query token")
        return hidden_states.index_select(0, last_indices).contiguous()

    def forward_local(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens_q is not None:
            hidden_states = self.select_prefill_hidden(hidden_states, cu_seqlens_q)
        return F.linear(hidden_states, self.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        *,
        gather_output: bool = True,
    ) -> torch.Tensor:
        logits = self.forward_local(hidden_states, cu_seqlens_q)
        if self.tp_size == 1 or not gather_output:
            return logits[..., : self.num_embeddings]
        if not dist.is_initialized():
            raise RuntimeError("distributed LM-head gathering requires a process group")

        gathered = (
            [torch.empty_like(logits) for _ in range(self.tp_size)]
            if self.tp_rank == 0
            else None
        )
        dist.gather(logits, gather_list=gathered, dst=0)
        if self.tp_rank == 0:
            return torch.cat(gathered, dim=-1)[..., : self.num_embeddings]
        # Worker ranks do not sample; returning local logits keeps the API tensor-only.
        return logits
