"""Validate embedding all-reduce and LM-head gather with local Gloo ranks."""

import argparse
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from day4.layers.embedding_head import ParallelLMHead, VocabParallelEmbedding


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(2026)
    vocab_size, hidden_size = 7, 4
    weight = torch.randn(vocab_size, hidden_size)
    input_ids = torch.tensor([[0, 4, 6], [2, 5, 1]], dtype=torch.long)

    embedding = VocabParallelEmbedding(vocab_size, hidden_size)
    embedding.weight_loader(embedding.weight, weight)
    actual_embedding = embedding(input_ids)
    expected_embedding = F.embedding(input_ids, weight)
    embedding_ok = torch.allclose(actual_embedding, expected_embedding)

    head = ParallelLMHead(vocab_size, hidden_size)
    head.tie_weights(embedding)
    hidden = torch.randn(2, hidden_size)
    actual_logits = head(hidden)
    logits_ok = True
    if rank == 0:
        expected_logits = F.linear(hidden, weight)
        logits_ok = torch.allclose(actual_logits, expected_logits)
        print(f"[VocabParallelEmbedding] allclose={embedding_ok}")
        print(f"[ParallelLMHead] allclose={logits_ok}")
        print(f"[TiedWeight] same_parameter={head.weight is embedding.weight}")

    passed = torch.tensor(int(embedding_ok and logits_ok))
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if passed.item() != 1:
        raise AssertionError("distributed embedding/head check failed")
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be positive")
    mp.spawn(
        worker,
        args=(args.processes, free_local_port()),
        nprocs=args.processes,
        join=True,
    )


if __name__ == "__main__":
    main()
