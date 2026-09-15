"""Validate real collectives with torchrun (GPU) or multiprocessing (CPU)."""

import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from day3.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVColumnParallelLinear,
    RowParallelLinear,
)


def init_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")
    return dist.get_rank(), dist.get_world_size(), device


def gather_repacked(
    local: torch.Tensor, local_sizes: list[int], world_size: int
) -> torch.Tensor:
    parts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(parts, local)
    offsets = [0]
    for size in local_sizes:
        offsets.append(offsets[-1] + size)
    groups = [
        torch.cat([part[:, offsets[i] : offsets[i + 1]] for part in parts], dim=-1)
        for i in range(len(local_sizes))
    ]
    return torch.cat(groups, dim=-1)


def report(name: str, actual: torch.Tensor, expected: torch.Tensor, rank: int) -> None:
    error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)
    if rank == 0:
        print(f"[{name}] allclose={passed}, max_abs_err={error:.6f}")
    if not passed:
        raise AssertionError(f"{name} does not match the full linear layer")


@torch.no_grad()
def run_checks(rank: int, world_size: int, device: torch.device) -> None:
    generator = torch.Generator(device="cpu").manual_seed(2026)
    batch, input_size = 3, 8 * world_size
    x = torch.randn(batch, input_size, generator=generator).to(device)

    # ColumnParallelLinear: concatenate output shards.
    output_size = 6 * world_size
    weight = torch.randn(output_size, input_size, generator=generator).to(device)
    bias = torch.randn(output_size, generator=generator).to(device)
    column = ColumnParallelLinear(input_size, output_size).to(device)
    column.weight_loader(column.weight, weight)
    column.bias_loader(column.bias, bias)
    local = column(x)
    parts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(parts, local)
    report("ColumnParallel", torch.cat(parts, dim=-1), F.linear(x, weight, bias), rank)

    # MergedColumnParallelLinear: gather each logical projection separately.
    merged_sizes = [4 * world_size, 6 * world_size]
    merged_weights = [
        torch.randn(size, input_size, generator=generator).to(device)
        for size in merged_sizes
    ]
    merged = MergedColumnParallelLinear(input_size, merged_sizes, bias=False).to(device)
    for weight_id, loaded in enumerate(merged_weights):
        merged.weight_loader(merged.weight, loaded, weight_id)
    merged_local_sizes = [size // world_size for size in merged_sizes]
    merged_actual = gather_repacked(merged(x), merged_local_sizes, world_size)
    merged_expected = torch.cat([F.linear(x, w) for w in merged_weights], dim=-1)
    report("MergedColumnParallel", merged_actual, merged_expected, rank)

    # QKVColumnParallelLinear: gather Q, K, and V groups independently.
    head_size = 2
    num_heads = 4 * world_size
    num_kv_heads = 2 * world_size
    qkv_sizes = [
        head_size * num_heads,
        head_size * num_kv_heads,
        head_size * num_kv_heads,
    ]
    qkv_weights = [
        torch.randn(size, input_size, generator=generator).to(device)
        for size in qkv_sizes
    ]
    qkv = QKVColumnParallelLinear(
        input_size, head_size, num_heads, num_kv_heads, bias=False
    ).to(device)
    for weight_id, loaded in zip(("q", "k", "v"), qkv_weights):
        qkv.weight_loader(qkv.weight, loaded, weight_id)
    qkv_local_sizes = [size // world_size for size in qkv_sizes]
    qkv_actual = gather_repacked(qkv(x), qkv_local_sizes, world_size)
    qkv_expected = torch.cat([F.linear(x, w) for w in qkv_weights], dim=-1)
    report("QKVColumnParallel", qkv_actual, qkv_expected, rank)

    # RowParallelLinear: input is sharded; forward() performs all_reduce.
    row_output_size = 7
    row_weight = torch.randn(
        row_output_size, input_size, generator=generator
    ).to(device)
    row_bias = torch.randn(row_output_size, generator=generator).to(device)
    row = RowParallelLinear(input_size, row_output_size).to(device)
    row.weight_loader(row.weight, row_weight)
    row.bias_loader(row.bias, row_bias)
    shard_size = input_size // world_size
    x_local = x.narrow(-1, rank * shard_size, shard_size)
    report("RowParallel", row(x_local), F.linear(x, row_weight, row_bias), rank)

    dist.barrier()
    dist.destroy_process_group()


def cpu_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    run_checks(rank, world_size, torch.device("cpu"))


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spawn-cpu",
        type=int,
        metavar="PROCESSES",
        help="spawn this many local Gloo processes without torchrun",
    )
    args = parser.parse_args()
    if args.spawn_cpu is not None:
        if args.spawn_cpu < 1:
            parser.error("--spawn-cpu must be positive")
        mp.spawn(
            cpu_worker,
            args=(args.spawn_cpu, free_local_port()),
            nprocs=args.spawn_cpu,
            join=True,
        )
        return

    rank, world_size, device = init_distributed()
    run_checks(rank, world_size, device)


if __name__ == "__main__":
    main()
