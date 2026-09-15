import statistics

import torch

from day6.layers.prefill_attention import prefill_attention, prefill_attention_reference


def measure(function, args, repeats=6):
    for _ in range(3):
        function(*args)
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        function(*args)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.mean(values), statistics.pstdev(values)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print("Qwen3-0.6B attention parameters: q_heads=16, kv_heads=8, head_dim=128")
    print("Each result uses 3 warmups and 6 timed runs (float16).")
    for lengths in ([128], [256], [128, 384]):
        total = sum(lengths)
        q = torch.randn(total, 16, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(total, 8, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        reference = measure(prefill_attention_reference, (q, k, v, cu, cu))
        triton = measure(lambda *a: prefill_attention(*a, backend="triton"), (q, k, v, cu, cu))
        print(f"lengths={str(lengths):<12} reference={reference[0]:8.3f}±{reference[1]:.3f} ms  triton={triton[0]:8.3f}±{triton[1]:.3f} ms")


if __name__ == "__main__":
    main()
