import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    Split the last dimension into x and y, then compute SiLU(x) * y.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

if __name__ == "__main__":
    # Example usage
    layer = SiluAndMul().cuda()
    # input_tensor = torch.randn(200, 400).cuda()
    # input_tensor = torch.randn(2000, 4000).cuda()
    input_tensor = torch.randn(2, 2000, 4000).cuda()

    for _ in range(10):  # Warm-up iterations
        _ = layer(input_tensor)

    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
