import torch
import torch.nn.functional as F

from day3.layers.linear import ColumnParallelLinear, RowParallelLinear


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 8)
    weight = torch.randn(6, 8)
    bias = torch.randn(6)
    reference = F.linear(x, weight, bias)

    column_parts = []
    for rank in range(2):
        layer = ColumnParallelLinear(8, 6, tp_rank=rank, tp_size=2)
        layer.weight_loader(layer.weight, weight)
        layer.bias_loader(layer.bias, bias)
        column_parts.append(layer(x))
    column_output = torch.cat(column_parts, dim=-1)

    row_parts = []
    for rank in range(2):
        layer = RowParallelLinear(8, 6, tp_rank=rank, tp_size=2)
        layer.weight_loader(layer.weight, weight)
        layer.bias_loader(layer.bias, bias)
        row_parts.append(layer.forward_local(x[:, rank * 4 : (rank + 1) * 4]))
    row_output = sum(row_parts) + bias

    print("full output shape   =", tuple(reference.shape))
    print("column local shapes =", [tuple(part.shape) for part in column_parts])
    print("column allclose     =", torch.allclose(column_output, reference))
    print("row partial shapes  =", [tuple(part.shape) for part in row_parts])
    print("row allclose        =", torch.allclose(row_output, reference))


if __name__ == "__main__":
    main()
