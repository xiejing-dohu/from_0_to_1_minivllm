import unittest

import torch
import torch.nn.functional as F

from day3.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


class TensorParallelLinearTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.x = torch.randn(3, 8)

    def test_replicated_linear_matches_pytorch(self) -> None:
        weight = torch.randn(6, 8)
        bias = torch.randn(6)
        layer = ReplicatedLinear(8, 6)
        layer.weight.weight_loader(layer.weight, weight)
        layer.bias.weight_loader(layer.bias, bias)
        torch.testing.assert_close(layer(self.x), F.linear(self.x, weight, bias))

    def test_column_parallel_two_virtual_ranks(self) -> None:
        weight = torch.randn(6, 8)
        bias = torch.randn(6)
        outputs = []
        for rank in range(2):
            layer = ColumnParallelLinear(8, 6, tp_rank=rank, tp_size=2)
            layer.weight.weight_loader(layer.weight, weight)
            layer.bias.weight_loader(layer.bias, bias)
            outputs.append(layer(self.x))
        actual = torch.cat(outputs, dim=-1)
        torch.testing.assert_close(actual, F.linear(self.x, weight, bias))

    def test_merged_column_parallel_layout(self) -> None:
        gate_weight = torch.randn(6, 8)
        up_weight = torch.randn(4, 8)
        rank_outputs = []
        for rank in range(2):
            layer = MergedColumnParallelLinear(
                8, [6, 4], bias=False, tp_rank=rank, tp_size=2
            )
            layer.weight.weight_loader(layer.weight, gate_weight, 0)
            layer.weight.weight_loader(layer.weight, up_weight, 1)
            rank_outputs.append(layer(self.x))

        gate_local, up_local = 3, 2
        gate = torch.cat([part[:, :gate_local] for part in rank_outputs], dim=-1)
        up = torch.cat([part[:, gate_local : gate_local + up_local] for part in rank_outputs], dim=-1)
        actual = torch.cat([gate, up], dim=-1)
        expected = torch.cat(
            [F.linear(self.x, gate_weight), F.linear(self.x, up_weight)], dim=-1
        )
        torch.testing.assert_close(actual, expected)

    def test_qkv_parallel_keeps_complete_heads(self) -> None:
        head_size, num_heads, num_kv_heads = 2, 4, 2
        q_weight = torch.randn(head_size * num_heads, 8)
        k_weight = torch.randn(head_size * num_kv_heads, 8)
        v_weight = torch.randn(head_size * num_kv_heads, 8)
        rank_outputs = []
        for rank in range(2):
            layer = QKVColumnParallelLinear(
                8,
                head_size,
                num_heads,
                num_kv_heads,
                tp_rank=rank,
                tp_size=2,
            )
            layer.weight.weight_loader(layer.weight, q_weight, "q")
            layer.weight.weight_loader(layer.weight, k_weight, "k")
            layer.weight.weight_loader(layer.weight, v_weight, "v")
            rank_outputs.append(layer(self.x))
            self.assertEqual(layer.local_num_heads, 2)
            self.assertEqual(layer.local_num_kv_heads, 1)

        q_local = head_size * num_heads // 2
        kv_local = head_size * num_kv_heads // 2
        q = torch.cat([part[:, :q_local] for part in rank_outputs], dim=-1)
        k = torch.cat(
            [part[:, q_local : q_local + kv_local] for part in rank_outputs], dim=-1
        )
        v = torch.cat([part[:, q_local + kv_local :] for part in rank_outputs], dim=-1)
        actual = torch.cat([q, k, v], dim=-1)
        expected = torch.cat(
            [
                F.linear(self.x, q_weight),
                F.linear(self.x, k_weight),
                F.linear(self.x, v_weight),
            ],
            dim=-1,
        )
        torch.testing.assert_close(actual, expected)

    def test_row_parallel_sums_partial_outputs_then_adds_bias_once(self) -> None:
        weight = torch.randn(5, 8)
        bias = torch.randn(5)
        partials = []
        for rank in range(2):
            layer = RowParallelLinear(8, 5, tp_rank=rank, tp_size=2)
            layer.weight.weight_loader(layer.weight, weight)
            layer.bias.weight_loader(layer.bias, bias)
            x_shard = self.x[:, rank * 4 : (rank + 1) * 4]
            partials.append(layer.forward_local(x_shard))
        actual = sum(partials) + bias
        torch.testing.assert_close(actual, F.linear(self.x, weight, bias))

    def test_invalid_partition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_size"):
            ColumnParallelLinear(8, 5, tp_rank=0, tp_size=2)
        with self.assertRaisesRegex(ValueError, "head counts"):
            QKVColumnParallelLinear(8, 2, 3, 2, tp_rank=0, tp_size=2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_single_rank_cuda_matches_pytorch(self) -> None:
        x = self.x.cuda()
        weight = torch.randn(6, 8, device="cuda")
        bias = torch.randn(6, device="cuda")
        layer = ColumnParallelLinear(8, 6).cuda()
        layer.weight_loader(layer.weight, weight)
        layer.bias_loader(layer.bias, bias)
        torch.testing.assert_close(layer(x), F.linear(x, weight, bias))


if __name__ == "__main__":
    unittest.main()
