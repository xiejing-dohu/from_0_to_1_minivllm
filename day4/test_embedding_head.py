import unittest

import torch
import torch.nn.functional as F

from day4.layers.embedding_head import ParallelLMHead, VocabParallelEmbedding
from day4.layers.sampler import Sampler


class EmbeddingAndHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.vocab_size = 7
        self.hidden_size = 4
        self.weight = torch.randn(self.vocab_size, self.hidden_size)

    def make_embedding(self, rank: int) -> VocabParallelEmbedding:
        layer = VocabParallelEmbedding(
            self.vocab_size, self.hidden_size, tp_rank=rank, tp_size=2
        )
        layer.weight_loader(layer.weight, self.weight)
        return layer

    def test_padding_and_weight_loading(self) -> None:
        first = self.make_embedding(0)
        second = self.make_embedding(1)
        self.assertEqual(first.padded_num_embeddings, 8)
        self.assertEqual(tuple(first.weight.shape), (4, 4))
        torch.testing.assert_close(first.weight, self.weight[:4])
        torch.testing.assert_close(second.weight[:3], self.weight[4:])
        torch.testing.assert_close(second.weight[3], torch.zeros(4))

    def test_virtual_rank_embedding_sum_matches_full_embedding(self) -> None:
        input_ids = torch.tensor([[0, 4, 6], [2, 5, 1]], dtype=torch.long)
        actual = sum(self.make_embedding(rank).forward_local(input_ids) for rank in range(2))
        torch.testing.assert_close(actual, F.embedding(input_ids, self.weight))

    def test_invalid_token_and_dtype_are_rejected(self) -> None:
        layer = self.make_embedding(0)
        with self.assertRaisesRegex(ValueError, "outside"):
            layer.forward_local(torch.tensor([7], dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "long"):
            layer.forward_local(torch.tensor([1.0]))

    def test_virtual_rank_lm_head_matches_full_linear(self) -> None:
        hidden = torch.randn(3, self.hidden_size)
        parts = []
        for rank in range(2):
            head = ParallelLMHead(
                self.vocab_size, self.hidden_size, tp_rank=rank, tp_size=2
            )
            head.weight_loader(head.weight, self.weight)
            parts.append(head.forward_local(hidden))
        actual = torch.cat(parts, dim=-1)[..., : self.vocab_size]
        torch.testing.assert_close(actual, F.linear(hidden, self.weight))

    def test_prefill_selects_last_query_of_each_sequence(self) -> None:
        head = ParallelLMHead(self.vocab_size, self.hidden_size)
        head.weight_loader(head.weight, self.weight)
        hidden = torch.randn(5, self.hidden_size)
        cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
        actual = head.forward_local(hidden, cu_seqlens_q)
        expected = F.linear(hidden[[1, 4]], self.weight)
        torch.testing.assert_close(actual, expected)

    def test_tied_weights_share_the_same_parameter(self) -> None:
        embedding = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        head = ParallelLMHead(self.vocab_size, self.hidden_size)
        head.tie_weights(embedding)
        self.assertIs(head.weight, embedding.weight)
        self.assertEqual(head.weight.data_ptr(), embedding.weight.data_ptr())


class SamplerTests(unittest.TestCase):
    def test_sampler_returns_one_valid_token_without_mutating_logits(self) -> None:
        torch.manual_seed(3)
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
        original = logits.clone()
        output = Sampler()(logits, torch.tensor([0.7, 1.2]))
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.all((0 <= output) & (output < 3)))
        torch.testing.assert_close(logits, original)

    def test_sampler_rejects_invalid_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            Sampler()(torch.randn(1, 3), torch.tensor([0.0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_embedding_head_and_sampler_on_cuda(self) -> None:
        vocab_size, hidden_size = 7, 4
        weight = torch.randn(vocab_size, hidden_size, device="cuda")
        embedding = VocabParallelEmbedding(vocab_size, hidden_size).cuda()
        embedding.weight_loader(embedding.weight, weight)
        input_ids = torch.tensor([0, 3, 6], device="cuda")
        hidden = embedding(input_ids)

        head = ParallelLMHead(vocab_size, hidden_size).cuda()
        head.tie_weights(embedding)
        logits = head(hidden)
        tokens = Sampler().cuda()(logits, torch.ones(3, device="cuda"))
        torch.testing.assert_close(hidden, F.embedding(input_ids, weight))
        self.assertEqual(tuple(tokens.shape), (3,))


if __name__ == "__main__":
    unittest.main()
