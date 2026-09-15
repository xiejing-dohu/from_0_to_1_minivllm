import torch
import torch.nn.functional as F

from day4.layers.embedding_head import ParallelLMHead, VocabParallelEmbedding
from day4.layers.sampler import Sampler


def main() -> None:
    torch.manual_seed(7)
    vocab_size, hidden_size, tp_size = 7, 4, 2
    full_weight = torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(
        vocab_size, hidden_size
    )
    input_ids = torch.tensor([[0, 4, 6], [2, 5, 1]], dtype=torch.long)

    local_embeddings = []
    local_logits = []
    hidden = torch.randn(2, hidden_size)
    for rank in range(tp_size):
        embedding = VocabParallelEmbedding(
            vocab_size, hidden_size, tp_rank=rank, tp_size=tp_size
        )
        embedding.weight_loader(embedding.weight, full_weight)
        local_embeddings.append(embedding.forward_local(input_ids))

        head = ParallelLMHead(
            vocab_size, hidden_size, tp_rank=rank, tp_size=tp_size
        )
        head.tie_weights(embedding)
        local_logits.append(head.forward_local(hidden))

    embedding_output = sum(local_embeddings)
    logits = torch.cat(local_logits, dim=-1)[..., :vocab_size]
    reference_embedding = F.embedding(input_ids, full_weight)
    reference_logits = F.linear(hidden, full_weight)

    print("padded vocab size   =", 8)
    print("local weight shapes =", [(4, 4), (4, 4)])
    print("embedding allclose  =", torch.allclose(embedding_output, reference_embedding))
    print("LM head allclose    =", torch.allclose(logits, reference_logits))

    sampler = Sampler()
    token_ids = sampler(logits, torch.tensor([0.7, 1.0]))
    print("sampled token ids   =", token_ids.tolist())


if __name__ == "__main__":
    main()
