import unittest
import torch
from day8.layers.rotary_embedding import QKRMSNorm, RotaryEmbedding


class RotaryTests(unittest.TestCase):
    def test_position_zero_is_identity(self):
        rope=RotaryEmbedding(8,max_position=16); q=torch.randn(3,4,8); k=torch.randn(3,2,8)
        qr,kr=rope(torch.zeros(3,dtype=torch.long),q,k)
        torch.testing.assert_close(qr,q); torch.testing.assert_close(kr,k)

    def test_packed_and_batched_positions(self):
        rope=RotaryEmbedding(8,max_position=16); q=torch.randn(2,3,4,8); k=torch.randn(2,3,2,8)
        positions=torch.tensor([0,2,5]); qb,kb=rope(positions,q,k)
        qp,kp=rope(positions.repeat(2),q.flatten(0,1),k.flatten(0,1))
        torch.testing.assert_close(qb.flatten(0,1),qp); torch.testing.assert_close(kb.flatten(0,1),kp)

    def test_partial_rotary_preserves_tail(self):
        rope=RotaryEmbedding(8,rotary_dim=4,max_position=8); x=torch.randn(2,1,8)
        out,_=rope(torch.tensor([1,2]),x,x)
        torch.testing.assert_close(out[...,4:],x[...,4:])

    def test_qk_rmsnorm_matches_formula(self):
        x=torch.randn(2,4,8); norm=QKRMSNorm(8,eps=1e-6); norm.weight.data.uniform_()
        expected=(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)*norm.weight).to(x.dtype)
        torch.testing.assert_close(norm(x),expected)

    def test_matches_hugging_face_qwen3_helper(self):
        try: from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        except (ImportError,ModuleNotFoundError): self.skipTest("Qwen3 helper unavailable")
        rope=RotaryEmbedding(8,max_position=16,base=1_000_000.0)
        q=torch.randn(2,3,4,8); k=torch.randn(2,3,2,8); pos=torch.tensor([[0,1,4],[2,5,7]])
        actual_q,actual_k=rope(pos,q,k)
        half_cos=rope.cos_cache[pos]; half_sin=rope.sin_cache[pos]
        cos=torch.cat((half_cos,half_cos),-1); sin=torch.cat((half_sin,half_sin),-1)
        expected_q,expected_k=apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),cos,sin,unsqueeze_dim=1)
        torch.testing.assert_close(actual_q,expected_q.transpose(1,2)); torch.testing.assert_close(actual_k,expected_k.transpose(1,2))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA is not available")
    def test_cuda_half_preserves_dtype_and_device(self):
        rope=RotaryEmbedding(32,max_position=32).cuda(); q=torch.randn(5,4,32,device="cuda",dtype=torch.float16); k=torch.randn(5,2,32,device="cuda",dtype=torch.float16)
        qr,kr=rope(torch.arange(5,device="cuda"),q,k)
        self.assertEqual(qr.dtype,torch.float16); self.assertTrue(kr.is_cuda)


if __name__ == "__main__": unittest.main()
