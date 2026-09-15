import unittest
import torch
from day9.modeling_qwen3 import Qwen3Config,Qwen3ForCausalLM,load_hf_state_dict


class Qwen3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config as HFConfig,Qwen3ForCausalLM as HFModel
        torch.manual_seed(2026)
        cls.hfc=HFConfig(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=128,rope_parameters={"rope_type":"default","rope_theta":1_000_000.0},rms_norm_eps=1e-6,attention_bias=False,tie_word_embeddings=True)
        cls.hf=HFModel(cls.hfc).eval(); cls.ours=Qwen3ForCausalLM(Qwen3Config.from_hf(cls.hfc)).eval()
        load_hf_state_dict(cls.ours,cls.hf.state_dict())

    def test_merged_parameter_shapes(self):
        layer=self.ours.model.layers[0]
        self.assertEqual(tuple(layer.self_attn.qkv_proj.weight.shape),(128,64))
        self.assertEqual(tuple(layer.mlp.gate_up_proj.weight.shape),(256,64))

    def test_tied_embedding_is_same_parameter(self):
        self.assertIs(self.ours.model.embed_tokens.weight,self.ours.lm_head.weight)

    def test_logits_match_hugging_face(self):
        ids=torch.tensor([[1,5,9,2,8,4,3],[6,2,7,1,4,8,5]])
        with torch.no_grad(): expected=self.hf(ids,use_cache=False).logits; actual=self.ours(ids)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)

    def test_loader_rejects_unmapped_key(self):
        state=dict(self.hf.state_dict()); state["unexpected.weight"]=torch.ones(1)
        fresh=Qwen3ForCausalLM(Qwen3Config.from_hf(self.hfc))
        with self.assertRaisesRegex(ValueError,"unmapped"): load_hf_state_dict(fresh,state)


if __name__ == "__main__": unittest.main()
