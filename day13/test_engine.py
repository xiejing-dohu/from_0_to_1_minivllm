import unittest
import torch
from transformers import Qwen3Config as HFConfig,Qwen3ForCausalLM as HFModel
from day9.modeling_qwen3 import Qwen3Config,Qwen3ForCausalLM,load_hf_state_dict
from day10.engine import SamplingParams
from day13.engine import LLMEngine,PagedQwen3Runner


class EngineTests(unittest.TestCase):
    def make_models(self):
        torch.manual_seed(2026); c=HFConfig(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=128,rope_parameters={"rope_type":"default","rope_theta":1_000_000.0},rms_norm_eps=1e-6,attention_bias=False,tie_word_embeddings=True)
        hf=HFModel(c).eval(); ours=Qwen3ForCausalLM(Qwen3Config.from_hf(c)).eval(); load_hf_state_dict(ours,hf.state_dict()); return hf,ours
    def baseline(self,model,prompt,count):
        ids=list(prompt); out=[]; device=next(model.parameters()).device
        for _ in range(count):
            token=int(model(torch.tensor([ids],device=device),use_cache=False).logits[0,-1].argmax()); ids.append(token); out.append(token)
        return out
    def test_end_to_end_tokens_match_hugging_face(self):
        hf,model=self.make_models(); runner=PagedQwen3Runner(model,32,4,greedy=True); engine=LLMEngine(runner,4,64,32,4,96)
        prompts=[[1,5,9,2,8],[6,2,7]]; params=[SamplingParams(max_tokens=3),SamplingParams(max_tokens=2)]
        actual=engine.generate_token_ids(prompts,params); expected=[self.baseline(hf,p,len(actual[i])) for i,p in enumerate(prompts)]
        self.assertEqual(actual,expected); self.assertTrue(engine.scheduler.is_finished); self.assertEqual(len(engine.scheduler.block_manager.used),0)
    def test_prefix_cache_is_used_by_later_request(self):
        hf,model=self.make_models(); runner=PagedQwen3Runner(model,16,4,greedy=True); engine=LLMEngine(runner,2,32,16,4,96)
        first=engine.add_request([1,2,3,4],SamplingParams(max_tokens=1)); engine.step()
        second=engine.add_request([1,2,3,4,5],SamplingParams(max_tokens=1)); batch,prefill=engine.scheduler.schedule()
        self.assertTrue(prefill); self.assertEqual(batch,[second]); self.assertEqual(second.num_cached_tokens,4)
        tokens=runner.run(batch,prefill); engine.scheduler.postprocess(batch,tokens)
        self.assertEqual(tokens,self.baseline(hf,[1,2,3,4,5],1))
    @unittest.skipUnless(torch.cuda.is_available(),"CUDA is unavailable")
    def test_cuda_triton_end_to_end_matches_hugging_face(self):
        hf,model=self.make_models(); hf=hf.half().cuda(); model=model.half().cuda()
        runner=PagedQwen3Runner(model,16,4,"cuda",greedy=True); engine=LLMEngine(runner,2,32,16,4,96)
        prompt=[1,5,9,2,8]; actual=engine.generate_token_ids([prompt],SamplingParams(max_tokens=2))[0]
        self.assertEqual(actual,self.baseline(hf,prompt,2))

    def test_string_tokenizer_api(self):
        class TinyTokenizer:
            def encode(self,text): return [1,5,9] if text=="hello" else [2]
            def decode(self,tokens): return " ".join(map(str,tokens))
        _,model=self.make_models(); runner=PagedQwen3Runner(model,8,4,greedy=True)
        engine=LLMEngine(runner,2,16,8,4,96,tokenizer=TinyTokenizer())
        output=engine.generate(["hello"],SamplingParams(max_tokens=1))
        self.assertEqual(len(output),1); self.assertRegex(output[0],r"^\d+$")


if __name__=="__main__": unittest.main()
