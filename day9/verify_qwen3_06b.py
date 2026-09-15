import torch
from transformers import AutoModelForCausalLM
from day9.modeling_qwen3 import Qwen3Config,Qwen3ForCausalLM,load_hf_state_dict


def main():
    name="Qwen/Qwen3-0.6B"; device="cuda" if torch.cuda.is_available() else "cpu"
    hf=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16 if device=="cuda" else torch.float32,local_files_only=True).to(device).eval()
    ours=Qwen3ForCausalLM(Qwen3Config.from_hf(hf.config)).to(device=device,dtype=next(hf.parameters()).dtype).eval()
    ignored=load_hf_state_dict(ours,hf.state_dict())
    ids=torch.tensor([[1,9707,11,487,30]],device=device)
    with torch.no_grad(): expected=hf(ids,use_cache=False).logits; actual=ours(ids)
    actual_float,expected_float=actual.float(),expected.float()
    error=(actual_float-expected_float).abs()
    print(f"device={device} shape={tuple(actual.shape)}")
    print(f"max_abs_error={error.max().item():.6f} mean_abs_error={error.mean().item():.6f}")
    print(f"allclose={torch.allclose(actual_float,expected_float,rtol=2e-2,atol=4e-2)} top1_match={torch.equal(actual.argmax(-1),expected.argmax(-1))} ignored={sorted(ignored)}")


if __name__=="__main__": main()
