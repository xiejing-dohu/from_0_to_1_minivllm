import statistics,time
import torch
from transformers import AutoModelForCausalLM
from day9.modeling_qwen3 import Qwen3Config,Qwen3ForCausalLM,load_hf_state_dict
from day10.engine import SamplingParams
from day13.engine import LLMEngine,PagedQwen3Runner


def main():
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    hf=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",dtype=torch.float16,local_files_only=True).cuda().eval()
    model=Qwen3ForCausalLM(Qwen3Config.from_hf(hf.config)).half().cuda().eval(); load_hf_state_dict(model,hf.state_dict())
    prompt=[1,9707,11,487,30]; count=2
    def run_hf():
        baseline=[]; ids=list(prompt)
        for _ in range(count):
            token=int(hf(torch.tensor([ids],device="cuda"),use_cache=False).logits[0,-1].argmax()); ids.append(token); baseline.append(token)
        return baseline
    runner=PagedQwen3Runner(model,32,16,"cuda",greedy=True); engine=LLMEngine(runner,4,256,32,16,151645)
    def run_engine():
        current=LLMEngine(runner,4,256,32,16,151645)
        return current.generate_token_ids([prompt],SamplingParams(max_tokens=count))[0],current
    run_hf(); run_engine()  # warm up kernels and Triton compilation
    hf_times=[]; engine_times=[]
    for _ in range(6):
        torch.cuda.synchronize(); start=time.perf_counter(); baseline=run_hf(); torch.cuda.synchronize(); hf_times.append((time.perf_counter()-start)*1000)
        torch.cuda.synchronize(); start=time.perf_counter(); actual,engine=run_engine(); torch.cuda.synchronize(); engine_times.append((time.perf_counter()-start)*1000)
    print(f"baseline_tokens={baseline} engine_tokens={actual} match={baseline==actual}")
    print(f"HF full-recompute={statistics.mean(hf_times):.3f}±{statistics.pstdev(hf_times):.3f} ms  paged engine={statistics.mean(engine_times):.3f}±{statistics.pstdev(engine_times):.3f} ms")
    print(f"blocks_free={len(engine.scheduler.block_manager.free)} blocks_used={len(engine.scheduler.block_manager.used)}")


if __name__=="__main__": main()
