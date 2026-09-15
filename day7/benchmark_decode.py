import statistics
import torch
from day7.layers.paged_attention import paged_attention_decode, paged_attention_reference


def measure(fn, args):
    for _ in range(3): fn(*args)
    torch.cuda.synchronize(); times=[]
    for _ in range(6):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); fn(*args); b.record(); b.synchronize(); times.append(a.elapsed_time(b))
    return statistics.mean(times), statistics.pstdev(times)


def main():
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    print("Qwen3-0.6B: q_heads=16, kv_heads=8, head_dim=128, block_size=16")
    print("3 warmups + 6 timed runs, float16, random physical block tables")
    for batch, length in ((1,128),(4,512),(4,1024)):
        blocks_per=(length+15)//16; total=batch*blocks_per+7
        q=torch.randn(batch,16,128,device="cuda",dtype=torch.float16)
        kc=torch.randn(total,16,8,128,device="cuda",dtype=torch.float16); vc=torch.randn_like(kc)
        generator=torch.Generator().manual_seed(2026+length)
        perm=torch.randperm(total,generator=generator)[:batch*blocks_per].reshape(batch,blocks_per).cuda()
        lens=torch.full((batch,),length,device="cuda",dtype=torch.int32)
        args=(q,kc,vc,perm,lens)
        ref=measure(paged_attention_reference,args)
        tri=measure(lambda *x:paged_attention_decode(*x,backend="triton"),args)
        print(f"batch={batch} context={length:<4} reference={ref[0]:8.3f}±{ref[1]:.3f} ms  triton={tri[0]:8.3f}±{tri[1]:.3f} ms")


if __name__ == "__main__": main()
