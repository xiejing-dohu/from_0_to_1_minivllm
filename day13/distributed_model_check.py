import argparse,socket
import torch,torch.distributed as dist,torch.multiprocessing as mp


def worker(rank,size,port):
    dist.init_process_group("gloo",init_method=f"tcp://127.0.0.1:{port}",rank=rank,world_size=size)
    try:
        from transformers import Qwen3Config as HFC,Qwen3ForCausalLM as HFM
        from day9.modeling_qwen3 import Qwen3Config
        from day13.tp_qwen3 import TPQwen3,load_tp_hf
        torch.manual_seed(2026); hfc=HFC(vocab_size=96,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=64,rope_parameters={"rope_type":"default","rope_theta":1_000_000.0},rms_norm_eps=1e-6,attention_bias=False,tie_word_embeddings=True)
        hf=HFM(hfc).eval(); model=TPQwen3(Qwen3Config.from_hf(hfc)).eval(); load_tp_hf(model,hf.state_dict()); ids=torch.tensor([[1,5,9,2,8]])
        with torch.no_grad(): actual=model(ids)
        if rank==0:
            expected=hf(ids,use_cache=False).logits; error=(actual-expected).abs().max().item(); ok=torch.allclose(actual,expected,rtol=2e-5,atol=2e-5)
            print(f"[TP=2 Qwen3] allclose={ok} max_abs_error={error:.8f} top1_match={torch.equal(actual.argmax(-1),expected.argmax(-1))}")
            if not ok: raise AssertionError("TP model does not match Hugging Face")
    finally: dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--processes",type=int,default=2); args=parser.parse_args()
    with socket.socket() as s: s.bind(("127.0.0.1",0)); port=s.getsockname()[1]
    mp.spawn(worker,args=(args.processes,port),nprocs=args.processes,join=True)


if __name__=="__main__": main()
