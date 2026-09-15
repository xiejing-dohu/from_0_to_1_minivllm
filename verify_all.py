"""Run the complete Day3-Day13 correctness gate from the repository root."""
import argparse
import subprocess
import sys
import unittest


TEST_MODULES = [f"day{day}.test_{name}" for day, name in (
    (3,"linear"),(4,"embedding_head"),(5,"kv_cache"),(6,"prefill_attention"),
    (7,"paged_attention"),(8,"rotary_embedding"),(9,"qwen3"),(10,"block_manager"),
    (11,"model_runner"),(12,"scheduler"),(13,"engine"),
)]


def run_module(module,*arguments):
    subprocess.run([sys.executable,"-m",module,*arguments],check=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--real-model",action="store_true",help="also load the locally cached Qwen3-0.6B checkpoint")
    args=parser.parse_args(); suite=unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
    run_module("day3.distributed_check","--spawn-cpu","2")
    run_module("day4.distributed_check","--processes","2")
    run_module("day13.distributed_model_check","--processes","2")
    if args.real_model:
        run_module("day9.verify_qwen3_06b"); run_module("day13.verify_engine_qwen3_06b")


if __name__=="__main__": main()
