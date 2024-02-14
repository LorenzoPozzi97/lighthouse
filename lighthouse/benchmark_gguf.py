# TODO
# offload based on the memory I want free
# track memory peak/usage

import gc
import os
from utils import *
import torch
import psutil
import argparse
import datetime
import llama_cpp
import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from typing import Dict
from memory_tracker import MemoryTracker
from llama_cpp import Llama, llama_get_timings, llama_free

np.random.seed(101)

MODELS_DIR = os.path.join(os.path.expanduser("~"), 'models')
#df = pd.read_csv('bulb.csv')

RUN_NAME = random_noun_adjective()
print(RUN_NAME)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='GGUF benchmarking',
        description='Generate inference stats for GGUF quantized models.'
    )
    parser.add_argument('--model', action='store', type=str, help="Path of Model to benchmark.")
    parser.add_argument('--n-threads', action='store',  nargs='+', type=int, default=[10], help="Number of threads to use for generation.")
    parser.add_argument('--n-threads_batch', action='store',  nargs='+', type=int, default=[10], help="Number of threads to use during batch and prompt processing.")
    parser.add_argument('--n-batch', action='store',  nargs='+', type=int, default=[512], help="Prompt processing maximum batch size.")
    parser.add_argument('--ngl', action='store', nargs='+', type=float, default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], help="Percentage of layers to store in VRAM.")
    parser.add_argument('--force', action='store_true', default=False, help="False the test even if the config already exists.")
    parser.add_argument("--prompt-length", nargs='+', type=int, default=[100], help="Length of the prompt.")
    parser.add_argument("--new-tokens", nargs='+', type=int, default=[100], help="Length of the generation.")
    parser.add_argument("--ctx", type=int, default=1100, help="Context length.")
    parser.add_argument("--replica", type=int, default=1, help="Number of repeated experiments.")
    parser.add_argument("--memo", type=str, default='', help="Description of the experiment.")
    parser.add_argument("--debug", action='store_true', default=False, help="Reslts won't be stored.")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose.")
    return parser.parse_args()

# def load_model(model_path: str,
#                n_gpu_layers: int,
#                n_batch: int,
#                n_threads: int,
#                n_threads_batch: int,
#                n_ctx: int,
#                verbose: bool) -> Tuple[Llama, float]:

#     model = Llama(model_path=model_path,
#                  n_gpu_layers=n_gpu_layers,
#                  n_batch=n_batch,
#                  n_threads=n_threads,
#                  n_threads_batch=n_threads_batch,
#                  n_ctx=n_ctx,
#                  verbose=verbose)

#     load_time = llama_get_timings(model._ctx.ctx).t_load_ms / 1000

#     return model, load_time
def get_timings(llm: Llama) -> Dict[str, float]:
    timings = llama_get_timings(llm._ctx.ctx)

    return {
        "Load Time (s)": timings.t_load_ms / 1000,
        #"Sample Time (s)": timings.t_sample_ms / 1000,
        "Prefill Time (s)": timings.t_p_eval_ms / 1000,
        "Decode Time (s)": timings.t_eval_ms / 1000,
        "Latency (s)": (timings.t_end_ms - timings.t_start_ms) / 1000,
        #"Sample Time (Tk/s)": (timings.n_sample * 1000) / timings.t_sample_ms,
        "Prefill Time (tk/s)": (timings.n_p_eval * 1000) / timings.t_p_eval_ms,
        "Decode Time (tk/s)": (timings.n_eval * 1000) / timings.t_eval_ms,
        "Latency (tk/s)": ((timings.n_p_eval + timings.n_eval) * 1000) / (timings.t_end_ms - timings.t_start_ms)
    }

def benchmark_gguf(model: Llama,
                   prompt_length: int,
                   new_tokens: int,
                   replica: int,
                   memory_tracker: MemoryTracker) -> Dict[str, int | str]:

    torch.cuda.empty_cache()
    gc.collect()
    
    prompt = np.random.randint(1, model._model.n_vocab(), size=prompt_length).tolist()

    timings = {
        "Load Time (s)": [],
        #"Sample Time (s)": [],
        "Prefill Time (s)": [],
        "Decode Time (s)": [],
        "Latency (s)": [],
        #"Sample Time (Tk/s)": [],
        "Prefill Time (tk/s)": [],
        "Decode Time (tk/s)": [],
        "Latency (tk/s)": []
    }

    
    for r in range(replica):
        if r == 0:
            with memory_tracker.track():
                output = model(
                    prompt,
                    max_tokens=new_tokens,
                    logit_bias={model._token_eos: float('-inf')},
                    temperature=0.3,
                    seed=101
                    )
                torch.cuda.synchronize()
                memory_stats = torch.cuda.memory_stats()
        else:
            output = model(
                prompt,
                max_tokens=new_tokens,
                logit_bias={model._token_eos: float('-inf')},
                temperature=0.3,
                seed=101
                )
        model_timings = get_timings(model)

        for key in timings:
            timings[key].append(model_timings[key])

        # Manually reset the model state to repeat the operation
        model.reset()

    #     torch.cuda.synchronize()

    # memory_stats = torch.cuda.memory_stats()

    peak_allocated_torch_mb = memory_stats["allocated_bytes.all.peak"] * 1e-6
    peak_reserved_torch_mb = memory_stats["reserved_bytes.all.peak"] * 1e-6
    peak_nvml_mb = memory_tracker.peak_memory
    peak_external_mb = peak_nvml_mb - peak_reserved_torch_mb
    peak_memory_mb = peak_allocated_torch_mb + peak_external_mb

    def average_timings(lst):
        return round(sum(lst) / len(lst), 2)

    averaged_timings = {key: average_timings(value) for key, value in timings.items()}
    run_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    return {
        "id": output["id"],
        "run_time": run_time,
        **averaged_timings,
        "Mem. Usage (GB)": f'{peak_memory_mb / 1000:.2f}'
    }

def main():
    args = parse_arguments()
    columns = ['Device',
               'CPU Count',
               'Model',
               'Prompt Length',
               'New Tokens',
               'llama_cpp_v',
               'Context Length',
               'Decode Threads',
               'Prefill Threads',
               'GPU Layers',
               'auto_gptq_v',
               'Num. Requests',
               'Kernel']

    bulb = initialize_bulb(columns) if not os.path.exists('bulb.csv') else pd.read_csv('bulb.csv')
    model_path = os.path.join(MODELS_DIR, args.model)
    memory_tracker = MemoryTracker()

    if not torch.cuda.is_available():
        args.ngl = [0]

    param_combinations = {
        "Decode Threads": args.n_threads,
        "Prefill Threads": args.n_threads_batch,
        "Batch Size": args.n_batch,
        "GPU Layers": args.ngl,
        "Prompt Length": args.prompt_length, 
        "New Tokens": args.new_tokens,
        "Context Length": [args.ctx]
    }

    if check_experiment(bulb, param_combinations, model_path):
        return None

    param_combinations = list(itertools.product(
        args.n_threads,
        args.n_threads_batch,
        args.n_batch,
        args.ngl,
        args.prompt_length, 
        args.new_tokens
    ))

    for params in tqdm(param_combinations, total=len(param_combinations), desc='Running experiments'):
        n_threads, n_threads_batch, n_batch, ngl, prompt_length, new_tokens = params
        int_ngl = int((int(Llama(model_path=model_path, verbose=False).metadata['llama.block_count'])+1)*ngl) # convert to the number of layers to offload

        model = Llama(model_path=model_path, n_gpu_layers=int_ngl, n_batch=n_batch, n_threads=n_threads, n_threads_batch=n_threads_batch, n_ctx=args.ctx, verbose=False)

        diagnostics = benchmark_gguf(model,
                           prompt_length - 1,
                           new_tokens,
                           args.replica,
                           memory_tracker)
        
        experiment = {
            # experiment configuration
            "memo": args.memo,
            "Run Name": RUN_NAME,
            # software configuration
            "llama_cpp_v": llama_cpp.__version__,
            # hardware configuration
            "Device": torch.cuda.get_device_name(),
            "VRAM (GB)": round(torch.cuda.get_device_properties('cuda').total_memory / 1024 / 1024 / 1024, 2),
            "RAM (GB)": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
            "CPU Count": len(psutil.Process().cpu_affinity()),
            # model configuration
            "Quant. Method": 'gguf',
            "Model": os.path.basename(model_path),
            "Model Size (GB)": round(os.path.getsize(model_path) / 1024 / 1024 / 1024, 2),
            "Batch Size": n_batch,
            "Context Length": args.ctx,
            "Decode Threads": n_threads,
            "Prefill Threads": n_threads_batch,
            "Prompt Length": prompt_length,
            "New Tokens": new_tokens,
            "GPU Layers": ngl,
            **diagnostics
            }

        if not args.debug:
            bulb = pd.concat([bulb, pd.DataFrame([experiment])], ignore_index=True)

    if not args.debug:
        bulb.to_csv('bulb.csv', index=False)

if __name__ == "__main__":
    print('Thinking...')
    main()
    print('Done')
