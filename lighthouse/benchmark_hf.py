# Inspired from https://github.com/huggingface/optimum/blob/main/tests/benchmark/benchmark_gptq.py

import argparse
import itertools
import gc
import os
import time
import auto_gptq
from utils import *

import numpy as np
import pandas as pd
import torch
from auto_gptq.utils import Perplexity
from memory_tracker import MemoryTracker
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    GPTQConfig,
)

from optimum.exporters import TasksManager

device = torch.device("cuda:0")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='GPTQ benchmarking',
        description='Generate inference stats for GPTQ quantized models.'
    )
    parser.add_argument("--model", type=str, help="Path of Model to benchmark.")
    parser.add_argument("--replica", type=int, default=3, help="Number of repeated experiments.")
    parser.add_argument("--n-request", action='store',  nargs='+', type=int, default=[1], help="Prompt processing maximum batch size.")
    parser.add_argument("--task", type=str, default=None, help="Task")
    parser.add_argument("--prompt-length", nargs='+', type=int, default=[100], help="Length of the prompt (ratio fro te context window menus new_tokens).")
    parser.add_argument("--new-tokens", nargs='+', type=int, default=[100], help="Length of the generation.")
    parser.add_argument("--gptq", action="store_true", help="Indicate that the model to benchmark is a GPTQ model.")
    parser.add_argument("--bitsandbytes", action="store_true", help="Indicate that the model uses bitsandbytes through transformers load_in_4bit=True.")
    #parser.add_argument("--exllama-version", type=int, default=0, help="Use Exllamav2 kernel. Set 1 in order to use exllama kernel")
    #parser.add_argument("--ppl", action="store_true", help="Calculate the perplexity on wikitext2 dataset")
    parser.add_argument('--kernel', action='store', default='exllamav2', choices=['exllamav2', 'exllama', 'autotogptq-cuda', 'autogptq-cuda-old'], type=str, help="Kernel.")
    parser.add_argument("--revision", default='main', help="Revision of the model to benchmark")
    parser.add_argument("--memo", type=str, default='', help="Description of the experiment.")
    parser.add_argument("--debug", action='store_true', default=False, help="Reslts won't be stored.")

    return parser.parse_args()

def load_model(model: str,
               gptq: bool,
               bitsandbytes: bool,
               revision: str,
               use_exllama,
               exllama_version: int):
        
        load_start = time.time_ns()

        if gptq:
            quantization_config = GPTQConfig(
                bits=4, use_exllama=use_exllama, exllama_config={"version": exllama_version}
            )
            model = AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif bitsandbytes:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
        else:
            with device:
                model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True)

        torch.cuda.synchronize()

        load_end = time.time_ns()

        load_time = (load_end - load_start) * 1e-9

        return model, load_time

def warmup(
    model,
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    new_tokens: int,
    pad_token_id: int):

    gen_config = GenerationConfig(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        use_cache=True,
        pad_token_id=pad_token_id,
        num_beams=1,
        do_sample=False,
        eos_token_id=None,  # This is required for min_new_tokens to actually have an effect.
    )
    model.generation_config.eos_token_id = None  # greedy_search falls back on this eos_token_id that we need to set to None as well for min_new_tokens to have an effect.
    model.generate(input_ids, attention_mask=masks, generation_config=gen_config)

    return gen_config

def benchmark_gptq(
    model,
    n_request: int,
    prompt_length: int,
    replica: int,
    new_tokens: int,
    pad_token_id: int,
    memory_tracker: MemoryTracker,
):
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = torch.randint(1, model.config.vocab_size - 1, size=(n_request, prompt_length)).to(device)
    masks = torch.ones(n_request, prompt_length, dtype=torch.int32).to(device)

    diagnostics = {
        "Prefill Time (s)": [],
        "Latency (s)": [],
        "Prefill Time (tk/s)": [],
        "Latency (tk/s)": []
    }

    gen_config = warmup(
        model,
        input_ids,
        masks,
        1,
        pad_token_id,
    )

    print("Measuring latency...")
    assert gen_config.min_new_tokens == gen_config.max_new_tokens

    # Prefill
    for r in range(replica):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        _ = model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
        end_event.record()

        torch.cuda.synchronize()
        latency_s = start_event.elapsed_time(end_event) / 1000
        diagnostics["Prefill Time (s)"].append(latency_s)
        diagnostics["Prefill Time (tk/s)"].append(latency_s / prompt_length)

    # Tot
    torch.cuda.empty_cache()
    gc.collect()
    gen_config = warmup(
        model,
        input_ids,
        masks,
        new_tokens,
        pad_token_id,
    )

    for r in range(replica):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        if r == 0:
            with memory_tracker.track():
                _ = model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
                end_event.record()
                memory_stats = torch.cuda.memory_stats()
        else:
            _ = model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
            end_event.record()

        torch.cuda.synchronize()
        latency_s = start_event.elapsed_time(end_event) / 1000
        diagnostics["Latency (s)"].append(latency_s)
        diagnostics["Latency (tk/s)"].append(latency_s / (prompt_length + new_tokens))

    averaged_timings = {key: average_timings(value) for key, value in diagnostics.items()}
    decode_time = averaged_timings["Latency (s)"] - averaged_timings["Prefill Time (s)"] # time needed to generate all tokens as the response to the prompt (excludes all pre-processing time, and it only measures the time since it starts outputting tokens).
    per_token_decode_time = decode_time / new_tokens 
    diagnostics["Decode Time (s)"] = decode_time
    diagnostics["Decode Time (tk/s)"] = per_token_decode_time
    #throughput = new_tokens * batch_size / averaged_timings["Latency (s)"]

    peak_allocated_torch_mb = memory_stats["allocated_bytes.all.peak"] * 1e-6
    peak_reserved_torch_mb = memory_stats["reserved_bytes.all.peak"] * 1e-6
    peak_nvml_mb = memory_tracker.peak_memory
    peak_external_mb = peak_nvml_mb - peak_reserved_torch_mb
    peak_memory_mb = peak_allocated_torch_mb + peak_external_mb

    diagnostics["Mem. Usage (GB)"] = f'{peak_memory_mb / 1000:.2f}'

    return diagnostics

def average_timings(lst):
        return round(sum(lst) / len(lst), 2)


def main():
    args = parse_arguments()

    run_name = random_noun_adjective()
    
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
    model_path = args.model

    if not torch.cuda.is_available():
        raise ValueError("A cuda device is necessary to benchmark GPTQ.")
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        raise ValueError(
            "Please set CUDA_VISIBLE_DEVICES variable to a single device index. This benchmark code is not tested for multi-device setup."
        )

    memory_tracker = MemoryTracker()
    
    #tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=False, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True, trust_remote_code=True)

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    use_exllama = False
    if args.kernel == 'exllamav2':
        exllama_version = 2
        use_exllama = True
    elif args.kernel == 'exllama':
        exllama_version = 1
        use_exllama = True

    model, load_time = load_model(args.model,
                                  args.gptq,
                                  args.bitsandbytes,
                                  args.revision,
                                  use_exllama,
                                  exllama_version)

    uses_gptq = args.gptq
    uses_bitsandbytes = args.bitsandbytes

    model = model.eval()

    if uses_gptq:
        quantization = 'gptq'
    elif uses_bitsandbytes:
        quantization = 'bitsandbytes'
    else:
        quantization = 'noquant'

    # if args.ppl:
    #     output_file = open(file_name + "_perplexity.csv", "w")
    #     header = "quantization, act_order, bits, group_size, kernel, perplexity\n"
    #     output_file.write(header)
    #     ppl = Perplexity(model, tokenizer)
    #     ppl_value = np.mean(ppl.calculate_perplexity())
    #     line = "{},{},{},{},{},{}\n".format(
    #         quantization,
    #         act_order,
    #         bits,
    #         group_size,
    #         kernel,
    #         f"{ppl_value:.2f}",
    #     )
    #     print(header)
    #     print(line)
    #     output_file.write(line)
    #     output_file.close()


    param_combinations = {
        "Num. Requests": args.n_request,
        "Prompt Length": args.prompt_length,
        "New Tokens": args.new_tokens,
        "Kernel": [args.kernel]
    }

    if check_experiment(bulb, param_combinations, '-'.join([model_path, args.revision])):
        return None

    param_combinations = list(itertools.product(
        args.n_request,
        args.prompt_length,
        args.new_tokens,
        [args.kernel]
    ))

    for params in tqdm(param_combinations, total=len(param_combinations), desc='Running experiments'):
        n_request, prompt_length, new_tokens, kernel = params
        print(f"---- Running: n_request={n_request}, prompt_length={prompt_length}, new_tokens={new_tokens}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            diagnostics = benchmark_gptq(
                model,
                n_request,
                prompt_length,
                args.replica,
                new_tokens,
                tokenizer.pad_token_id,
                memory_tracker=memory_tracker,
            )

        experiment = {
            # experiment configuration
            "memo": args.memo,
            "Run Name": run_name,
            # software configuration
            "auto_gptq_v": auto_gptq.__version__,
            "Kernel": kernel,
            # harware configuration
            "Device": torch.cuda.get_device_name(),
            "VRAM (GB)": round(torch.cuda.get_device_properties('cuda').total_memory / 1024 / 1024 / 1024, 2),
            "RAM (GB)": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
            "CPU Count": len(psutil.Process().cpu_affinity()),
            # model configuration
            "Quant. Method": quantization,
            "Model": os.path.basename(model_path)+'-'+args.revision,
            #"Model Size (GB)": round(os.path.getsize(model_path) / 1024 / 1024 / 1024, 2),
            "Num. Requests": n_request,
            "Prompt Length": prompt_length,
            "New Tokens": new_tokens,
            # experiment diagnostics
            **diagnostics
            }

        if not args.debug:
            bulb = pd.concat([bulb, pd.DataFrame([experiment])], ignore_index=True)

    if not args.debug:
        bulb.to_csv('bulb.csv', index=False)
     
if __name__ == '__main__':
     main()
    