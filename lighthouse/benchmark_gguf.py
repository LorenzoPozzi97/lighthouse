# TODO
# fix kfold

import os
import sys
import json
import torch
import psutil
import random
import argparse
import datetime
import itertools

import numpy as np
import pandas as pd

#from tqdm import tqdm
from datetime import datetime
from nltk.corpus import wordnet
from llama_cpp import Llama, llama_get_timings

np.random.seed(101)
sys.dont_write_bytecode = True

MODELS_DIR = os.path.join(os.path.expanduser("~"), 'models')
df = pd.read_csv('output/bulb.csv')

def get_wordnet_word(pos):
    """ Get a list of words for a specific part of speech from WordNet. """
    synset = random.choice(list(wordnet.all_synsets(pos)))
    return random.choice(synset.lemmas()).name()

def random_noun_adjective():
    noun = get_wordnet_word(wordnet.NOUN)
    adjective = get_wordnet_word(wordnet.ADJ)
    return f"{adjective}_{noun}".replace('-', '_')

RUN_NAME = random_noun_adjective()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='LLMInferenceTest',
        description='The script test perforance of LLM models in inference.',
        epilog='Text at the bottom of help'
    )
    parser.add_argument('--model_path', action='store', type=str, default=os.path.join(os.path.expanduser("~"), 'models', 'llama-2-13b-chat.Q5_K_M.gguf'), help="Model path.")
    parser.add_argument('--n_threads', action='store',  nargs='+', type=int, default=[10], help="Number of threads to use for generation.")
    parser.add_argument('--n_threads_batch', action='store',  nargs='+', type=int, default=[10], help="Number of threads to use during batch and prompt processing.")
    parser.add_argument('--n_batch', action='store',  nargs='+', type=int, default=[512], help="Prompt processing maximum batch size.")
    parser.add_argument('--ngl', action='store', nargs='+', type=float, default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], help="Percentage of layers to store in VRAM.")
    parser.add_argument('--force', action='store_true', default=False, help="False the test even if the config already exists.")
    parser.add_argument("--prompt_length", nargs='+', type=int, default=[500], help="Length of the prompt.")
    parser.add_argument("--new_tokens", nargs='+', type=int, default=[100], help="Length of the generation.")
    parser.add_argument("--k_folds", type=int, default=1, help="Number of repeated tests.")
    parser.add_argument("--memo", type=str, default='', help="Description of the experiment.")
    return parser.parse_args()

def get_llm_config(llm):
    """Returns the test configuration."""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties('cuda').name
        VRAM = round(torch.cuda.get_device_properties('cuda').total_memory / 1024 / 1024 / 1024, 2)
        torch.cuda.empty_cache()
    else:
        device = 'cpu only'
        VRAM = 0

    return {
        "Device": device,
        "VRAM (GB)": VRAM,
        "RAM (GB)": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
        "CPU Count": len(psutil.Process().cpu_affinity()),
        "Model": os.path.basename(llm.model_path),
        "Model Size (GB)": round(os.path.getsize(llm.model_path) / 1024 / 1024 / 1024, 2),
        "Context Window": llm.n_ctx(),
        "Batch": llm.n_batch,
        "Threads": llm.n_threads,
        "Batch Threads": llm.n_threads_batch
    }

def get_timings(llm):
    timings = llama_get_timings(llm._ctx.ctx)
    return {
        "Load Time (s)": timings.t_load_ms / 1000,
        "Sample Time (s)": timings.t_sample_ms / 1000,
        "Prompt Eval Time (s)": timings.t_p_eval_ms / 1000,
        "Eval Time (s)": timings.t_eval_ms / 1000,
        "Total Time (s)": (timings.t_end_ms - timings.t_start_ms) / 1000,
        "Sample Time (Tk/s)": (timings.n_sample * 1000) / timings.t_sample_ms,
        "Prompt Eval Time (Tk/s)": (timings.n_p_eval * 1000) / timings.t_p_eval_ms,
        "Eval Time (Tk/s)": (timings.n_eval * 1000) / timings.t_eval_ms
    }

def benchmark_gguf(prompt_length, model_path, n_threads, n_threads_batch, n_batch, ngl, new_tokens, k_folds, p_ngl, memo):
    print(os.path.basename(os.path.normpath(model_path)), n_threads, n_threads_batch, n_batch, ngl)

    llm = Llama(model_path=model_path, n_gpu_layers=ngl, n_batch=n_batch, n_threads=n_threads, n_threads_batch=n_threads_batch, n_ctx=1100, verbose=True)
    prompt = np.random.randint(1, llm._model.n_vocab(), size=prompt_length).tolist()

    load_times = []
    sample_times = []
    prompt_eval_times = []
    eval_times = []
    total_times = []
    sample_tks = []
    prompt_eval_tks = []
    eval_tks = []

    for i in range(k_folds):
        output = llm(
            prompt,
            max_tokens=new_tokens,
            logit_bias={llm._token_eos: float('-inf')},
            temperature=0.3,
            seed=101
            )
        timings = get_timings(llm)
        load_times.append(timings["Load Time (s)"])
        sample_times.append(timings["Sample Time (s)"])
        prompt_eval_times.append(timings["Prompt Eval Time (s)"])
        eval_times.append(timings["Eval Time (s)"])
        total_times.append(timings["Total Time (s)"])
        sample_tks.append(timings["Sample Time (Tk/s)"])
        prompt_eval_tks.append(timings["Prompt Eval Time (Tk/s)"])
        eval_tks.append(timings["Eval Time (Tk/s)"])

    run_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    log = {
        "id": output["id"],
        "memo": memo,
        "Run Name": RUN_NAME,
        "run_time": run_time,
        **get_llm_config(llm),
        "GPU Layers": p_ngl,
        "Prompt Length": prompt_length+1,
        "New Tokens": new_tokens,
        "Load Time (s)": sum(load_times) / len(load_times),
        "Sample Time (s)": sum(sample_times) / len(sample_times),
        "Prompt Eval Time (s)": sum(prompt_eval_times) / len(prompt_eval_times),
        "Eval Time (s)": sum(eval_times) / len(eval_times),
        "Total Time (s)": round(sum(total_times) / len(total_times), 2),
        "Sample Time (Tk/s)": round(sum(sample_tks) / len(sample_tks), 2),
        "Prompt Eval Time (Tk/s)": round(sum(prompt_eval_tks) / len(prompt_eval_tks), 2),
        "Eval Time (Tk/s)": round(sum(eval_tks) / len(eval_tks), 2)
    }

    with open(os.path.join('input', f'{run_time}.json'), 'w') as fp:
        json.dump(log, fp)

def main():
    """Main execution function."""
    args = parse_arguments()

    # create the folder that will host the output (if doesn't exists already)
    os.makedirs('input', exist_ok=True)

    # create the node id
    # device = 'cpu only' if not torch.cuda.is_available() else torch.cuda.get_device_properties('cuda').name
    # vram = '0' if not torch.cuda.is_available() else str(round(torch.cuda.get_device_properties('cuda').total_memory / 1024 / 1024 / 1024, 2))
    # ram = str(round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2))
    # n_cpu = str(len(psutil.Process().cpu_affinity()))
    # model_name = os.path.basename(os.path.normpath(args.model_path))
    # node_id = '-'.join([device, vram, ram, n_cpu, model_name])

    # safe condition in case GPU is not avalable
    if not torch.cuda.is_available():
        raise EnvironmentError('cuda not avalable')
    print()
    for params in itertools.product(args.n_threads, args.n_threads_batch, args.n_batch, args.ngl, args.prompt_length, args.new_tokens):
        n_threads, n_threads_batch, n_batch, ngl, prompt_length, new_tokens = params
        model_path = os.path.join(MODELS_DIR, args.model_path)
        int_ngl = int((int(Llama(model_path=model_path, verbose=False).metadata['llama.block_count'])+1)*ngl) # convert to the number of layers to offload

        # check if the configuration has already been tested on the current machine
        config_already_tried = not df[(df['Threads'] == n_threads) & 
                                  (df['Batch Threads'] == n_threads_batch) & 
                                  (df['Batch'] == n_batch) & 
                                  (df['GPU Layers'] == ngl)].empty

        if config_already_tried and not args.force:
            print(f'Configuration already tested on this machine\n\tn_threads: {n_threads}\n\tn_threads_batch: {n_threads_batch}\n\tn_batch: {n_batch}\n\tngl: {ngl} ({ngl})\n')
        else:
            benchmark_gguf(prompt_length-1, model_path, n_threads, n_threads_batch, n_batch, int_ngl, new_tokens, args.k_folds, ngl, args.memo)


if __name__ == "__main__":
    print('Thinking...')
    main()
    print('Done')
