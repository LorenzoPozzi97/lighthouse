import os
import sys
import json
import torch
import psutil
import argparse
import datetime
import itertools

#from tqdm import tqdm
from llama_cpp import Llama, llama_get_timings

sys.dont_write_bytecode = True

DOCUMENTS = '\n'.join([
    "Gli scambi culturali arricchiscono la nostra comprensione del mondo; per esempio, il programma Erasmus ha permesso a oltre 3 milioni di studenti europei di studiare all'estero, esplorando nuove culture e ampliando le loro prospettive.",
    "Attraverso l'esperienza degli scambi culturali, impariamo a sfidare i nostri pregiudizi e stereotipi; studi hanno dimostrato che gli studenti che studiano all'estero sviluppano una maggiore tolleranza e una mente più aperta.",
    "La condivisione delle arti, della musica e della letteratura nelle esperienze di scambio culturale crea ponti di comunicazione; festival come il Carnevale di Rio coinvolgono persone da tutto il mondo, celebrando la diversità culturale attraverso la musica e la danza.",
    "Gli scambi culturali promuovono la collaborazione e l'innovazione; programmi come Fulbright hanno finanziato oltre 380.000 studiosi e professionisti, favorirendo scambi accademici e professionali tra 160 paesi.",
    "Partecipare a scambi culturali aiuta i giovani a sviluppare competenze globali; secondo l'UNESCO, gli studenti che partecipano a programmi di scambio sono più propensi a sviluppare capacità di problem solving in contesti multiculturali.",
    "L'interazione diretta con diverse culture attraverso gli scambi culturali può ispirare creatività e nuove idee; molti innovatori e artisti famosi, come Steve Jobs, hanno citato i viaggi e l'esposizione a diverse culture come fonte di ispirazione.",
    "Gli scambi culturali sono vitali per costruire una comunità globale più pacifica; secondo il Dipartimento di Stato degli Stati Uniti, i programmi di scambio culturale hanno contribuito a migliorare le relazioni diplomatiche e a ridurre i conflitti internazionali.",
    ])

KEYWORDS = ', '.join(["scambi culturali", "comprensione", "erasmus", "tolleranza", "innovazione", "UNESCO", "creatività", "relazioni diplomatiche", "diversità culturale", "collaborazione internazionale"])
MAX_MODEL_LAYERS = 44
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='LLMInferenceTest',
        description='The script test perforance of LLM models in inference.',
        epilog='Text at the bottom of help'
    )
    parser.add_argument('-m', '--model_path', action='store', default=os.path.join(os.path.expanduser("~"), 'models', 'llama-2-13b-chat.Q5_K_M.gguf'), type=str, help="model path")
    parser.add_argument('-t', '--n_threads', action='store',  nargs='+', type=int, default=[10], help="Number of threads to use for generation")
    parser.add_argument('--n_threads_batch', action='store',  nargs='+', type=int, default=[10], help="Number of threads to use for batch processing")
    parser.add_argument('-b', '--n_batch', action='store',  nargs='+', type=int, default=[256, 512], help="Prompt processing maximum batch size")
    parser.add_argument('--ngl', action='store', nargs='+', type=float, default=[0, 0.25, 0.5, 0.75, 1], help="Percentage of layers to store in VRAM")
    return parser.parse_args()

def get_llm_config(llm, ngl):
    """Returns the test configuration."""
    if torch.cuda.is_available() and ngl>0:
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
        "Batch Threads": llm.n_threads,
        "GPU Layers": ngl
    }

def get_timings(llm):
    timings = llama_get_timings(llm._ctx.ctx)
    return {
        "Load Time (s)": timings.t_load_ms / 1000,
        "Sample Time (s)": timings.t_sample_ms / 1000,
        "Prompt Eval Time (s)": timings.t_p_eval_ms / 1000,
        "Eval Time (s)": timings.t_eval_ms / 1000,
        "Total Time (s)": (timings.t_end_ms - timings.t_start_ms) / 1000,
        "Sample Time (Tk/s)": int((timings.n_sample * 1000) / timings.t_sample_ms),
        "Prompt Eval Time (Tk/s)": int((timings.n_p_eval * 1000) / timings.t_p_eval_ms),
        "Eval Time (Tk/s)": int((timings.n_eval * 1000) / timings.t_eval_ms)
    }

def get_inference_summary(llm_output, llm):
    """Generates output for a single document."""
    return {
        "id": llm_output["id"],
        **llm_output["usage"],
        **get_timings(llm)
    }
def run_stress_test(prompt, model_path, n_threads, n_threads_batch, n_batch, ngl):
    llm = Llama(model_path=model_path, n_gpu_layers=ngl, n_batch=n_batch, n_threads=n_threads, n_threads_batch=n_threads_batch, n_ctx=1100, verbose=False)

    # Initialize a StringIO object to capture stderr
    output = llm(
        prompt, 
        max_tokens=500, 
        stop=["\n"], 
        temperature=0.3,
        seed=101
        )

    log = {
        **get_llm_config(llm, ngl), 
        **get_inference_summary(output, llm)
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'input/{timestamp}.json', 'w') as fp:
        json.dump(log, fp)

def main():
    """Main execution function."""
    args = parse_arguments()

    prompt = f"""Ho un dataset contenente i seguenti documenti: \n{DOCUMENTS}\n\nI documenti sono descritti dalle seguenti parole chiave: {KEYWORDS}\n\nIn base a queste informazioni, fai un riassunto di tutto il dataset.\nIl dataset descrive"""
    if not torch.cuda.is_available():
        args.ngl=[0]

    for params in itertools.product(args.n_threads, args.n_threads_batch, args.n_batch, args.ngl):
        n_threads, n_threads_batch, n_batch, ngl = params
        ngl = int(MAX_MODEL_LAYERS*ngl) # convert to the number of layers to offload
        try:
            run_stress_test(prompt, args.model_path, n_threads, n_threads_batch, n_batch, ngl)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    print('Thinking...')
    main()
    print('Done')