import os
import torch
import psutil
import random
import itertools
import auto_gptq
import llama_cpp

import pandas as pd

from pathlib import Path
from pandas import DataFrame
from nltk.corpus import wordnet

def get_wordnet_word(pos):
    """ Get a list of words for a specific part of speech from WordNet. """
    synset = random.choice(list(wordnet.all_synsets(pos)))
    return random.choice(synset.lemmas()).name()

def random_noun_adjective():
    noun = get_wordnet_word(wordnet.NOUN)
    adjective = get_wordnet_word(wordnet.ADJ)
    return f"{adjective}_{noun}".replace('-', '_').replace('\'', '_')

def initialize_bulb(columns):
    """
    Initialize an empty DataFrame with specified columns.

    Parameters:
    - columns (list): A list of strings representing the column names.

    Returns:
    - pd.DataFrame: An empty DataFrame with the specified columns.
    """
    return pd.DataFrame(columns=columns)

def check_experiment(bulb: DataFrame,
                     param_combinations: dict,
                     model_path: Path) -> bool:
    run_names = []
    for params_i in itertools.product(*param_combinations.values()):

        params_d = {k: v for k, v in zip(param_combinations.keys(), params_i)} 

        # General values to check
        values_to_check = {
            "Device": torch.cuda.get_device_name(),
            "CPU Count": len(psutil.Process().cpu_affinity()),
            "Model": os.path.basename(model_path),
            "Prompt Length": params_d["Prompt Length"],
            "New Tokens": params_d["New Tokens"]
            }
        if 'gguf' in model_path.lower():
            values_to_check.update(
                {
                    "llama_cpp_v": llama_cpp.__version__,
                    "Context Length": params_d["Context Length"],
                    "Decode Threads": params_d["Decode Threads"],
                    "Prefill Threads": params_d["Prefill Threads"],
                    "GPU Layers": params_d["GPU Layers"],
                }
            )
        elif 'gptq' in model_path.lower():
            values_to_check.update(
                {
                    "auto_gptq_v": auto_gptq.__version__,
                    "Num. Requests": params_d["Num. Requests"],
                    "Kernel": params_d["Kernel"],
                }
            )

        mask = (bulb[list(values_to_check)] == pd.Series(values_to_check)).all(axis=1)
        if not mask.any():
            return False

        run_names.extend(bulb[mask]["Run Name"].tolist())

    runs_to_check = '\n- '+'\n- '.join(set(run_names))
    print(f"Warning: This experiment has already been run. Check:{runs_to_check}\n\nUse the parameter --force to ignore this message.")
    return True