import os
import torch
import psutil
import random
import itertools

import pandas as pd

from nltk.corpus import wordnet

def get_wordnet_word(pos):
    """ Get a list of words for a specific part of speech from WordNet. """
    synset = random.choice(list(wordnet.all_synsets(pos)))
    return random.choice(synset.lemmas()).name()

def random_noun_adjective():
    noun = get_wordnet_word(wordnet.NOUN)
    adjective = get_wordnet_word(wordnet.ADJ)
    return f"{adjective}_{noun}".replace('-', '_').replace('\'', '_')

def check_experiment(bulb, args, model_path):
    run_names = []

    for params in itertools.product(args.n_threads, args.n_threads_batch, args.n_batch, args.ngl, args.prompt_length, args.new_tokens):
        n_threads, n_threads_batch, n_batch, ngl, prompt_length, new_tokens = params
        values_to_check = {"Device": torch.cuda.get_device_name(),
                           "CPU Count": len(psutil.Process().cpu_affinity()),
                           "Model": os.path.basename(model_path),
                           "Context Window": args.ctx,
                           "Batch": n_batch,
                           "Threads": n_threads,
                           "Batch Threads": n_threads_batch,
                           "GPU Layers": ngl,
                           "Prompt Length": prompt_length,
                           "New Tokens": new_tokens}

        mask = (bulb[list(values_to_check)] == pd.Series(values_to_check)).all(axis=1)
        if not mask.any():
            return False

        run_names.extend(bulb[mask]["Run Name"].tolist())

    runs_to_check = '\n- '+'\n- '.join(set(run_names))
    print(f"Warning: This experiment has already been runned. Check:{runs_to_check}\n\nUse the parameter --force to ignore this message.")
    return True