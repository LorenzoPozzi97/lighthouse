from llama_cpp import Llama
from typing import Generator
import time

def chat_like_printer(generator):
    for message in generator:
        for char in message['choices'][0]['text']:
            print(char, end='', flush=True)
            time.sleep(0.03)  # Adjust the delay time as needed
        time.sleep(0.1)  # Add a pause between messages

if __name__ == "__main__":

    # Load model
    llama = Llama(model_path="/home/navya/models/solar-10.7b-instruct-v1.0.Q4_K_M.gguf", n_gpu_layers=15, n_ctx = 2000)

    # Prompt
    prompt = """Non ho potuto mettera la borsa in macchina perchè era troppo piccola. Domanda: Cosa era piccola?\nRisposta: """

    # Generate
    print('Thinking...\n\n')
    output = llama(prompt, stream=True, temperature=0, echo=True, max_tokens=500)

    if isinstance(output, Generator):
        chat_like_printer(output)
    else:
        print(output)
