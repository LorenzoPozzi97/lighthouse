# Lighthouse
<img src="https://github.com/LorenzoPozzi97/lighthouse/assets/83987444/283cb75c-c033-4b9d-8dcd-668b4054ad18" width="400" height="400">

## Benchmarking inference of quantized LLM models
Quantization is the current way-to-go to run local models. However, performance of a model can remarkably vary depending many factors: quantization method, but also hardware, model architecture, kernel, etc. The lighthouse intention is sheding light on LLMs inference performance, providing out-of-the-box funcitonalities to benchmark quantized models and quickly understand their potention across different configurations.

## Setup
To use the fuctionalities of the lighthouse on your machine just clone the repo with:
```
git clone https://github.com/LorenzoPozzi97/lighthouse.git
```
Call the script setup.sh that will create a virtual environemnt with all the necessary packages and activate the venv:
```
bash setup.sh
source .lighthouse/bin/activate
```
### Supported platforms
- [x] Linux
- [ ] Windows
- [ ] Mac OS

## Usage
The lighthouse works in three steps:
1) Decide the parameters of your experiments
2) Store the results in your personal database, namely the bulb 💡
3) Interrogate the bulb 💡 to create interactive graphs

Here is a typical interaction with the library to test a modle quantized in GGUF format:
```
python ./lighthouse/benchmark_gguf.py --model-path solar-10.7b-instruct-v1.0.Q4_K_M.gguf
```
Each configuration in the experiment will be appended to you 💡. 
To create an interactive parallel coordinated graphs use:
```
python ./lighthouse/parallel_coordinates.py
```
or for a 2D graph (with the name autogenerated for the test):
```
python ./lighthouse/bidimensional_graphs.py --run-anchor straightforward_turkey_trot
```

## What you can do
Experments can track a number of different metrics.

### Experiment configuration
- ```memo``` : A brief comment on the run experiment.
- ```Run Name``` : A unique autogenerated name given to the experiment.

### Software configuration
- auto_gptq_v
- llama_cpp_v
- kernel --> GPTQ

### Model configuration
- Quant. Method
- Model
- Model Size
- Batch
- Threads
- Batch Threads
- Context Window
- Prompt Length
- New Tokens
- GPU Layers --> GGUF

### Hardware configuration
- Device
- VRAM
- RAM
- CPU Count

### Experiment diagnostics
- Mem. Usage (TODO) [GB]
- Time To First Token (TTFT) [s] (= Prompt Eval Time)
- TTFT [tk/s] (=TTFT/prompt tokens = prompt Eval Time [tk/s])
- Time Per Output Token [s/tk] (TPOT)
- Eval Time [s] (=TPOT^-1)
- Latency [s] (= TTFT + TPOT * new tokens)
- Latency [tk/s] = L / (new tokens + prompt tokems)
- Load time [s]

### Supported quantization formats
- [x] GGUF
- [ ] GPTQ
- [ ] EETQ
- [ ] bitsandbytes
- [ ] AWQ



## Contributions
Any contribution is very well appreciated! This project is still in its embryonal stage but it could save a lot of time to many people. Here's how you can contribute:

### Step 1: Set Up Your Environment
1) **Fork the Repository**: Start by forking the repository to your GitHub account. This creates your own copy of the project where you can make changes.
2) **Clone Your Fork**: Clone your fork to your local machine using Git. Replace YOUR-USERNAME with your GitHub username.
```
git clone https://github.com/YOUR-USERNAME/project-name.git
```
3) **Create a Branch**: Navigate into the cloned repository and create a branch for your contribution.
```
git checkout -b feature/your-feature-name
```
### Step 2: Make Your Changes
1) **Make Your Changes Locally**: Implement your changes or fixes in your branch.

### Step 3: Submit Your Contribution
1) **Commit Your Change**s: Once you're happy with your changes, commit them to your branch. Make sure your commit messages are clear and descriptive.
```
git commit -am "Add a concise commit message describing your change"
```
2) **Push to Your Fork**: Push your changes to your GitHub fork.
```
git push origin feature/your-feature-name
```
3) **Open a Pull Request (PR)**: Go to the original repository on GitHub, and you'll see a prompt to open a pull request from your fork. Fill in the PR template with details about your changes.

### Step 4: Review Process
1) **Review**: Once your PR is submitted, the project maintainers will review your changes.
2) **Merge**: If the review is passed, the maintainers will merge your PR. Thank you for your contribution!

