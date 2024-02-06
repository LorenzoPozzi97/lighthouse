# Create a virtual environment
python3.10 -m venv .lighthouse

# Activate the virtual environment
source .lighthouse/bin/activate

# Install packages
pip install pandas==2.1.4
pip install nltk==3.8.1
pip install plotly==5.18.0
pip install psutil==5.9.7
pip install torch==2.1.2
pip install torchvision==0.16.2
pip install torchaudio==2.1.2
pip install -U kaleido==0.2.1

# GGUF pkgs
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.32 --no-cache-dir

# GPTQ pkgs
pip install accelerate==0.26.1
pip install optimum==1.16.1
pip install auto-gptq==0.6.0
pip install ninja==1.11.1.1
MAX_JOBS=4 pip install flash-attn==2.4.2 --no-build-isolation
pip install transformers==4.36.2
pip install einops==0.7.0

echo "Virtual environment and packages installed successfully."