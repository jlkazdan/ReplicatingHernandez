# Replicating Hernandez

## Setup

0. Install `uv`: `conda install conda-forge::uv -y`
1. Create the virtual environment: `uv venv -p 3.12 scaling_mem_env`
2. Activate the virtual environment: `source scaling_mem_env/bin/activate`
3. Install the requirements: `uv pip install -r requirements.txt`
4. Install flash_attention: `uv pip install flash-attn --no-build-isolation`

(If you're on `ampere*`, create a tmp dir `export UV_CACHE_DIR=.uv-cache`)
