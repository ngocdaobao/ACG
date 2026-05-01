# Create conda environment
# conda create -n acg python=3.10 -y
# conda activate acg

uv venv --python=3.10

source .venv/bin/activate

# Install PyTorch (adjust CUDA version if needed)
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
uv pip install -e libs/Isaac-GR00T-N1
# uv pip install --no-index --no-deps flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl
uv pip install --no-build-isolation flash-attn==2.8.3

# Robosuite and RoboCasa
uv pip install libs/robosuite/
uv pip install -e libs/robocasa/

# Download RoboCasa assets (~5GB)
python libs/robocasa/robocasa/scripts/download_kitchen_assets.py

# Robomimic and DexMimicGen
uv pip install -e libs/robomimic/ --no-deps
uv pip install -e libs/dexmimicgen/ --no-deps

# Remaining requirements
uv pip install -r requirements.txt