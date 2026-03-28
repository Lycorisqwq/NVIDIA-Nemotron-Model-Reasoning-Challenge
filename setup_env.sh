#!/bin/bash
# ============================================================
# Setup conda environment for NVIDIA Nemotron competition
# Usage: bash setup_env.sh
# ============================================================
set -e

ENV_NAME="kaggle"
PYTHON_VERSION="3.10"

echo "============================================================"
echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo "============================================================"

# Create conda env
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo ""
echo "============================================================"
echo "Installing PyTorch (CUDA 12.6, compatible with CUDA 12.8 driver)"
echo "============================================================"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "============================================================"
echo "Installing training dependencies"
echo "============================================================"
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "Verifying installation"
echo "============================================================"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import transformers, peft, trl, datasets, polars
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'trl: {trl.__version__}')
print(f'datasets: {datasets.__version__}')
print(f'polars: {polars.__version__}')
print()
print('All dependencies OK!')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "Usage:"
echo "  conda activate ${ENV_NAME}"
echo "  python train.py"
echo "============================================================"
