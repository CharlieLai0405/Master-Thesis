#!/bin/bash
# Environment Setup Script for Master Thesis (AnomalyGPT + PromptAD)
# Target: Python 3.12, CUDA 12.8+, Driver 580+, 2x RTX 5090 (sm_120)
# Last verified: 2026-05-15

set -e

echo "=== Creating virtual environments ==="

# ==========================================
# Environment 1: anomaly_gpt
# ==========================================
python3 -m venv /venv/anomaly_gpt
source /venv/anomaly_gpt/bin/activate

echo "[anomaly_gpt] Installing PyTorch 2.11.0+cu128..."
pip install --upgrade pip
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

echo "[anomaly_gpt] Installing core packages..."
pip install transformers==4.38.2
pip install peft==0.9.0
pip install deepspeed==0.9.3
pip install "pydantic<2"  # CRITICAL: deepspeed 0.9.3 breaks with pydantic v2
pip install gradio==3.50.2
pip install open_clip_torch==3.3.0
pip install numpy==1.26.4  # CRITICAL: numpy 2.x breaks deepspeed
pip install scikit-learn scipy pillow tqdm
pip install sentencepiece protobuf
pip install pytorchvideo  # needs torchvision.transforms.functional_tensor shim

echo "[anomaly_gpt] Installing from requirements (full)..."
pip install -r requirements_anomaly_gpt.txt 2>/dev/null || true

# CRITICAL: Patch DeepSpeed for RTX 5090 (sm_120)
# File: /venv/anomaly_gpt/lib/python3.12/site-packages/deepspeed/ops/op_builder/builder.py
# Change: num = int(cc[0]) * 10 + int(cc[1])
# To:     num = int(cc.replace('.','').replace('+PTX',''))
echo "[anomaly_gpt] Patching DeepSpeed for sm_120..."
BUILDER_PY="/venv/anomaly_gpt/lib/python3.12/site-packages/deepspeed/ops/op_builder/builder.py"
if [ -f "$BUILDER_PY" ]; then
    sed -i "s/num = int(cc\[0\]) \* 10 + int(cc\[1\])/num = int(cc.replace('.','').replace('+PTX',''))/" "$BUILDER_PY"
    echo "  DeepSpeed patched."
fi

deactivate

# ==========================================
# Environment 2: promptad
# ==========================================
python3 -m venv /venv/promptad
source /venv/promptad/bin/activate

echo "[promptad] Installing PyTorch 2.11.0+cu128..."
pip install --upgrade pip
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

echo "[promptad] Installing core packages..."
pip install open_clip_torch==3.3.0
pip install scikit-learn scikit-image scipy
pip install numpy  # promptad can use numpy 2.x
pip install pillow tqdm tabulate

echo "[promptad] Installing from requirements (full)..."
pip install -r requirements_promptad.txt 2>/dev/null || true

deactivate

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT NOTES:"
echo "  1. anomaly_gpt: MUST use pydantic<2 and numpy<2"
echo "  2. DeepSpeed builder.py patched for sm_120 (RTX 5090)"
echo "  3. Never install packages from promptad env into anomaly_gpt (numpy conflict!)"
echo "  4. Use separate venvs: source /venv/anomaly_gpt/bin/activate OR /venv/promptad/bin/activate"
echo ""
echo "=== Pretrained Models (download separately) ==="
echo "  Place in: /workspace/Master-Thesis/pretrained_ckpt/"
echo "  1. imagebind_ckpt/imagebind_huge.pth (4.5GB)"
echo "     - https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
echo "  2. vicuna_ckpt/7b_v0/ (13GB)"
echo "     - huggingface: lmsys/vicuna-7b-delta-v0 (apply delta to llama-7b)"
echo "  3. pandagpt_ckpt/7b/pytorch_model.pt (20GB)"
echo "     - https://huggingface.co/openllmplayground/pandagpt_7b_max_len_1024"
