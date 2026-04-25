#!/usr/bin/env bash
# One-shot setup for a fresh Vultr A40 (or A100/L40S) instance running
# Ubuntu 22.04 with NVIDIA drivers + CUDA already installed.
#
# Usage on the box (run as the default `ubuntu` or `root` user):
#     curl -sSL https://raw.githubusercontent.com/mohosy/affordance-vlm/main/infra/vultr/setup.sh | bash
# or:
#     git clone https://github.com/mohosy/affordance-vlm.git
#     cd affordance-vlm
#     bash infra/vultr/setup.sh

set -euo pipefail

echo "[setup] checking GPU"
nvidia-smi || { echo "no nvidia-smi — wrong instance type?"; exit 1; }

echo "[setup] system deps"
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3-pip git tmux htop ffmpeg

REPO_DIR="${HOME}/affordance-vlm"
if [ ! -d "$REPO_DIR" ]; then
    echo "[setup] cloning repo"
    git clone https://github.com/mohosy/affordance-vlm.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "[setup] python venv"
python3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo "[setup] python deps (this takes a few minutes — torch + cuda)"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install qwen-vl-utils sentencepiece

echo "[setup] sanity check the GPU + bf16"
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("vram (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

cat <<'EOF'

[setup] done.

Next:
  1. cp .env.example .env && nano .env       # fill in API keys
  2. # optional: pull pre-extracted frames + manifest from your Mac:
     #   rsync -avzP -e ssh user@your-mac:/path/to/affordance-vlm/data/frames/ data/frames/
     #   rsync -avzP -e ssh user@your-mac:/path/to/affordance-vlm/data/sequences/ data/sequences/
  3. python data_pipeline/generate_qa_temporal.py \
        --sequences data/sequences/sequences.jsonl \
        --out data/qa/train_temporal.jsonl \
        --pairs-per-sequence 3 --shuffle
  4. tmux new -s train
     source .venv/bin/activate
     python training/finetune_qwen.py --config training/configs/lora.yaml
     # detach: Ctrl-B then D
EOF
