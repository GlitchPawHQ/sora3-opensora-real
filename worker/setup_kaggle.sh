#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENSORA_REPO="${OPENSORA_REPO:-/kaggle/working/Open-Sora}"
OPENSORA_REVISION="${OPENSORA_REVISION:-7ad6a96a135feb81f755c84fb391818718f6beb2}"
DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-true}"

echo "== GPU inspection =="
nvidia-smi || {
  echo "ERROR: nvidia-smi is unavailable. Enable a GPU accelerator in the notebook/runtime." >&2
  exit 2
}
python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}; {p.total_memory / 1024**3:.2f} GiB")
    else:
        raise SystemExit("ERROR: PyTorch cannot access CUDA.")
except ImportError:
    print("PyTorch is not importable in this Python environment.", file=sys.stderr)
    raise
PY

if [[ ! -d "$OPENSORA_REPO/.git" ]]; then
  echo "== Clone official Open-Sora =="
  git clone https://github.com/hpcaitech/Open-Sora "$OPENSORA_REPO"
fi

git -C "$OPENSORA_REPO" fetch --all --tags
git -C "$OPENSORA_REPO" checkout "$OPENSORA_REVISION"

echo "== Install Open-Sora package =="
python -m pip install --upgrade pip
python -m pip install -v "$OPENSORA_REPO"

echo "== Verify inference entry point =="
test -f "$OPENSORA_REPO/scripts/diffusion/inference.py" || {
  echo "ERROR: official inference script was not found at the selected revision." >&2
  exit 3
}
test -f "$OPENSORA_REPO/configs/diffusion/inference/t2i2v_256px.py" || {
  echo "ERROR: expected 256px T2I2V config was not found." >&2
  exit 3
}

if [[ "$DOWNLOAD_CHECKPOINTS" == "true" ]]; then
  echo "== Download Open-Sora v2 checkpoints =="
  python -m pip install "huggingface_hub[cli]"
  huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir "$OPENSORA_REPO/ckpts"
else
  echo "Skipping model checkpoint download because DOWNLOAD_CHECKPOINTS=$DOWNLOAD_CHECKPOINTS"
fi

echo
echo "Setup completed. This does not prove that the model fits the available GPU memory."
echo "Try a short prompt with: bash '$PROJECT_ROOT/worker/generate.sh' 'A red fox runs through grass at dawn.'"
