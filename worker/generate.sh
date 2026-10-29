#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENSORA_REPO="${OPENSORA_REPO:-/kaggle/working/Open-Sora}"
OPENSORA_CONFIG="${OPENSORA_CONFIG:-configs/diffusion/inference/t2i2v_256px.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs}"
NUM_FRAMES="${NUM_FRAMES:-17}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
OFFLOAD="${OFFLOAD:-true}"
NPROC="${NPROC:-1}"
SEED="${SEED:-42}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 'Describe the video to generate'" >&2
  exit 64
fi
PROMPT="$*"

if ! command -v torchrun >/dev/null 2>&1; then
  echo "ERROR: torchrun is missing. Run worker/setup_kaggle.sh in the same Python environment first." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA tools are unavailable; this worker requires a CUDA GPU." >&2
  exit 2
fi
nvidia-smi
INFERENCE="$OPENSORA_REPO/scripts/diffusion/inference.py"
CONFIG="$OPENSORA_REPO/$OPENSORA_CONFIG"
if [[ ! -f "$INFERENCE" ]]; then
  echo "ERROR: Open-Sora inference script not found: $INFERENCE" >&2
  exit 3
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Open-Sora config not found: $CONFIG" >&2
  exit 3
fi
if [[ ! -d "$OPENSORA_REPO/ckpts" ]]; then
  echo "ERROR: checkpoint directory missing: $OPENSORA_REPO/ckpts" >&2
  echo "Run setup with DOWNLOAD_CHECKPOINTS=true or download compatible official checkpoints manually." >&2
  exit 3
fi
if [[ "$NUM_FRAMES" =~ ^[0-9]+$ ]] && (( NUM_FRAMES < 1 || NUM_FRAMES >= 129 || (NUM_FRAMES - 1) % 4 != 0 )); then
  echo "ERROR: NUM_FRAMES must be 4k+1 and less than 129 (for example 5, 9, 17)." >&2
  exit 64
fi
mkdir -p "$OUTPUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_DIR="$OUTPUT_DIR/opensora-$STAMP"
mkdir -p "$JOB_DIR"
LOG="$JOB_DIR/inference.log"

ARGS=(
  --save-dir "$JOB_DIR"
  --prompt "$PROMPT"
  --num_frames "$NUM_FRAMES"
  --aspect_ratio "$ASPECT_RATIO"
  --sampling_option.seed "$SEED"
  --seed "$SEED"
)
if [[ "$OFFLOAD" == "true" ]]; then
  ARGS+=(--offload True)
fi

echo "Launching official Open-Sora inference (no API server)."
echo "Config: $OPENSORA_CONFIG"
echo "GPUs/processes: $NPROC"
echo "Frames: $NUM_FRAMES; aspect ratio: $ASPECT_RATIO"
echo "Output: $JOB_DIR"
set +e
torchrun --nproc_per_node "$NPROC" --standalone "$INFERENCE" "$CONFIG" "${ARGS[@]}" 2>&1 | tee "$LOG"
RESULT="${PIPESTATUS[0]}"
set -e
if (( RESULT != 0 )); then
  echo "ERROR: Open-Sora inference failed with exit code $RESULT. See $LOG" >&2
  exit "$RESULT"
fi

python - "$JOB_DIR" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
videos = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mov"}]
if not videos:
    print(f"ERROR: inference exited successfully but no video file was found in {root}", file=sys.stderr)
    raise SystemExit(4)
for video in sorted(videos, key=lambda p: p.stat().st_mtime):
    print("REAL VIDEO:", video, f"({video.stat().st_size:,} bytes)")
PY
echo "Done. Inspect the real output under: $JOB_DIR"
