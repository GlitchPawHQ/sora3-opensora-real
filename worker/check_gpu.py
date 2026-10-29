#!/usr/bin/env python3
"""Report GPU visibility and basic compatibility without starting inference."""
import sys

try:
    import torch
except ImportError as exc:
    print(f"ERROR: PyTorch is not installed in this Python environment: {exc}", file=sys.stderr)
    raise SystemExit(2)

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA runtime reported by PyTorch: {torch.version.cuda}")
print(f"Visible GPU count: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA is unavailable. Enable a compatible NVIDIA GPU runtime.", file=sys.stderr)
    raise SystemExit(3)

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {props.name}")
    print(f"  VRAM: {props.total_memory / (1024**3):.2f} GiB")
    print(f"  Compute capability: {props.major}.{props.minor}")

print("Diagnostic complete. This checks visibility only; it does not guarantee Open-Sora will fit in VRAM.")
