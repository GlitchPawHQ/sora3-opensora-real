# Standalone Open-Sora GPU Worker (no API)

This folder runs Open-Sora directly from the command line. It does **not** start FastAPI, expose a port, or create an HTTP service.

## Requirements

- Linux with a supported NVIDIA GPU and working CUDA driver.
- Python 3.10, as recommended by the official Open-Sora setup.
- Enough GPU memory for the selected Open-Sora config and checkpoints.
- Disk space for the repository, model weights, and generated videos.
- A Hugging Face account/token only if a selected checkpoint requires gated access. Never commit tokens.

## Kaggle setup

In a Kaggle notebook with GPU enabled, clone this repository and run:

```bash
bash worker/setup_kaggle.sh
```

The setup script clones the official Open-Sora repository at the known project-tested revision and installs its package. It deliberately does not force-install a random PyTorch/xFormers CUDA build over Kaggle's preinstalled stack. If dependency installation reports a conflict, stop and resolve that environment mismatch rather than ignoring it.

## Generate a real video

After setup succeeds:

```bash
cd /kaggle/working/sora3-opensora-real
bash worker/generate.sh "A red fox runs naturally through wet grass at dawn, stable anatomy, continuous motion, realistic contact with the ground, smooth tracking camera."
```

Optional environment overrides:

```bash
export OPENSORA_REPO=/kaggle/working/Open-Sora
export OPENSORA_CONFIG=configs/diffusion/inference/t2i2v_256px.py
export OUTPUT_DIR=/kaggle/working/sora3-opensora-real/outputs
export NUM_FRAMES=17
export ASPECT_RATIO=16:9
export OFFLOAD=true
bash worker/generate.sh "Your video prompt"
```

The official Open-Sora README describes frame counts as values of the form 4k+1 below 129. Start with a short clip to test compatibility. The default config is the official 256px text-to-image-to-video pipeline; change it only to a config that matches the downloaded checkpoints and the GPU topology.

## Checkpoints

The official model card documents downloading Open-Sora v2 with:

```bash
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir /kaggle/working/Open-Sora/ckpts
```

The setup script can attempt this download. Model weights are large and Kaggle sessions/storage are limited. The download may take a long time or run out of disk space. Do not commit model weights to this Git repository.

## Hardware honesty

Open-Sora's published 256px inference benchmark reports peak GPU memory around 52.5 GB on one GPU; the 768px benchmark reports around 60.3 GB on one GPU. Two T4 GPUs with around 15 GB each are not equivalent to one 30 GB GPU and do not automatically provide model-parallel execution. The 11B model may fail on T4s even with offloading. This repository does not claim otherwise.

If inference fails with CUDA out-of-memory or incompatible kernels, the worker prints the actual error and exits. It does not create a placeholder video.

## Output

Real outputs are saved under `OUTPUT_DIR` (default: `outputs` in the project). The script prints the output directory and exits non-zero if inference fails or no video file is produced.

## Official references

- [Open-Sora repository and inference instructions](https://github.com/hpcaitech/Open-Sora)
- [Open-Sora v2 model card](https://huggingface.co/hpcai-tech/Open-Sora-v2)
