# Sora 3 — Open-Sora Real

An honest, GPU-worker-first starter for real text-to-video generation with the official [Open-Sora](https://github.com/hpcaitech/Open-Sora) inference code. This is a community project, not an official OpenAI product and not the unreleased Sora model.

## What this repository does

- FastAPI job API with job status and real output downloads.
- Calls the official Open-Sora inference script on a configured GPU worker.
- No mock or placeholder video generation. If the model, checkpoint, CUDA, or command is missing, the job fails with an error instead of fabricating an output.
- Optional FFmpeg post-processing to resize an actual generated video to 3840×2160. Upscaling changes pixel dimensions; it cannot guarantee native 4K detail.
- Optional external audio command hook. Open-Sora's video inference does not by itself create a complete synchronized speech/music/SFX soundtrack; audio is only attached when a real audio pipeline is configured.
- Prompt guidance for coherent motion, physical continuity, camera behavior, and stable character identity.

## Architecture

- **API / web UI:** lightweight FastAPI service. Can run on a CPU host such as Render, but generation stays disabled unless explicitly configured.
- **GPU worker:** run this project on a machine with CUDA, the official Open-Sora repository, its dependencies, and compatible checkpoints.
- **Storage:** generated files live under OUTPUT_DIR; use persistent storage or object storage for deployments where local disks are ephemeral.

## Safety defaults

Generation is disabled by default. Set GENERATION_ENABLED=true only on the GPU worker after you have installed the real model and tested the command. Never set it on a CPU-only web host expecting it to generate videos. Do not commit model checkpoints, API keys, or private prompts.

## Quick start — API only

Python 3.11 is recommended for the API.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. The API will report that generation is disabled until configured. This is intentional and is not a demo/mock generation mode.

## Configure a real Open-Sora GPU worker

1. Follow the installation and checkpoint instructions in the official [Open-Sora repository](https://github.com/hpcaitech/Open-Sora). Use the version and checkpoint compatible with your chosen inference config.
2. Set OPENSORA_REPO to the path of the official clone, for example /workspace/Open-Sora.
3. Set OPENSORA_CONFIG to a config that matches your downloaded model/checkpoints. The default is the official repo's configs/diffusion/inference/256px.py; do not assume it fits every GPU.
4. Set OPENSORA_NPROC to the number of GPUs actually available to the worker. Do not set this to a number greater than the available GPUs.
5. Set GENERATION_ENABLED=true, then run uvicorn app:app --host 0.0.0.0 --port 8000 in that GPU environment.
6. Submit a short prompt first and check the job logs and output. The wrapper calls the official script:
   torchrun --nproc_per_node N --standalone scripts/diffusion/inference.py CONFIG --save-dir OUTPUT --prompt PROMPT.

Open-Sora may require multiple GPUs and significant VRAM depending on the config. Two Tesla T4 GPUs are not a guarantee that the 11B/768px configuration will fit or run successfully. Validate the selected official config and checkpoints before offering it publicly.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| GENERATION_ENABLED | false | Explicitly enable real generation on a suitable worker |
| OPENSORA_REPO | /workspace/Open-Sora | Path to official Open-Sora checkout |
| OPENSORA_CONFIG | configs/diffusion/inference/256px.py | Relative config path inside that checkout |
| OPENSORA_NPROC | 1 | torchrun process count; must match available GPU topology |
| OUTPUT_DIR | outputs | Directory for jobs and resulting media |
| JOB_TIMEOUT_SECONDS | 3600 | Maximum inference runtime per job |
| MAX_CONCURRENT_JOBS | 1 | In-process concurrency cap |
| FFMPEG_BIN | ffmpeg | FFmpeg executable for optional post-processing |
| AUDIO_COMMAND | unset | Optional external audio tool command template; must produce a real audio file |

## 4K and audio notes

The Open-Sora config determines the model's native output resolution. The optional resize step can produce a 3840×2160 file only after a real video has been generated and FFmpeg is available. It does not make low-resolution output native 4K. Audio is not fabricated; if no audio tool is configured, the result remains silent and the API reports that audio was not generated.

## API

- GET / — small web UI
- GET /health — readiness and generation configuration
- POST /api/jobs — queue a text-to-video job
- GET /api/jobs/{job_id} — job state and logs
- GET /api/jobs/{job_id}/video — download the actual generated video when complete

Example request:

```json
{"prompt":"A small red fox runs through wet grass at dawn, natural gait, consistent body proportions, stable camera tracking, realistic contact with the ground, soft mist, cinematic lighting.","upscale_4k":false}
```

## License

Project code: MIT. Open-Sora and its checkpoints retain their own licenses and terms; review them before use.