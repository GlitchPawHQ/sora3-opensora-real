import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

APP_NAME = "Sora 3 — Open-Sora Real"
OUTPUT_ROOT = Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
GENERATION_ENABLED = os.getenv("GENERATION_ENABLED", "false").lower() in {"1", "true", "yes"}
OPENSORA_REPO = Path(os.getenv("OPENSORA_REPO", "/workspace/Open-Sora")).resolve()
OPENSORA_CONFIG = os.getenv("OPENSORA_CONFIG", "configs/diffusion/inference/256px.py")
OPENSORA_NPROC = max(1, int(os.getenv("OPENSORA_NPROC", "1")))
JOB_TIMEOUT_SECONDS = max(60, int(os.getenv("JOB_TIMEOUT_SECONDS", "3600")))
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1")))
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
AUDIO_COMMAND = os.getenv("AUDIO_COMMAND", "").strip()

app = FastAPI(title=APP_NAME, version="0.1.0")
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
lock = threading.Lock()
jobs: dict[str, dict[str, Any]] = {}
active_jobs = 0


class JobRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    upscale_4k: bool = False
    realistic_motion: bool = True
    generate_audio: bool = False


def _motion_prompt(prompt: str, enabled: bool) -> str:
    if not enabled:
        return prompt
    return (
        prompt.rstrip()
        + "\n\nMotion and continuity guidance: natural body mechanics and believable weight; "
        "continuous movement with clear cause and effect; stable character identity, proportions, "
        "and clothing across frames; consistent object permanence; realistic contact and shadows; "
        "smooth camera movement with no accidental cuts, warping, duplicated limbs, or flicker."
    )


def _set_job(job_id: str, **updates: Any) -> None:
    with lock:
        jobs[job_id].update(updates)
        jobs[job_id]["updated_at"] = time.time()


def _find_video(folder: Path) -> Path | None:
    videos = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mov"}]
    if not videos:
        return None
    return max(videos, key=lambda p: p.stat().st_mtime)


def _run_audio_pipeline(prompt: str, video_path: Path, job_dir: Path) -> Path | None:
    if not AUDIO_COMMAND:
        return None
    audio_path = job_dir / "generated_audio.wav"
    placeholders = {
        "{prompt}": prompt,
        "{video}": str(video_path),
        "{output_audio}": str(audio_path),
    }
    argv = shlex.split(AUDIO_COMMAND)
    argv = [next((value.replace(key, replacement) for key, replacement in placeholders.items() if key in value), value)
            for value in argv]
    completed = subprocess.run(
        argv, cwd=str(job_dir), capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError("Configured audio pipeline failed: " + completed.stderr[-3000:])
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise RuntimeError("Audio command exited successfully but did not create generated_audio.wav")
    return audio_path


def _mux_audio(video_path: Path, audio_path: Path, job_dir: Path) -> Path:
    output_path = job_dir / "final_with_audio.mp4"
    command = [
        FFMPEG_BIN, "-y", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-shortest", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS, check=False)
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError("FFmpeg audio mux failed: " + completed.stderr[-3000:])
    return output_path


def _upscale_4k(video_path: Path, job_dir: Path) -> Path:
    output_path = job_dir / "final_4k.mp4"
    command = [
        FFMPEG_BIN, "-y", "-i", str(video_path),
        "-vf", "scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "copy", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS, check=False)
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError("4K resize failed: " + completed.stderr[-3000:])
    return output_path


def _generate(job_id: str, request: JobRequest) -> None:
    global active_jobs
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _set_job(job_id, status="running", message="Starting official Open-Sora inference", started_at=time.time())
    try:
        if not GENERATION_ENABLED:
            raise RuntimeError("Real generation is disabled. Configure a CUDA GPU worker and set GENERATION_ENABLED=true.")
        if not shutil.which("torchrun"):
            raise RuntimeError("torchrun was not found. Install the official Open-Sora environment in this worker.")
        inference_script = OPENSORA_REPO / "scripts" / "diffusion" / "inference.py"
        config_path = (OPENSORA_REPO / OPENSORA_CONFIG).resolve()
        if not inference_script.is_file():
            raise RuntimeError(f"Open-Sora inference script not found: {inference_script}")
        if not config_path.is_file():
            raise RuntimeError(f"Open-Sora config not found: {config_path}")
        if not (OPENSORA_REPO / "opensora").is_dir():
            raise RuntimeError(f"OPENSORA_REPO is not an Open-Sora checkout: {OPENSORA_REPO}")

        prompt = _motion_prompt(request.prompt, request.realistic_motion)
        command = [
            "torchrun", "--nproc_per_node", str(OPENSORA_NPROC), "--standalone",
            str(inference_script), str(config_path), "--save-dir", str(job_dir), "--prompt", prompt,
        ]
        if request.seed is not None:
            command.extend(["--seed", str(request.seed), "--sampling_option.seed", str(request.seed)])
        _set_job(job_id, message="Running the official Open-Sora inference script", command=command)
        completed = subprocess.run(
            command, cwd=str(OPENSORA_REPO), capture_output=True, text=True,
            timeout=JOB_TIMEOUT_SECONDS, check=False,
        )
        log_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        (job_dir / "inference.log").write_text(log_text[-200_000:], encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Open-Sora exited with code {completed.returncode}: {log_text[-3500:]}")
        video_path = _find_video(job_dir)
        if video_path is None:
            raise RuntimeError("Open-Sora completed but no real video file was found in the job output directory.")

        audio_status = "not requested"
        if request.generate_audio:
            audio_path = _run_audio_pipeline(prompt, video_path, job_dir)
            if audio_path is None:
                audio_status = "not generated: AUDIO_COMMAND is not configured"
            else:
                video_path = _mux_audio(video_path, audio_path, job_dir)
                audio_status = "generated and muxed by configured external audio pipeline"

        if request.upscale_4k:
            video_path = _upscale_4k(video_path, job_dir)
        _set_job(
            job_id, status="completed", message="Real video generated",
            video_path=str(video_path), audio_status=audio_status,
            output_bytes=video_path.stat().st_size, finished_at=time.time(),
        )
    except subprocess.TimeoutExpired:
        _set_job(job_id, status="failed", message=f"Job exceeded timeout of {JOB_TIMEOUT_SECONDS} seconds.")
    except Exception as exc:
        _set_job(job_id, status="failed", message=str(exc)[:5000])
    finally:
        with lock:
            active_jobs = max(0, active_jobs - 1)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sora 3 — Open-Sora Real</title>
<style>
body{font:16px system-ui;max-width:850px;margin:36px auto;padding:0 18px;background:#10131a;color:#f3f5f7}
main{background:#1a202b;border:1px solid #303948;border-radius:16px;padding:24px}
textarea,input,button{font:inherit;border-radius:8px;padding:10px;border:1px solid #465064;background:#0f141d;color:#fff}
textarea{box-sizing:border-box;width:100%;min-height:140px}button{cursor:pointer;background:#4b66d1;border:0;margin-top:10px}
small{color:#b8c1d0}.status{white-space:pre-wrap;overflow-wrap:anywhere}
</style></head><body><main><h1>Sora 3 — Open-Sora Real</h1>
<p>Real Open-Sora inference worker. No placeholder videos.</p>
<p id="health">Checking worker status…</p>
<label for="prompt">Video prompt</label><textarea id="prompt" placeholder="Describe the scene, action, camera, lighting, and motion…"></textarea>
<p><label><input type="checkbox" id="motion" checked> Motion consistency guidance</label></p>
<p><label><input type="checkbox" id="upscale"> Resize result to 4K (3840×2160; not native detail)</label></p>
<p><label><input type="checkbox" id="audio"> Request external audio generation (requires configured audio pipeline)</label></p>
<button id="submit">Generate real video</button><p id="status" class="status"></p>
<script>
async function health(){try{let r=await fetch('/health'),j=await r.json();document.querySelector('#health').textContent='Generation enabled: '+j.generation_enabled+' · CUDA worker must be configured';}catch(e){document.querySelector('#health').textContent='API status unavailable';}}
health();
document.querySelector('#submit').onclick=async()=>{const prompt=document.querySelector('#prompt').value.trim();if(prompt.length<3){alert('Enter a longer prompt.');return}
const s=document.querySelector('#status');s.textContent='Submitting job…';
try{let r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,realistic_motion:document.querySelector('#motion').checked,upscale_4k:document.querySelector('#upscale').checked,generate_audio:document.querySelector('#audio').checked})});let j=await r.json();if(!r.ok)throw new Error(j.detail||'Request failed');s.textContent='Job '+j.job_id+' submitted.';
while(true){await new Promise(resolve=>setTimeout(resolve,3000));let q=await fetch('/api/jobs/'+j.job_id);let state=await q.json();s.textContent=JSON.stringify(state,null,2);if(state.status==='completed'){s.innerHTML+='\n';let a=document.createElement('a');a.href='/api/jobs/'+j.job_id+'/video';a.textContent='Download generated video';s.appendChild(a);break}if(state.status==='failed')break}
}catch(e){s.textContent='Error: '+e.message}};
</script></main></body></html>"""


@app.get("/health")
def health() -> dict[str, Any]:
    repo_ok = (OPENSORA_REPO / "scripts" / "diffusion" / "inference.py").is_file()
    return {
        "status": "ok",
        "generation_enabled": GENERATION_ENABLED,
        "opensora_repo_found": repo_ok,
        "opensora_config": OPENSORA_CONFIG,
        "opensora_processes": OPENSORA_NPROC,
        "queued_or_running_jobs": active_jobs,
        "audio_pipeline_configured": bool(AUDIO_COMMAND),
        "warning": None if GENERATION_ENABLED and repo_ok else "Generation is not ready; no mock generation is provided.",
    }


@app.post("/api/jobs", status_code=202)
def create_job(request: JobRequest) -> dict[str, Any]:
    global active_jobs
    if not GENERATION_ENABLED:
        raise HTTPException(status_code=503, detail="Real generation is disabled until a compatible GPU worker is configured.")
    with lock:
        if active_jobs >= MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="GPU worker is busy; try again later.")
        active_jobs += 1
        job_id = uuid.uuid4().hex
        jobs[job_id] = {
            "job_id": job_id, "status": "queued", "message": "Queued for real Open-Sora inference",
            "created_at": time.time(), "updated_at": time.time(),
        }
    try:
        executor.submit(_generate, job_id, request)
    except Exception:
        with lock:
            active_jobs = max(0, active_jobs - 1)
            jobs.pop(job_id, None)
        raise
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found in this worker process.")
        public_job = {key: value for key, value in job.items() if key != "command"}
        return public_job


@app.get("/api/jobs/{job_id}/video")
def download_video(job_id: str) -> FileResponse:
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.get("status") != "completed" or not job.get("video_path"):
            raise HTTPException(status_code=409, detail="The real video is not ready yet.")
        video_path = Path(job["video_path"])
    if not video_path.is_file():
        raise HTTPException(status_code=410, detail="The generated video is no longer present on this worker.")
    return FileResponse(video_path, media_type="video/mp4", filename=f"sora3-{job_id}{video_path.suffix}")
