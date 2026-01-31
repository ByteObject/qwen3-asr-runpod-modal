#!/bin/bash
set -e

echo "Qwen3-ASR API Setup"
echo "==================="

# Clean disk space before model download
echo "Cleaning disk space..."
rm -rf /workspace/.cache/huggingface/hub/models--* 2>/dev/null || true
rm -rf /root/.cache/huggingface/* 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
pip cache purge 2>/dev/null || true
echo "Disk space: $(df -h /workspace | tail -1 | awk '{print $4}') available"

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install --upgrade pip

# Core dependencies
python3 -m pip install runpod
python3 -m pip install hf-transfer
python3 -m pip install soundfile librosa pydub
python3 -m pip install "transformers>=4.40.0" accelerate sentencepiece

# Install flash-attn (optional)
python3 -m pip install flash-attn --no-build-isolation || echo "flash-attn skipped"

# Install qwen-asr from GitHub
python3 -m pip install git+https://github.com/QwenLM/Qwen3-ASR.git

# API dependencies
python3 -m pip install fastapi uvicorn

# Clean cache after install
python3 -m pip cache purge
echo "Dependencies installed!"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"

# Create minimal API server
echo "Creating API server..."
cat > /workspace/qwen_asr_api.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
import base64
import tempfile
import os
import asyncio
import uuid
from typing import Optional

app = FastAPI(title="Qwen3-ASR API")
model = None
jobs = {}  # In-memory job store

@app.on_event("startup")
async def startup():
    global model
    print("Loading Qwen3-ASR model...")

    from qwen_asr import Qwen3ASR

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
    model = Qwen3ASR(model_name)

    print(f"Model loaded: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

class TranscribeRequest(BaseModel):
    audio: str
    language: Optional[str] = None
    return_timestamps: bool = False

class TranscribeResponse(BaseModel):
    text: str
    language: str
    timestamps: Optional[list] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    detail: Optional[str] = None

def decode_audio(audio_input: str) -> str:
    """Decode audio from base64 or URL to a temporary file path."""
    import urllib.request

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    try:
        if audio_input.startswith("data:"):
            header, encoded = audio_input.split(",", 1)
            audio_data = base64.b64decode(encoded)
            with open(temp_path, "wb") as f:
                f.write(audio_data)
        elif audio_input.startswith(("http://", "https://")):
            req = urllib.request.Request(audio_input, headers={"User-Agent": "Qwen3-ASR/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(temp_path, "wb") as f:
                    f.write(response.read())
        else:
            audio_data = base64.b64decode(audio_input)
            with open(temp_path, "wb") as f:
                f.write(audio_data)
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise ValueError(f"Failed to decode audio: {e}")

def transcribe_audio(request: TranscribeRequest):
    """Synchronous transcription"""
    temp_path = decode_audio(request.audio)
    try:
        transcribe_kwargs = {}
        if request.language:
            transcribe_kwargs["language"] = request.language

        result = model.transcribe(temp_path, **transcribe_kwargs)

        output = {
            "text": result.get("text", ""),
            "language": result.get("language", request.language or "auto"),
        }

        if request.return_timestamps:
            output["timestamps"] = result.get("segments", result.get("timestamps", []))

        return output
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

async def run_transcription_job(job_id: str, request: TranscribeRequest):
    """Async job runner"""
    jobs[job_id] = {"status": "running"}
    try:
        print(f"Job {job_id}: Transcribing...")
        result = transcribe_audio(request)
        jobs[job_id] = {"status": "done", **result}
        print(f"Job {job_id}: Complete!")
    except Exception as e:
        jobs[job_id] = {"status": "error", "detail": str(e)}
        print(f"Job {job_id}: Failed - {e}")

@app.get("/")
async def root():
    return {"message": "Qwen3-ASR API", "docs": "/docs"}

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """Synchronous transcription"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    print(f"Transcribing audio...")
    result = transcribe_audio(request)
    print("Transcription complete!")
    return TranscribeResponse(**result)

@app.post("/transcribe_async", response_model=JobStatus)
async def transcribe_async(request: TranscribeRequest):
    """Async transcription (recommended for long audio)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued"}
    asyncio.create_task(run_transcription_job(job_id, request))
    return JobStatus(job_id=job_id, status="queued")

@app.get("/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    """Check job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = jobs[job_id]
    return JobStatus(job_id=job_id, status=entry["status"], detail=entry.get("detail"))

@app.get("/result/{job_id}", response_model=TranscribeResponse)
async def job_result(job_id: str):
    """Get job result"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = jobs[job_id]
    if entry["status"] != "done":
        raise HTTPException(status_code=202, detail=f"Job status: {entry['status']}")
    return TranscribeResponse(text=entry["text"], language=entry["language"], timestamps=entry.get("timestamps"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

echo "Setup complete!"
echo "Starting API server..."
python3 /workspace/qwen_asr_api.py
