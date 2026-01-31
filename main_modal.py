"""
Modal Serverless Handler for Qwen3-ASR

Supports single and batch audio transcription with optional language specification
and timestamp output.
"""

import base64
import ipaddress
import os
import socket
import tempfile
import urllib.request
from typing import Optional, List
from urllib.parse import urlparse

import modal
from pydantic import BaseModel

# Modal App
app = modal.App("qwen3-asr")

# Image definition
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "transformers>=4.40.0",
        "accelerate",
        "sentencepiece",
        "soundfile",
        "librosa",
        "torch",
        "torchaudio",
    )
    .pip_install("git+https://github.com/QwenLM/Qwen3-ASR.git")
    .env({"HF_HOME": "/cache/huggingface"})
)

# Volume for model cache
model_volume = modal.Volume.from_name("qwen-asr-models", create_if_missing=True)

# Constants
MAX_AUDIO_SIZE_MB = 100
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024
MAX_BASE64_SIZE = 150 * 1024 * 1024
DEFAULT_MAX_BATCH_SIZE = 32
URL_TIMEOUT_SECONDS = 30

# Security: Blocked hosts and networks for SSRF protection
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"}
BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
]

ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac",
    "audio/ogg", "audio/webm",
    "audio/mp4", "audio/m4a", "audio/x-m4a",
    "application/octet-stream",
}


# Request/Response Models
class TranscribeRequest(BaseModel):
    audio: str  # base64, URL, or data URI
    language: Optional[str] = None
    return_timestamps: bool = False


class BatchTranscribeRequest(BaseModel):
    batch: List[TranscribeRequest]


class TranscribeResponse(BaseModel):
    text: str
    language: str
    timestamps: Optional[List[dict]] = None


class BatchTranscribeResponse(BaseModel):
    results: List[dict]


# Helper functions
def is_url_safe(url: str) -> bool:
    """Validate URL is not targeting internal resources (SSRF protection)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False

        if hostname.lower() in BLOCKED_HOSTS:
            return False

        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)

        for network in BLOCKED_NETWORKS:
            if ip_obj in network:
                return False

        return True
    except (socket.gaierror, ValueError):
        return False


def get_audio_extension(url: str = None, content_type: str = None) -> str:
    """Determine appropriate file extension from URL or content-type."""
    if url:
        parsed_path = urlparse(url).path.lower()
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a"]:
            if parsed_path.endswith(ext):
                return ext

    if content_type:
        mime_to_ext = {
            "audio/wav": ".wav", "audio/wave": ".wav", "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
            "audio/flac": ".flac", "audio/x-flac": ".flac",
            "audio/ogg": ".ogg", "audio/webm": ".webm",
            "audio/mp4": ".m4a", "audio/m4a": ".m4a", "audio/x-m4a": ".m4a",
        }
        return mime_to_ext.get(content_type, ".wav")

    return ".wav"


def download_with_limit(url: str, dest_path: str, max_size: int = MAX_AUDIO_SIZE_BYTES) -> str:
    """Download URL with size limit and timeout."""
    req = urllib.request.Request(url, headers={"User-Agent": "Qwen3-ASR/1.0"})

    with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_size:
            raise ValueError(f"File too large: {int(content_length)} bytes exceeds limit")

        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()

        downloaded = 0
        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > max_size:
                    raise ValueError(f"Download exceeded {max_size} byte limit")
                f.write(chunk)

        return content_type


def decode_audio(audio_input: str) -> str:
    """Decode audio from base64, URL, or data URI to a temporary file path."""
    if not audio_input.startswith(("http://", "https://", "data:")):
        if len(audio_input) > MAX_BASE64_SIZE:
            raise ValueError(f"Base64 input too large")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    try:
        if audio_input.startswith("data:"):
            if len(audio_input) > MAX_BASE64_SIZE:
                raise ValueError(f"Data URI too large")
            header, encoded = audio_input.split(",", 1)
            audio_data = base64.b64decode(encoded)
            if len(audio_data) > MAX_AUDIO_SIZE_BYTES:
                raise ValueError(f"Decoded audio too large")
            with open(temp_path, "wb") as f:
                f.write(audio_data)

        elif audio_input.startswith(("http://", "https://")):
            if not is_url_safe(audio_input):
                raise ValueError("URL points to blocked or internal resource")

            ext = get_audio_extension(url=audio_input)
            if ext != ".wav":
                os.unlink(temp_path)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                temp_path = temp_file.name
                temp_file.close()

            download_with_limit(audio_input, temp_path)

        else:
            audio_data = base64.b64decode(audio_input)
            if len(audio_data) > MAX_AUDIO_SIZE_BYTES:
                raise ValueError(f"Decoded audio too large")
            with open(temp_path, "wb") as f:
                f.write(audio_data)

        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise ValueError(f"Failed to decode audio: {e}")


@app.cls(
    image=image,
    gpu="L4",  # Good balance of cost/performance for ASR
    volumes={"/cache/huggingface": model_volume},
    scaledown_window=60,  # Keep warm for 60 seconds
    allow_concurrent_inputs=4,  # Handle multiple requests per container
)
class ASRServer:
    """Qwen3-ASR Server with lazy model loading."""

    @modal.enter()
    def load_model(self):
        """Load model on container start (cold start)."""
        from qwen_asr import Qwen3ASR

        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
        print(f"Loading model: {model_name}")

        self.model = Qwen3ASR(model_name)

        print("Model loaded successfully")

    def _transcribe_single(
        self,
        audio_path: str,
        language: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> dict:
        """Transcribe a single audio file."""
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs["language"] = language

        result = self.model.transcribe(audio_path, **transcribe_kwargs)

        output = {
            "text": result.get("text", ""),
            "language": result.get("language", language or "auto"),
        }

        if return_timestamps:
            output["timestamps"] = result.get("segments", result.get("timestamps", []))
        else:
            output["timestamps"] = None

        return output

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        """Transcribe a single audio file."""
        audio_path = decode_audio(request.audio)

        try:
            result = self._transcribe_single(
                audio_path,
                language=request.language,
                return_timestamps=request.return_timestamps,
            )
            return TranscribeResponse(**result)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    @modal.fastapi_endpoint(method="POST")
    def transcribe_batch(self, request: BatchTranscribeRequest) -> BatchTranscribeResponse:
        """Transcribe multiple audio files."""
        if len(request.batch) > DEFAULT_MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum {DEFAULT_MAX_BATCH_SIZE}")

        results = []
        temp_files = []

        try:
            for item in request.batch:
                try:
                    audio_path = decode_audio(item.audio)
                    temp_files.append(audio_path)

                    result = self._transcribe_single(
                        audio_path,
                        language=item.language,
                        return_timestamps=item.return_timestamps,
                    )
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

            return BatchTranscribeResponse(results=results)
        finally:
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "model": "Qwen3-ASR-1.7B"}


@app.local_entrypoint()
def main():
    """Test the ASR server locally."""
    server = ASRServer()

    # Test with a sample audio URL
    test_url = "https://github.com/QwenLM/Qwen3-ASR/raw/main/assets/audio_en.wav"

    result = server.transcribe.remote(
        TranscribeRequest(audio=test_url, return_timestamps=True)
    )

    print(f"Transcription: {result.text}")
    print(f"Language: {result.language}")
    if result.timestamps:
        print(f"Timestamps: {result.timestamps[:3]}...")
