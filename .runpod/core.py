"""
RunPod Serverless Handler for Qwen3-ASR

Supports single and batch audio transcription with optional language specification
and timestamp output.
"""

import ipaddress
import logging
import os
import base64
import socket
import tempfile
import urllib.request
from typing import Optional
from urllib.parse import urlparse

import runpod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MAX_AUDIO_SIZE_MB = 100
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024
MAX_BASE64_SIZE = 150 * 1024 * 1024  # ~100MB after decode
DEFAULT_MAX_BATCH_SIZE = 32
URL_TIMEOUT_SECONDS = 30

# Security: Blocked hosts and networks for SSRF protection
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"}
BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local/metadata
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
]

# Allowed content types for audio
ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac",
    "audio/ogg", "audio/webm",
    "audio/mp4", "audio/m4a", "audio/x-m4a",
    "application/octet-stream",  # Fallback for some servers
}

# Lazy-loaded model instance
MODEL = None


def get_model():
    """Load model once on cold start."""
    global MODEL

    if MODEL is None:
        from qwen_asr import Qwen3ASR

        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
        logger.info(f"Loading model: {model_name}")

        MODEL = Qwen3ASR(model_name)

        logger.info("Model loaded successfully")

    return MODEL


def is_url_safe(url: str) -> bool:
    """
    Validate URL is not targeting internal resources (SSRF protection).

    Returns True if URL is safe to fetch, False otherwise.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False

        # Check against blocked hostnames
        if hostname.lower() in BLOCKED_HOSTS:
            return False

        # Resolve hostname and check IP against blocked networks
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
    """
    Download URL with size limit and timeout to prevent DoS.

    Returns the content-type header value.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Qwen3-ASR/1.0"})

    with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as response:
        # Check content length header
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_size:
            raise ValueError(f"File too large: {int(content_length)} bytes exceeds {max_size} limit")

        # Validate content type
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type and content_type not in ALLOWED_CONTENT_TYPES:
            logger.warning(f"Unexpected content type: {content_type}, proceeding anyway")

        # Download with streaming size check
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
    """
    Decode audio from base64, URL, or data URI to a temporary file path.

    Returns the path to a temporary audio file.
    """
    # Validate base64 size before decoding
    if not audio_input.startswith(("http://", "https://", "data:")):
        if len(audio_input) > MAX_BASE64_SIZE:
            raise ValueError(f"Base64 input too large: {len(audio_input)} bytes")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    try:
        if audio_input.startswith("data:"):
            # Data URI format
            if len(audio_input) > MAX_BASE64_SIZE:
                raise ValueError(f"Data URI too large: {len(audio_input)} bytes")
            header, encoded = audio_input.split(",", 1)
            audio_data = base64.b64decode(encoded)
            if len(audio_data) > MAX_AUDIO_SIZE_BYTES:
                raise ValueError(f"Decoded audio too large: {len(audio_data)} bytes")
            with open(temp_path, "wb") as f:
                f.write(audio_data)

        elif audio_input.startswith(("http://", "https://")):
            # URL - validate for SSRF first
            if not is_url_safe(audio_input):
                raise ValueError("URL points to blocked or internal resource")

            # Get file extension from URL
            ext = get_audio_extension(url=audio_input)
            if ext != ".wav":
                # Create new temp file with correct extension
                os.unlink(temp_path)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                temp_path = temp_file.name
                temp_file.close()

            content_type = download_with_limit(audio_input, temp_path)

            # Update extension based on content-type if different
            if content_type:
                expected_ext = get_audio_extension(content_type=content_type)
                if expected_ext != ext and expected_ext != ".wav":
                    new_path = temp_path.rsplit(".", 1)[0] + expected_ext
                    os.rename(temp_path, new_path)
                    temp_path = new_path

        else:
            # Plain base64
            audio_data = base64.b64decode(audio_input)
            if len(audio_data) > MAX_AUDIO_SIZE_BYTES:
                raise ValueError(f"Decoded audio too large: {len(audio_data)} bytes")
            with open(temp_path, "wb") as f:
                f.write(audio_data)

        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise ValueError(f"Failed to decode audio: {sanitize_error(e)}")


def sanitize_error(error: Exception) -> str:
    """Return safe error message without internal path details."""
    error_str = str(error)
    # Remove internal path information
    if "/app/" in error_str or "/tmp/" in error_str or "\\tmp\\" in error_str.lower():
        return "Audio processing failed"
    return error_str


def transcribe_single(
    model,
    audio_path: str,
    language: Optional[str] = None,
    return_timestamps: bool = False,
) -> dict:
    """
    Transcribe a single audio file.

    Args:
        model: Qwen3ASR model instance
        audio_path: Path to audio file
        language: Optional language hint (e.g., "English", "Chinese")
        return_timestamps: Whether to return word/segment timestamps

    Returns:
        dict with text, language, and optional timestamps
    """
    transcribe_kwargs = {}

    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(audio_path, **transcribe_kwargs)

    output = {
        "text": result.get("text", ""),
        "language": result.get("language", language or "auto"),
    }

    if return_timestamps:
        output["timestamps"] = result.get("segments", result.get("timestamps", []))
    else:
        output["timestamps"] = None

    return output


def process_request(job_input: dict) -> dict:
    """
    Process a single or batch transcription request.

    Single input format:
    {
        "audio": "base64_or_url",
        "language": "English",      # optional
        "return_timestamps": true   # optional
    }

    Batch input format:
    {
        "batch": [
            {"audio": "...", "language": "Chinese", "return_timestamps": true},
            {"audio": "...", "language": null, "return_timestamps": false}
        ]
    }
    """
    model = get_model()
    max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", str(DEFAULT_MAX_BATCH_SIZE)))

    if "batch" in job_input:
        batch = job_input["batch"]

        if len(batch) > max_batch_size:
            raise ValueError(
                f"Batch size {len(batch)} exceeds maximum {max_batch_size}"
            )

        results = []
        temp_files = []

        try:
            for item in batch:
                audio_input = item.get("audio")
                if not audio_input:
                    results.append({"error": "Missing 'audio' field"})
                    continue

                try:
                    audio_path = decode_audio(audio_input)
                    temp_files.append(audio_path)

                    result = transcribe_single(
                        model,
                        audio_path,
                        language=item.get("language"),
                        return_timestamps=item.get("return_timestamps", False),
                    )
                    results.append(result)
                except Exception as e:
                    results.append({"error": sanitize_error(e)})

            return {"results": results}
        finally:
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)

    else:
        audio_input = job_input.get("audio")
        if not audio_input:
            raise ValueError("Missing 'audio' field in input")

        audio_path = decode_audio(audio_input)

        try:
            result = transcribe_single(
                model,
                audio_path,
                language=job_input.get("language"),
                return_timestamps=job_input.get("return_timestamps", False),
            )
            return result
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)


def handler(event):
    """
    RunPod serverless handler entry point.

    Args:
        event: RunPod event dictionary with 'input' key

    Returns:
        Transcription result or error
    """
    try:
        job_input = event["input"]

        if not job_input:
            return {"error": "No input provided"}

        return process_request(job_input)

    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": sanitize_error(e)}


# Pre-load model on cold start (lazy - only when handler is called)
# Model will be loaded on first request
logger.info("Qwen3-ASR RunPod Serverless Handler initialized")
