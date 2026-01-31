"""
RunPod Serverless Handler for Qwen3-ASR
"""
import runpod
import os
import base64
import tempfile
import logging
from typing import Optional

# Global model instance (loaded once on cold start)
MODEL = None

def load_model():
    """Load model once during cold start"""
    global MODEL
    if MODEL is not None:
        return MODEL

    print("Loading Qwen3-ASR model...")
    from qwen_asr import Qwen3ASR

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
    MODEL = Qwen3ASR(model_name)

    print(f"Model loaded: {model_name}")
    return MODEL

def transcribe(job):
    """
    RunPod handler function
    Input format: {"input": {"audio": "base64_or_url", "language": "English", "return_timestamps": true}}
    Output format: {"text": "...", "language": "...", "timestamps": [...]}
    """
    job_input = job["input"]

    audio_input = job_input.get("audio")
    if not audio_input:
        return {"error": "audio is required"}

    language = job_input.get("language", None)
    return_timestamps = job_input.get("return_timestamps", False)

    print(f"Transcribing audio...")

    # Load model if not already loaded
    model = load_model()

    # Decode audio to temp file
    temp_path = None
    try:
        temp_path = decode_audio(audio_input)

        # Transcribe
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(temp_path, **transcribe_kwargs)

        output = {
            "text": result.get("text", ""),
            "language": result.get("language", language or "auto"),
        }

        if return_timestamps:
            output["timestamps"] = result.get("segments", result.get("timestamps", []))

        print("Transcription complete!")
        return output

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def decode_audio(audio_input: str) -> str:
    """Decode audio from base64 or URL to a temporary file path."""
    import urllib.request

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    try:
        if audio_input.startswith("data:"):
            # Data URI format
            header, encoded = audio_input.split(",", 1)
            audio_data = base64.b64decode(encoded)
            with open(temp_path, "wb") as f:
                f.write(audio_data)

        elif audio_input.startswith(("http://", "https://")):
            # URL
            req = urllib.request.Request(audio_input, headers={"User-Agent": "Qwen3-ASR/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(temp_path, "wb") as f:
                    f.write(response.read())

        else:
            # Plain base64
            audio_data = base64.b64decode(audio_input)
            with open(temp_path, "wb") as f:
                f.write(audio_data)

        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise ValueError(f"Failed to decode audio: {e}")


if __name__ == "__main__":
    runpod.serverless.start({"handler": transcribe})
