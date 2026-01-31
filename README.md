# Qwen3-ASR Serverless

State-of-the-art speech recognition powered by [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), deployable on **Modal** or **RunPod**.

## Features

- **30+ Languages**: English, Chinese, Japanese, Korean, French, German, Spanish, and more
- **22 Chinese Dialects**: Cantonese, Shanghainese, Sichuanese, and regional variants
- **Automatic Language Detection**: No need to specify language
- **Timestamp Output**: Word/segment-level timestamps
- **Batch Processing**: Process multiple audio files in one request
- **Flexible Input**: Base64, URL, or data URI

## Model Variants

| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| Qwen3-ASR-1.7B | ~3.4GB | 4-6GB | Best accuracy (default) |
| Qwen3-ASR-0.6B | ~1.2GB | 2-3GB | Faster, cost-efficient |

## Deployment Options

| Platform | Best For | Cost | Setup |
|----------|----------|------|-------|
| **Modal** | Prototyping, fast iteration | ~$0.80/hr (L4) | `modal deploy` |
| **RunPod** | Production, cost optimization | ~$0.44/hr (L4) | Docker + runpodctl |

---

## Option 1: Modal (Recommended for Quick Start)

### Setup

```bash
pip install modal
modal setup  # One-time authentication
```

### Deploy

```bash
modal deploy main_modal.py
```

### Local Test

```bash
modal run main_modal.py
```

### API Usage (Modal)

```python
import requests

MODAL_ENDPOINT = "https://your-workspace--qwen3-asr-asrserver-transcribe.modal.run"

# Single transcription
response = requests.post(MODAL_ENDPOINT, json={
    "audio": "https://example.com/audio.wav",
    "language": "English",  # optional
    "return_timestamps": True  # optional
})

print(response.json())
# {
#     "text": "Hello, this is a test.",
#     "language": "English",
#     "timestamps": [...]
# }
```

### Batch Transcription (Modal)

```python
BATCH_ENDPOINT = "https://your-workspace--qwen3-asr-asrserver-transcribe-batch.modal.run"

response = requests.post(BATCH_ENDPOINT, json={
    "batch": [
        {"audio": "https://example.com/audio1.wav", "return_timestamps": True},
        {"audio": "https://example.com/audio2.wav", "language": "Chinese"}
    ]
})

print(response.json())
# {"results": [{"text": "...", "language": "...", "timestamps": [...]}, ...]}
```

### cURL Example (Modal)

```bash
curl -X POST "https://your-workspace--qwen3-asr-asrserver-transcribe.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "https://example.com/audio.wav",
    "return_timestamps": true
  }'
```

---

## Option 2: RunPod (Best for Production)

### Setup

```bash
# Build Docker image
docker build -t qwen3-asr .

# Push to registry (Docker Hub, GHCR, etc.)
docker tag qwen3-asr your-registry/qwen3-asr:latest
docker push your-registry/qwen3-asr:latest

# Deploy via RunPod console or CLI
runpodctl deploy
```

### API Usage (RunPod)

```python
import runpod
import base64

runpod.api_key = "YOUR_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Using URL
result = endpoint.run_sync({
    "input": {
        "audio": "https://example.com/audio.wav"
    }
})

# Using base64
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

result = endpoint.run_sync({
    "input": {
        "audio": audio_b64,
        "language": "English",  # optional
        "return_timestamps": True  # optional
    }
})

print(result)
# {
#     "text": "Hello, this is a test.",
#     "language": "English",
#     "timestamps": [...]
# }
```

### Batch Transcription (RunPod)

```python
result = endpoint.run_sync({
    "input": {
        "batch": [
            {"audio": "https://example.com/audio1.wav", "return_timestamps": True},
            {"audio": "https://example.com/audio2.wav", "language": "Chinese"},
            {"audio": audio_b64}
        ]
    }
})

print(result)
# {"results": [{"text": "...", "language": "...", "timestamps": [...]}, ...]}
```

### cURL Example (RunPod)

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio": "https://example.com/audio.wav",
      "return_timestamps": true
    }
  }'
```

---

## Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio` | string | Yes | Audio input (base64, URL, or data URI) |
| `language` | string | No | Language hint (auto-detect if not specified) |
| `return_timestamps` | boolean | No | Return word/segment timestamps (default: false) |
| `batch` | array | No | Array of audio items for batch processing |

## Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A
- WebM

## Supported Languages

**Major Languages**: English, Chinese (Mandarin), Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian, Arabic, Hindi, Vietnamese, Thai, Indonesian, Malay, Turkish, Polish, Dutch, Swedish

**Chinese Dialects**: Cantonese, Shanghainese, Sichuanese, Hokkien, Hakka, Wu, Min, Xiang, Gan, and more regional variants

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-1.7B` | Model to use |
| `MAX_BATCH_SIZE` | `32` | Maximum batch size |
| `HF_HOME` | `/cache/huggingface` | HuggingFace cache directory |

### GPU Requirements

- **Minimum VRAM**: 8GB (16GB recommended)
- **Actual Usage**: 4-6GB for 1.7B model
- **Recommended GPUs**: L4, RTX 4090, A10, A40, A100

## Performance

| Metric | Value |
|--------|-------|
| Cold Start | ~30-60s (model loading) |
| Inference | ~0.1-0.5x real-time |
| Max Batch Size | 32 files |

## File Structure

```
qwen3-asr/
├── main_modal.py        # Modal deployment
├── handler.py           # RunPod handler
├── Dockerfile           # RunPod container
├── runpod.toml          # RunPod config
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
└── .runpod/
    ├── hub.json
    └── tests.json
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## References

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-1.7B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ASR-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Modal Documentation](https://modal.com/docs)
- [RunPod Documentation](https://docs.runpod.io/)
