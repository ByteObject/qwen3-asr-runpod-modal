FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV HF_HOME=/runpod-volume
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV MODEL_NAME=Qwen/Qwen3-ASR-1.7B
ENV MAX_BATCH_SIZE=32
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    hf-transfer \
    soundfile \
    librosa \
    pydub \
    "transformers>=4.40.0" \
    accelerate \
    sentencepiece

# Install flash-attn for better performance (optional, may fail on some GPUs)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn installation skipped (not critical)"

# Install qwen-asr from GitHub
RUN pip install --no-cache-dir git+https://github.com/QwenLM/Qwen3-ASR.git

# Copy handler
COPY handler.py /app/handler.py

# Pre-download model during build (optional, speeds up cold start)
# Uncomment the following lines to bake the model into the image
# ARG HF_TOKEN
# RUN python -c "from qwen_asr import Qwen3ASR; Qwen3ASR('${MODEL_NAME}')"

CMD ["python", "-u", "/app/handler.py"]
