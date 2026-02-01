FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure PATH includes standard bin directories
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

# Environment variables
ENV HF_HOME=/runpod-volume
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV MODEL_NAME=Qwen/Qwen3-ASR-1.7B
ENV MAX_BATCH_SIZE=32
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install Python dependencies
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install runpod
RUN python3 -m pip install hf-transfer
RUN python3 -m pip install soundfile librosa pydub
RUN python3 -m pip install "transformers>=4.40.0" accelerate sentencepiece

# Install flash-attn for better performance (optional, may fail on some GPUs)
RUN python3 -m pip install flash-attn --no-build-isolation || \
    echo "flash-attn installation skipped (not critical)"

# Remove Debian-installed blinker that conflicts with flask dependency
RUN pip uninstall -y blinker || rm -rf /usr/lib/python3/dist-packages/blinker* || true

# Install qwen-asr from PyPI
RUN python3 -m pip install qwen-asr

RUN python3 -m pip cache purge

# Copy handler
COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
