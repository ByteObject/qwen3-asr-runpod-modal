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

# Clean cache after install
python3 -m pip cache purge
echo "Dependencies installed!"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"

echo "Setup complete!"
echo "Starting handler..."
python3 /workspace/handler.py
