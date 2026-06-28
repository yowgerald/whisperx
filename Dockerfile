# Python 3.11 chosen deliberately: WhisperX requires Python >=3.10,<3.14,
# and 3.11 has the broadest, most stable wheel availability for torch/torchaudio.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# ffmpeg: required for audio decoding
# git: required by some pip installs that build from source
# libsndfile1: required by audio processing libs (soundfile/torchaudio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch/torchaudio FIRST, from PyTorch's CPU wheel index.
# Doing this before installing whisperx avoids pip accidentally pulling
# in the much larger CUDA-enabled build.
RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY app.py /app/app.py

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
