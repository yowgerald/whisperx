FROM runpod/pytorch:1.0.7-cu1290-torch271-ubuntu2204

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    HF_HOME=/runpod-volume/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# whisperx deps pull torchvision from PyPI → circular import in _meta_registrations
# (torchvision.extension not yet initialized when _meta_registrations references it).
# Overwrite with CUDA-index torchvision/torchaudio (matching base torch 2.7.1).
# --no-deps --force-reinstall keeps base image's CUDA torch untouched.
RUN pip install --upgrade pip && pip install -r requirements.txt && \
    pip install --no-deps --force-reinstall \
      torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
      --extra-index-url https://download.pytorch.org/whl/cu128

# Verify torch still CUDA-built (pip must not swap base image's CUDA torch for CPU build)
RUN python -c "import torch; assert torch.version.cuda is not None, 'CUDA torch lost!'; print(f'torch {torch.__version__} + CUDA {torch.version.cuda} OK')"

# Verify torchvision imports cleanly (no circular import in _meta_registrations)
RUN python -c "import torchvision; print(f'torchvision {torchvision.__version__} OK')"

COPY transcribe_core.py /app/transcribe_core.py
COPY rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "rp_handler.py"]
