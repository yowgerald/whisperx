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

RUN pip install --upgrade pip && pip install -r requirements.txt

# Reinstall PyTorch family from consistent index so torchvision C++ extensions
# match the torch build (fixes "operator torchvision::nms does not exist")
RUN pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify torch still CUDA-built (pip must not swap base image's CUDA torch for CPU build)
RUN python -c "import torch; assert torch.version.cuda is not None, 'CUDA torch lost!'; print(f'torch {torch.__version__} + CUDA {torch.version.cuda} OK')"

# Verify torchvision imports cleanly
RUN python -c "import torchvision; print(f'torchvision {torchvision.__version__} OK')"

COPY transcribe_core.py /app/transcribe_core.py
COPY rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "rp_handler.py"]
