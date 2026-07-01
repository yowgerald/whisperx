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

# torchvision 0.22.x _meta_registrations.py references torchvision.extension
# before it's available (extension C module fails to load or import order issue).
# Patch: guard with getattr so import doesn't crash when extension is missing.
# Must NOT import torchvision (would trigger the crash). Find file via site-packages.
RUN echo 'import pathlib as _pl, site as _site
for _sp in _site.getsitepackages():
    _f = _pl.Path(_sp) / "torchvision" / "_meta_registrations.py"
    if _f.exists():
        src = _f.read_text()
        old = "torchvision.extension._has_ops()"
        new = "getattr(torchvision, \"extension\", None) is not None and torchvision.extension._has_ops()"
        if old in src and new not in src:
            _f.write_text(src.replace(old, new))
            print("patched _meta_registrations.py")
        else:
            print("_meta_registrations.py already patched or pattern not found")
        break
' > /tmp/patch_tv.py && python /tmp/patch_tv.py && rm /tmp/patch_tv.py

# Verify torch still CUDA-built (pip must not swap base image's CUDA torch for CPU build)
RUN python -c "import torch; assert torch.version.cuda is not None, 'CUDA torch lost!'; print(f'torch {torch.__version__} + CUDA {torch.version.cuda} OK')"

# Verify torchvision imports cleanly (no circular import in _meta_registrations)
RUN python -c "import torchvision; print(f'torchvision {torchvision.__version__} OK')"

COPY transcribe_core.py /app/transcribe_core.py
COPY rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "rp_handler.py"]
