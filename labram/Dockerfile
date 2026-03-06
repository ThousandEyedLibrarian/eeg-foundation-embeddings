FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# PyTorch (CUDA 11.8)
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Python dependencies
RUN pip3 install --no-cache-dir \
    braindecode \
    einops \
    mne \
    numpy \
    scipy \
    zarr">=3,<4" \
    numcodecs \
    pyyaml \
    tqdm \
    timm==0.4.12

# Pre-download braindecode checkpoint
RUN python3 -c "\
from braindecode.models import Labram; \
model = Labram.from_pretrained('braindecode/labram-pretrained'); \
print('braindecode checkpoint cached')"

# Pre-download original checkpoint
RUN curl -L -o /opt/labram-base.pth \
    https://huggingface.co/935963004/LaBraM/resolve/main/labram-base.pth \
    || echo "Warning: could not download original LaBraM checkpoint"

# Copy application code
COPY shared/ /app/shared/
COPY labram/ /app/labram/

WORKDIR /app
ENTRYPOINT ["/app/labram/entrypoint.sh"]
