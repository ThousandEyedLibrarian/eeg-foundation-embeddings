FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

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

# PyTorch (CUDA 12.4)
RUN pip3 install --no-cache-dir \
    torch==2.4.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Python dependencies
RUN pip3 install --no-cache-dir \
    transformers>=4.57.1 \
    safetensors \
    huggingface_hub \
    mne \
    numpy \
    scipy \
    zarr">=3,<4" \
    numcodecs \
    pyyaml \
    tqdm \
    einops

# Pre-download reve-positions (open access)
RUN python3 -c "\
from transformers import AutoModel; \
AutoModel.from_pretrained('brain-bzh/reve-positions', trust_remote_code=True); \
print('reve-positions cached')"

# reve-base is GATED - downloaded at runtime with HF_TOKEN
# User must: 1) Accept licence at huggingface.co/brain-bzh/reve-base
#             2) Pass HF_TOKEN env var at docker run

# Copy application code
COPY shared/ /app/shared/
COPY reve/ /app/reve/

WORKDIR /app
ENTRYPOINT ["/app/reve/entrypoint.sh"]
