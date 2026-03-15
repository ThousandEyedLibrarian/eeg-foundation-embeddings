FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml uv.lock ./

# PyTorch (CUDA 12.4) - installed separately due to custom index
RUN uv pip install --system \
    torch==2.4.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install project deps from lockfile (pinned versions for reproducibility)
RUN uv export --extra reve --no-emit-project --frozen > /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && uv pip install --system --no-deps . \
    && rm /tmp/requirements.txt

# Pre-download reve-positions (open access)
RUN python3 -c "\
from transformers import AutoModel; \
AutoModel.from_pretrained('brain-bzh/reve-positions', trust_remote_code=True); \
print('reve-positions cached')"

# reve-base is GATED - downloaded at runtime with HF_TOKEN
# User must: 1) Accept licence at huggingface.co/brain-bzh/reve-base
#             2) Pass HF_TOKEN env var at docker run

# Copy application code (after deps for better caching)
COPY shared/ /app/shared/
COPY reve/ /app/reve/

ENTRYPOINT ["/app/reve/entrypoint.sh"]
