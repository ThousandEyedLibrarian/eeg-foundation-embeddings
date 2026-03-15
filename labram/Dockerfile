FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

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

# PyTorch (CUDA 11.8) - installed separately due to custom index
RUN uv pip install --system \
    torch==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install project deps from lockfile (pinned versions for reproducibility)
RUN uv export --extra labram --no-emit-project --frozen > /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && uv pip install --system --no-deps . \
    && rm /tmp/requirements.txt

# Pre-download braindecode checkpoint
RUN python3 -c "\
from braindecode.models import Labram; \
model = Labram.from_pretrained('braindecode/labram-pretrained'); \
print('braindecode checkpoint cached')"

# Pre-download original checkpoint (optional - only needed for --backend original)
RUN curl -L -o /opt/labram-base.pth \
    https://huggingface.co/935963004/LaBraM/resolve/main/labram-base.pth \
    || echo "Warning: could not download original LaBraM checkpoint"

# Copy application code (after deps for better caching)
COPY shared/ /app/shared/
COPY labram/ /app/labram/

ENTRYPOINT ["/app/labram/entrypoint.sh"]
