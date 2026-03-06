# eeg-foundation-embeddings

Reusable Docker images for extracting embeddings from raw EDF files using EEG foundation models.

Each model runs in its own Docker container with the correct CUDA/PyTorch versions, taking raw EDF files as input and producing Zarr files containing per-window embeddings and mean-pooled summaries.

## Supported Models

| Model | Embedding Dim | CUDA | PyTorch | Source |
|-------|--------------|------|---------|--------|
| **LaBraM** | 200 | 11.8 | 2.0.1 | [GitHub](https://github.com/935963004/LaBraM) / [braindecode](https://braindecode.org/) |
| **REVE** | 512 (base) / 1250 (large) | 12.4 | 2.4.0 | [HuggingFace](https://huggingface.co/brain-bzh/reve-base) |

## Quick Start

### LaBraM

```bash
docker build -t labram-embeddings -f labram/Dockerfile .

# Single file
docker run --gpus all \
  -v /path/to/edfs:/data/input \
  -v /path/to/output:/data/output \
  labram-embeddings /data/input/recording.edf /data/output

# Entire directory
docker run --gpus all \
  -v /path/to/edfs:/data/input \
  -v /path/to/output:/data/output \
  labram-embeddings /data/input /data/output
```

### REVE

REVE is a gated model. You must first accept the licence at
[huggingface.co/brain-bzh/reve-base](https://huggingface.co/brain-bzh/reve-base),
then provide your HuggingFace token:

```bash
docker build -t reve-embeddings -f reve/Dockerfile .

docker run --gpus all \
  -e HF_TOKEN=hf_xxx \
  -v /path/to/edfs:/data/input \
  -v /path/to/output:/data/output \
  reve-embeddings /data/input /data/output
```

## Output Format

Each EDF produces one Zarr store:

```
recording.zarr/
  embeddings/        # (n_windows, emb_dim) float32
  padding_mask/      # (n_windows,) bool
  mean_embedding/    # (emb_dim,) float32
  .zattrs            # metadata (model, preprocessing, channels, timestamps, ...)
```

## Configuration

Both models accept a YAML config file and/or CLI overrides:

```bash
docker run --gpus all \
  -v /data:/data \
  labram-embeddings --config /data/config.yaml --batch-size 8 /data/input /data/output
```

See `labram/config.yaml` and `reve/config.yaml` for defaults.

## Development

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run tests (no GPU needed)
uv sync --extra dev
uv run pytest tests/
```

## Requirements

- Docker with NVIDIA Container Toolkit (`--gpus` support)
- NVIDIA GPU with sufficient VRAM (4+ GB recommended)
