# CLAUDE.md

## Project Overview

Docker images for extracting embeddings from raw EDF files using EEG foundation models (LaBraM, REVE). Each model gets its own container with the correct CUDA/PyTorch versions.

## Style Guidelines

- **Australian English** - colour, behaviour, normalisation, analyse, etc.
- **No emojis** in any context
- **No emdashes** - use hyphens or restructure sentences

## Architecture

- `shared/` - Common modules used by all models (EDF loading, preprocessing, Zarr output, channel mapping, config)
- `labram/` - LaBraM-specific extraction, Dockerfile, config
- `reve/` - REVE-specific extraction, Dockerfile, config
- `tests/` - Unit tests using synthetic EDF data (no GPU needed)

## Key Design Decisions

- One .zarr per EDF file
- GPU only (no CPU fallback)
- Default aggregation: mean pooling stored alongside per-window embeddings
- Zarr format v2 for cross-language compatibility (MATLAB, Julia)
- LaBraM supports both braindecode wrapper and original repo backend (config flag)
- REVE requires HF_TOKEN at runtime for gated model access

## Running Tests

```bash
python -m pytest tests/
```

## Building Docker Images

```bash
docker build -t labram-embeddings -f labram/Dockerfile .
docker build -t reve-embeddings -f reve/Dockerfile .
```
