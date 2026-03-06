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

## Dependencies

Uses `uv` for all package management. Dependencies declared in `pyproject.toml` with optional extras:

- `dev` - pytest (for running tests locally)
- `labram` - braindecode, timm (LaBraM-specific, used in Docker)
- `reve` - transformers, safetensors, huggingface_hub (REVE-specific, used in Docker)

PyTorch is excluded from `pyproject.toml` because labram and reve need different CUDA-specific builds. It is installed separately in each Dockerfile.

```bash
# Local dev (no torch needed for tests)
uv sync --extra dev
uv run pytest tests/

# Add a new package
uv add <package-name>
```

## Running Tests

```bash
uv sync --extra dev
uv run pytest tests/
```

## Building Docker Images

```bash
docker build -t labram-embeddings -f labram/Dockerfile .
docker build -t reve-embeddings -f reve/Dockerfile .
```
