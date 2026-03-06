"""Zarr output writer - one .zarr store per EDF with rich metadata."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

logger = logging.getLogger(__name__)

SOFTWARE_VERSION = "0.1.0"


def _get_compressor(compression: str, level: int) -> Blosc:
    """Get a numcodecs Blosc compressor for zarr format 2."""
    return Blosc(cname=compression, clevel=level, shuffle=Blosc.SHUFFLE)


def write_embeddings(
    output_path: str | Path,
    embeddings: np.ndarray,
    padding_mask: np.ndarray,
    metadata: dict[str, Any],
    compression: str = "zstd",
    compression_level: int = 5,
    zarr_format: int = 2,
    store_per_window: bool = True,
    store_mean_pooled: bool = True,
) -> Path:
    """Write embeddings to a Zarr store.

    Creates one .zarr directory per EDF file with:
        - embeddings/: (n_windows, emb_dim) float32
        - padding_mask/: (n_windows,) bool
        - mean_embedding/: (emb_dim,) float32

    Args:
        output_path: Path for the .zarr output directory.
        embeddings: Array of shape (n_windows, emb_dim).
        padding_mask: Boolean array of shape (n_windows,).
            True = valid window, False = padded/invalid.
        metadata: Dict of metadata to store as root attributes.
        compression: Blosc compression algorithm name.
        compression_level: Compression level (1-9).
        zarr_format: Zarr format version (2 for cross-language compat).
        store_per_window: Whether to store per-window embeddings.
        store_mean_pooled: Whether to store mean-pooled embedding.

    Returns:
        Path to the created .zarr store.
    """
    output_path = Path(output_path)
    if output_path.suffix != ".zarr":
        output_path = output_path.with_suffix(".zarr")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compressor = _get_compressor(compression, compression_level)

    root = zarr.open_group(
        str(output_path),
        mode="w",
        zarr_format=zarr_format,
    )

    n_windows, emb_dim = embeddings.shape

    # Store per-window embeddings
    if store_per_window:
        root.create_array(
            name="embeddings",
            data=embeddings.astype(np.float32),
            chunks=(min(n_windows, 64), emb_dim),
            compressor=compressor,
        )

        root.create_array(
            name="padding_mask",
            data=padding_mask.astype(bool),
            chunks=(min(n_windows, 256),),
        )

    # Compute and store mean embedding (over valid windows only)
    if store_mean_pooled:
        valid_mask = padding_mask.astype(bool)
        if valid_mask.any():
            mean_emb = embeddings[valid_mask].mean(axis=0)
        else:
            mean_emb = np.zeros(emb_dim, dtype=np.float32)

        root.create_array(
            name="mean_embedding",
            data=mean_emb.astype(np.float32),
            chunks=(emb_dim,),
            compressor=compressor,
        )

    # Write metadata as root attributes
    root_attrs = {
        "embedding_dim": int(emb_dim),
        "n_windows": int(n_windows),
        "n_valid_windows": int(padding_mask.astype(bool).sum()),
        "created": datetime.now(timezone.utc).isoformat(),
        "software_version": SOFTWARE_VERSION,
        "zarr_format": zarr_format,
    }

    # Merge in caller-provided metadata (preprocessing params, model info, etc.)
    # Convert any non-serialisable values
    for key, value in metadata.items():
        root_attrs[key] = _make_json_safe(value)

    root.attrs.update(root_attrs)

    logger.info("Wrote %s (%d windows, %d-dim)", output_path, n_windows, emb_dim)
    return output_path


def _make_json_safe(value: Any) -> Any:
    """Convert numpy/pathlib types to JSON-serialisable Python types."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    return value
