"""REVE embedding extraction via HuggingFace AutoModel."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shared.channel_maps import map_channels_to_reve
from shared.config import FullConfig
from shared.edf_loader import load_edf
from shared.preprocessing import preprocess
from shared.zarr_writer import write_embeddings

logger = logging.getLogger(__name__)

REVE_EMBEDDING_DIMS = {
    "base": 512,
    "large": 1250,
}


def load_model(
    size: str = "base",
    hf_token: str = "",
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load REVE model and position bank from HuggingFace.

    Args:
        size: Model size - 'base' (512-dim) or 'large' (1250-dim).
        hf_token: HuggingFace token for gated model access.
        device: Torch device string.

    Returns:
        Tuple of (model, pos_bank) both in eval mode on device.
    """
    from transformers import AutoModel

    token = hf_token or None

    # Position bank is open access
    pos_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions",
        trust_remote_code=True,
    )
    pos_bank = pos_bank.to(device)
    pos_bank.eval()

    # REVE model is gated - requires token
    model_name = f"brain-bzh/reve-{size}"
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
    )
    model = model.to(device)
    model.eval()

    logger.info("Loaded REVE-%s from %s", size, model_name)
    return model, pos_bank


@torch.no_grad()
def extract_embeddings(
    model: Any,
    pos_bank: Any,
    windows: np.ndarray,
    ch_names: list[str],
    batch_size: int = 2,
    device: str = "cuda",
) -> np.ndarray:
    """Extract embeddings from preprocessed EEG windows.

    REVE output is 4D: (batch, channels, patches, emb_dim).
    Aggregation: mean across patches, then mean across channels,
    giving one vector per window.

    Args:
        model: Loaded REVE model.
        pos_bank: Loaded REVE position bank.
        windows: Array of shape (n_windows, n_channels, n_times).
        ch_names: Normalised channel names matching windows axis 1.
        batch_size: Batch size for inference.
        device: Torch device string.

    Returns:
        Embeddings array of shape (n_windows, emb_dim).
    """
    n_windows = windows.shape[0]
    all_embeddings = []

    # Get 3D positions for channels (n_channels, 3)
    positions = map_channels_to_reve(ch_names, pos_bank)
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.float32)
    positions = positions.to(device)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = torch.from_numpy(windows[start:end]).float().to(device)
        B = batch.shape[0]

        # Expand positions for the batch: (n_channels, 3) -> (B, n_channels, 3)
        pos_batch = positions.unsqueeze(0).expand(B, -1, -1)

        # Forward pass - output shape: (B, n_channels, n_patches, emb_dim)
        output = model(batch, pos_batch)

        # Aggregate: mean over patches, then mean over channels -> (B, emb_dim)
        emb = output.mean(dim=2).mean(dim=1)
        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def process_edf(
    edf_path: Path,
    output_dir: Path,
    config: FullConfig,
    model: Any = None,
    pos_bank: Any = None,
) -> Path | None:
    """Process a single EDF file and write REVE embeddings to Zarr.

    Args:
        edf_path: Path to the EDF file.
        output_dir: Directory for Zarr output.
        config: Full configuration.
        model: Pre-loaded REVE model (to avoid reloading per file).
        pos_bank: Pre-loaded REVE position bank.

    Returns:
        Path to the created .zarr store, or None on failure.
    """
    try:
        # Load EDF
        data, ch_names, edf_metadata = load_edf(
            edf_path,
            target_sr=config.preprocessing.target_sr,
        )

        logger.info(
            "Loaded %s: %d channels, %.1fs",
            edf_path.name, len(ch_names), edf_metadata["original_duration_sec"],
        )

        # Preprocess
        windows, padding_mask, preproc_metadata = preprocess(
            data, config.preprocessing.target_sr, config.preprocessing,
        )

        # Load model if not provided
        if model is None or pos_bank is None:
            model, pos_bank = load_model(
                size=config.model.size,
                hf_token=config.model.hf_token,
                device=config.model.device,
            )

        # Determine embedding dim
        emb_dim = REVE_EMBEDDING_DIMS.get(config.model.size, 512)

        # Extract embeddings
        embeddings = extract_embeddings(
            model, pos_bank, windows, ch_names,
            config.model.batch_size, config.model.device,
        )

        # Build combined metadata
        metadata = {
            "model_name": "reve",
            "model_size": config.model.size,
            "embedding_dim": emb_dim,
            "channel_names": ch_names,
            **edf_metadata,
            "preprocessing": preproc_metadata,
        }

        # Write Zarr
        output_name = edf_path.stem
        output_path = output_dir / f"{output_name}.zarr"

        return write_embeddings(
            output_path=output_path,
            embeddings=embeddings,
            padding_mask=padding_mask,
            metadata=metadata,
            compression=config.output.zarr_compression,
            compression_level=config.output.zarr_compression_level,
            zarr_format=config.output.zarr_format,
            store_per_window=config.output.store_per_window,
            store_mean_pooled=config.output.store_mean_pooled,
        )

    except Exception:
        logger.exception("Failed to process %s", edf_path.name)
        return None
