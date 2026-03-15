"""BENDR embedding extraction using braindecode."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shared.channel_maps import get_bendr_matched_channels
from shared.config import FullConfig
from shared.edf_loader import load_edf
from shared.preprocessing import preprocess
from shared.zarr_writer import write_embeddings

logger = logging.getLogger(__name__)

BENDR_EMBEDDING_DIM = 512
BENDR_EXPECTED_CHANNELS = 19
BENDR_MIN_CHANNELS = 15


def load_model(
    checkpoint: str = "",
    device: str = "cuda",
) -> Any:
    """Load BENDR via braindecode wrapper.

    Args:
        checkpoint: Path to local checkpoint. Empty string uses the
            default braindecode pretrained weights.
        device: Torch device string.

    Returns:
        Model in eval mode on the specified device.
    """
    from braindecode.models import BENDR

    if checkpoint:
        model = BENDR(
            n_chans=20,
            n_times=2560,
            n_outputs=2,
            final_layer=False,
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        # n_outputs and n_times are required by the constructor even though
        # final_layer=False means the classification head is discarded.
        model = BENDR.from_pretrained(
            "braindecode/braindecode-bendr",
            n_outputs=2,
            n_times=2560,
            final_layer=False,
        )

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    model: Any,
    windows: np.ndarray,
    batch_size: int = 4,
    device: str = "cuda",
) -> np.ndarray:
    """Extract CLS token embeddings from BENDR.

    Args:
        model: Loaded braindecode BENDR model.
        windows: Array of shape (n_windows, 20, n_times).
        batch_size: Batch size for inference.
        device: Torch device string.

    Returns:
        Embeddings array of shape (n_windows, 512).
    """
    n_windows = windows.shape[0]
    all_embeddings = []

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = torch.from_numpy(windows[start:end]).float().to(device)

        encoded = model.encoder(batch)
        context = model.contextualizer(encoded)
        # Index 0 along the temporal dimension is the CLS token
        emb = context[:, :, 0]
        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def process_edf(
    edf_path: Path,
    output_dir: Path,
    config: FullConfig,
    model: Any = None,
) -> Path | None:
    """Process a single EDF file and write embeddings to Zarr.

    Args:
        edf_path: Path to the EDF file.
        output_dir: Directory for Zarr output.
        config: Full configuration.
        model: Pre-loaded model (to avoid reloading per file).

    Returns:
        Path to the created .zarr store, or None on failure.
    """
    try:
        data, ch_names, edf_metadata = load_edf(
            edf_path,
            target_sr=config.preprocessing.target_sr,
        )

        matched_names, matched_indices = get_bendr_matched_channels(ch_names)
        if len(matched_names) < BENDR_MIN_CHANNELS:
            logger.warning(
                "Only %d/%d BENDR channels matched in %s (need >= %d), skipping",
                len(matched_names), BENDR_EXPECTED_CHANNELS,
                edf_path.name, BENDR_MIN_CHANNELS,
            )
            return None

        # Select matched channels from data (already in BENDR canonical order)
        data = data[matched_indices]

        if len(matched_names) < BENDR_EXPECTED_CHANNELS:
            # Zero-fill missing channels to maintain 19-channel shape.
            # Channels are in canonical order so missing ones get zeros
            # at the correct positions.
            logger.warning(
                "Matched %d/%d BENDR channels for %s, zero-filling %d missing",
                len(matched_names), BENDR_EXPECTED_CHANNELS,
                edf_path.name, BENDR_EXPECTED_CHANNELS - len(matched_names),
            )
            n_missing = BENDR_EXPECTED_CHANNELS - len(matched_names)
            padding = np.zeros((n_missing, data.shape[1]), dtype=data.dtype)
            data = np.concatenate([data, padding], axis=0)

        logger.info(
            "Matched %d/%d channels for %s",
            len(matched_names), len(ch_names), edf_path.name,
        )

        # Compute relative amplitude channel (mean across all matched channels)
        rel_amp = data.mean(axis=0, keepdims=True)
        data = np.concatenate([data, rel_amp], axis=0)

        windows, padding_mask, preproc_metadata = preprocess(
            data, config.preprocessing.target_sr, config.preprocessing,
        )

        if model is None:
            model = load_model(
                config.model.checkpoint,
                config.model.device,
            )

        embeddings = extract_embeddings(
            model, windows,
            config.model.batch_size, config.model.device,
        )

        metadata = {
            "model_name": "bendr",
            "embedding_dim": BENDR_EMBEDDING_DIM,
            "channel_names": matched_names + ["REL_AMP"],
            **edf_metadata,
            "preprocessing": preproc_metadata,
        }

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
