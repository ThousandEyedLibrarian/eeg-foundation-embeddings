"""LaBraM embedding extraction with braindecode and original backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shared.channel_maps import get_labram_matched_channels
from shared.config import FullConfig
from shared.edf_loader import load_edf
from shared.preprocessing import PreprocessingConfig, preprocess
from shared.zarr_writer import write_embeddings

logger = logging.getLogger(__name__)

LABRAM_EMBEDDING_DIM = 200


def load_model_braindecode(
    checkpoint: str = "",
    device: str = "cuda",
) -> Any:
    """Load LaBraM via braindecode wrapper.

    Args:
        checkpoint: Path to local checkpoint. Empty string uses the
            default braindecode pretrained weights.
        device: Torch device string.

    Returns:
        Model in eval mode on the specified device.
    """
    from braindecode.models import Labram

    if checkpoint:
        model = Labram(
            n_chans=128,
            n_times=2000,
            n_outputs=2,
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = Labram.from_pretrained("braindecode/labram-pretrained")

    model = model.to(device)
    model.eval()
    return model


def load_model_original(
    checkpoint: str = "/opt/labram-base.pth",
    device: str = "cuda",
) -> Any:
    """Load LaBraM from the original repository implementation.

    Args:
        checkpoint: Path to the original LaBraM checkpoint.
        device: Torch device string.

    Returns:
        Model in eval mode on the specified device.
    """
    import sys
    # The original repo code must be on the path
    if "/app/labram" not in sys.path:
        sys.path.insert(0, "/app/labram")

    from labram_original.modeling_finetune import labram_base_patch200_200

    model = labram_base_patch200_200(num_classes=2, use_mean_pooling=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings_braindecode(
    model: Any,
    windows: np.ndarray,
    input_chans: list[int],
    batch_size: int = 4,
    device: str = "cuda",
) -> np.ndarray:
    """Extract embeddings using the braindecode backend.

    Args:
        model: Loaded braindecode Labram model.
        windows: Array of shape (n_windows, n_channels, n_times).
        input_chans: LaBraM position indices (1-based, CLS at 0).
        batch_size: Batch size for inference.
        device: Torch device string.

    Returns:
        Embeddings array of shape (n_windows, 200).
    """
    n_windows = windows.shape[0]
    all_embeddings = []
    input_chans_tensor = torch.tensor(input_chans, dtype=torch.long, device=device)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = torch.from_numpy(windows[start:end]).float().to(device)

        # braindecode Labram.forward_features expects (B, n_chans, n_times)
        # and input_chans as a 1D tensor of position indices
        emb = model.forward_features(batch, input_chans_tensor)
        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def extract_embeddings_original(
    model: Any,
    windows: np.ndarray,
    input_chans: list[int],
    batch_size: int = 4,
    device: str = "cuda",
) -> np.ndarray:
    """Extract embeddings using the original LaBraM repo backend.

    The original model expects 4D input:
        (B, n_channels, n_patches, patch_size)
    where patch_size=200 and n_patches = n_times / 200.

    Args:
        model: Loaded original LaBraM model.
        windows: Array of shape (n_windows, n_channels, n_times).
            n_times must be divisible by 200.
        input_chans: LaBraM position indices (1-based, CLS at 0).
        batch_size: Batch size for inference.
        device: Torch device string.

    Returns:
        Embeddings array of shape (n_windows, 200).
    """
    from einops import rearrange

    n_windows = windows.shape[0]
    all_embeddings = []
    input_chans_tensor = torch.tensor(input_chans, dtype=torch.long, device=device)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = torch.from_numpy(windows[start:end]).float().to(device)

        # Reshape to (B, n_channels, n_patches, 200)
        batch_4d = rearrange(batch, "B N (A T) -> B N A T", T=200)

        emb = model.forward_features(batch_4d, input_chans_tensor)
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
        # Load EDF
        data, ch_names, edf_metadata = load_edf(
            edf_path,
            target_sr=config.preprocessing.target_sr,
        )

        # Match channels to LaBraM vocabulary
        matched_names, input_chans = get_labram_matched_channels(ch_names)
        if not matched_names:
            logger.warning("No LaBraM-compatible channels in %s, skipping", edf_path.name)
            return None

        # Select only matched channels from data
        ch_indices = [ch_names.index(name) for name in matched_names]
        data = data[ch_indices]

        logger.info(
            "Matched %d/%d channels for %s",
            len(matched_names), len(ch_names), edf_path.name,
        )

        # Preprocess
        windows, padding_mask, preproc_metadata = preprocess(
            data, config.preprocessing.target_sr, config.preprocessing,
        )

        # Load model if not provided
        if model is None:
            if config.model.backend == "original":
                model = load_model_original(
                    config.model.checkpoint or "/opt/labram-base.pth",
                    config.model.device,
                )
            else:
                model = load_model_braindecode(
                    config.model.checkpoint,
                    config.model.device,
                )

        # Extract embeddings
        if config.model.backend == "original":
            embeddings = extract_embeddings_original(
                model, windows, input_chans,
                config.model.batch_size, config.model.device,
            )
        else:
            embeddings = extract_embeddings_braindecode(
                model, windows, input_chans,
                config.model.batch_size, config.model.device,
            )

        # Build combined metadata
        metadata = {
            "model_name": "labram",
            "model_backend": config.model.backend,
            "embedding_dim": LABRAM_EMBEDDING_DIM,
            "channel_names": matched_names,
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
