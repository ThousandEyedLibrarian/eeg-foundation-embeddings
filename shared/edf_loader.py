"""EDF file loading with encoding fallback and automatic channel detection."""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np

from shared.channel_maps import detect_channels, normalise_channel_name

logger = logging.getLogger(__name__)

# Encodings to try when reading EDF headers
_ENCODINGS = ("utf-8", "latin1", "iso-8859-1")


def load_edf(
    path: Path,
    target_sr: int = 200,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Load an EDF file, extract EEG channels, and resample.

    Tries multiple encodings for EDF header parsing. Automatically detects
    EEG channels and filters out non-EEG signals (EMG, ECG, EOG, etc.).

    Args:
        path: Path to the EDF file.
        target_sr: Target sample rate in Hz. Data is resampled if different.

    Returns:
        Tuple of:
            - data: np.ndarray of shape (n_channels, n_samples) in microvolts
            - ch_names: List of normalised channel names
            - metadata: Dict with recording info (duration, sample rate, etc.)

    Raises:
        FileNotFoundError: If the EDF file does not exist.
        ValueError: If no EEG channels are found.
        RuntimeError: If the file cannot be read with any encoding.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    raw = _read_edf_with_fallback(path)

    # Get original metadata before any modifications
    original_sr = raw.info["sfreq"]
    original_duration = raw.times[-1] if len(raw.times) > 0 else 0.0

    # Detect EEG channels
    all_ch_names = raw.ch_names
    normalised_names = detect_channels(all_ch_names)

    if not normalised_names:
        raise ValueError(
            f"No EEG channels found in {path.name}. "
            f"Available channels: {all_ch_names}"
        )

    # Build a mapping from normalised name back to original name for picking
    normalised_set = set(normalised_names)
    norm_to_orig = {}
    for orig_name in all_ch_names:
        norm = normalise_channel_name(orig_name)
        if norm in normalised_set and norm not in norm_to_orig:
            norm_to_orig[norm] = orig_name

    # Pick only the EEG channels (using original names for MNE)
    orig_picks = [norm_to_orig[n] for n in normalised_names if n in norm_to_orig]
    raw.pick_channels(orig_picks)

    # Resample if needed
    if raw.info["sfreq"] != target_sr:
        raw.resample(target_sr)

    # Get data in microvolts
    # MNE internally stores in volts, scale to uV
    data = raw.get_data() * 1e6  # V -> uV

    # Build metadata
    meas_date = raw.info.get("meas_date")
    if isinstance(meas_date, datetime):
        recording_date = meas_date.isoformat()
    elif meas_date is not None:
        recording_date = str(meas_date)
    else:
        recording_date = None

    subject_info = raw.info.get("subject_info", {}) or {}

    metadata = {
        "edf_filename": path.name,
        "edf_path": str(path),
        "original_duration_sec": float(original_duration),
        "original_sample_rate": float(original_sr),
        "resampled_sample_rate": float(target_sr),
        "n_channels": len(normalised_names),
        "recording_date": recording_date,
        "subject_info": {
            k: str(v) for k, v in subject_info.items()
        } if subject_info else {},
    }

    # The channel order in data matches orig_picks order, which matches
    # the normalised_names order (filtered to those we could map back)
    final_ch_names = [n for n in normalised_names if n in norm_to_orig]

    return data, final_ch_names, metadata


def _read_edf_with_fallback(path: Path) -> mne.io.Raw:
    """Try reading an EDF file with multiple encodings."""
    last_error = None

    for encoding in _ENCODINGS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_edf(
                    str(path),
                    preload=True,
                    verbose=False,
                    encoding=encoding,
                )
            return raw
        except (UnicodeDecodeError, ValueError) as e:
            last_error = e
            logger.debug(
                "Failed to read %s with encoding %s: %s",
                path.name, encoding, e,
            )
            continue

    raise RuntimeError(
        f"Could not read {path.name} with any encoding "
        f"({', '.join(_ENCODINGS)}). Last error: {last_error}"
    )
