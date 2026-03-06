"""Model-agnostic EEG preprocessing: filtering, windowing, normalisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for EEG preprocessing pipeline."""

    target_sr: int = 200
    bandpass_low: float = 0.1
    bandpass_high: float = 75.0
    notch_freq: float = 50.0
    skip_start_sec: float = 0.0
    max_duration_sec: float = 0.0       # 0 = use all
    min_duration_sec: float = 0.0       # 0 = no minimum
    window_size_sec: float = 10.0
    window_overlap_sec: float = 0.0
    normalisation: str = "none"         # "none", "zscore", "robust"
    clip_std: float = 0.0              # 0 = no clipping
    scale_factor: float = 1.0          # e.g., 1/100 for LaBraM
    units: str = "uV"


def apply_filters(
    data: np.ndarray,
    sr: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Apply bandpass and notch filters.

    Args:
        data: EEG data of shape (n_channels, n_samples).
        sr: Sample rate in Hz.
        config: Preprocessing configuration.

    Returns:
        Filtered data, same shape as input.
    """
    filtered = data.copy()

    # Bandpass filter
    if config.bandpass_low > 0 and config.bandpass_high > 0:
        nyq = sr / 2.0
        low = config.bandpass_low / nyq
        high = config.bandpass_high / nyq
        # Clamp to valid range
        high = min(high, 0.999)
        if low < high:
            sos = scipy_signal.butter(4, [low, high], btype="band", output="sos")
            filtered = scipy_signal.sosfiltfilt(sos, filtered, axis=-1)

    # Notch filter
    if config.notch_freq > 0:
        nyq = sr / 2.0
        if config.notch_freq < nyq:
            b, a = scipy_signal.iirnotch(config.notch_freq, Q=30.0, fs=sr)
            filtered = scipy_signal.filtfilt(b, a, filtered, axis=-1)

    return filtered


def extract_time_window(
    data: np.ndarray,
    sr: int,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply skip/trim/reject based on duration constraints.

    Args:
        data: EEG data of shape (n_channels, n_samples).
        sr: Sample rate in Hz.
        config: Preprocessing configuration.

    Returns:
        Tuple of (trimmed_data, info_dict).

    Raises:
        ValueError: If recording is shorter than min_duration_sec.
    """
    n_samples = data.shape[-1]
    total_duration = n_samples / sr
    info = {"original_samples": n_samples, "original_duration_sec": total_duration}

    # Check minimum duration
    if config.min_duration_sec > 0 and total_duration < config.min_duration_sec:
        raise ValueError(
            f"Recording duration ({total_duration:.1f}s) is shorter than "
            f"minimum ({config.min_duration_sec:.1f}s)"
        )

    # Skip start
    start_sample = 0
    if config.skip_start_sec > 0:
        start_sample = int(config.skip_start_sec * sr)
        if start_sample >= n_samples:
            raise ValueError(
                f"skip_start_sec ({config.skip_start_sec}s) exceeds "
                f"recording duration ({total_duration:.1f}s)"
            )

    # Max duration
    if config.max_duration_sec > 0:
        max_samples = int(config.max_duration_sec * sr)
        end_sample = min(start_sample + max_samples, n_samples)
    else:
        end_sample = n_samples

    trimmed = data[..., start_sample:end_sample]
    info["used_start_sec"] = start_sample / sr
    info["used_end_sec"] = end_sample / sr
    info["used_duration_sec"] = (end_sample - start_sample) / sr

    return trimmed, info


def _apply_normalisation(
    data: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Apply per-channel normalisation across the full recording."""
    if config.normalisation == "none":
        return data

    if config.normalisation == "zscore":
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        data = (data - mean) / std

    elif config.normalisation == "robust":
        median = np.median(data, axis=-1, keepdims=True)
        iqr = np.percentile(data, 75, axis=-1, keepdims=True) - np.percentile(
            data, 25, axis=-1, keepdims=True
        )
        iqr = np.where(iqr == 0, 1.0, iqr)
        data = (data - median) / iqr

    else:
        raise ValueError(f"Unknown normalisation: {config.normalisation!r}")

    return data


def _apply_clipping(data: np.ndarray, clip_std: float) -> np.ndarray:
    """Clip values beyond N standard deviations per channel."""
    if clip_std <= 0:
        return data

    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    lower = mean - clip_std * std
    upper = mean + clip_std * std
    return np.clip(data, lower, upper)


def create_windows(
    data: np.ndarray,
    sr: int,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Split data into fixed-size windows with optional overlap and padding.

    Args:
        data: EEG data of shape (n_channels, n_samples).
        sr: Sample rate in Hz.
        config: Preprocessing configuration.

    Returns:
        Tuple of:
            - windows: (n_windows, n_channels, window_samples) float32
            - padding_mask: (n_windows,) bool - True for valid (non-padded) windows
            - timestamps: List of window start times in seconds
    """
    n_channels, n_samples = data.shape
    window_samples = int(config.window_size_sec * sr)
    overlap_samples = int(config.window_overlap_sec * sr)
    step_samples = window_samples - overlap_samples

    if step_samples <= 0:
        raise ValueError(
            f"Window overlap ({config.window_overlap_sec}s) must be less than "
            f"window size ({config.window_size_sec}s)"
        )

    # Calculate number of windows
    if n_samples <= window_samples:
        n_windows = 1
    else:
        n_windows = 1 + (n_samples - window_samples) // step_samples
        # Add one more window if there's a partial remainder
        remainder = (n_samples - window_samples) % step_samples
        if remainder > 0:
            n_windows += 1

    windows = np.zeros((n_windows, n_channels, window_samples), dtype=np.float32)
    padding_mask = np.ones(n_windows, dtype=bool)
    timestamps = []

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        timestamps.append(start / sr)

        if end <= n_samples:
            windows[i] = data[:, start:end]
        else:
            # Partial window - pad with zeros
            available = n_samples - start
            if available > 0:
                windows[i, :, :available] = data[:, start:]
            # Mark as padded if less than half the window is real data
            if available < window_samples // 2:
                padding_mask[i] = False

    return windows, padding_mask, timestamps


def preprocess(
    data: np.ndarray,
    sr: int,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Full preprocessing pipeline: filter, trim, normalise, clip, window.

    Args:
        data: Raw EEG data of shape (n_channels, n_samples) in microvolts.
        sr: Sample rate in Hz.
        config: Preprocessing configuration.

    Returns:
        Tuple of:
            - windows: (n_windows, n_channels, window_samples) float32
            - padding_mask: (n_windows,) bool
            - metadata: Dict with preprocessing details
    """
    # 1. Bandpass + notch filtering
    filtered = apply_filters(data, sr, config)

    # 2. Time window extraction (skip start, max duration, min check)
    trimmed, time_info = extract_time_window(filtered, sr, config)

    # 3. Normalisation (per-channel, across recording)
    normalised = _apply_normalisation(trimmed, config)

    # 4. Clipping
    clipped = _apply_clipping(normalised, config.clip_std)

    # 5. Scale factor (e.g., /100 for LaBraM)
    if config.scale_factor != 1.0:
        clipped = clipped * config.scale_factor

    # 6. Windowing
    windows, padding_mask, timestamps = create_windows(clipped, sr, config)

    metadata = {
        **time_info,
        "n_windows": int(len(windows)),
        "n_valid_windows": int(padding_mask.sum()),
        "window_size_sec": config.window_size_sec,
        "window_overlap_sec": config.window_overlap_sec,
        "window_timestamps_sec": timestamps,
        "bandpass": [config.bandpass_low, config.bandpass_high],
        "notch_freq": config.notch_freq,
        "target_sr": config.target_sr,
        "normalisation": config.normalisation,
        "clip_std": config.clip_std,
        "scale_factor": config.scale_factor,
    }

    logger.info(
        "Preprocessed: %d windows (%d valid) from %.1fs recording",
        len(windows), padding_mask.sum(), time_info["used_duration_sec"],
    )

    return windows, padding_mask, metadata
