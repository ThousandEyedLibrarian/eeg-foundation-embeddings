"""Channel name normalisation and mapping for EEG foundation models.

EDF files use inconsistent channel naming (e.g., 'EEG FP1-REF', 'Fp1',
'EEG Fp1', 'FP1'). This module normalises them to standard 10-20 names
and maps them to model-specific channel indices.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# LaBraM 128-channel order (from braindecode pretrained weights)
# Index 0 in the model's position embedding is reserved for the CLS token,
# so channel i maps to position i+1.
# --------------------------------------------------------------------------- #

LABRAM_STANDARD_1020: tuple[str, ...] = (
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2", "IZ", "O10",
    "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9H", "TTP7H", "TPP9H", "FTT10H", "TPP8H", "TPP10H",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
)

# Pre-built lookup: normalised name -> index in LABRAM_STANDARD_1020
_LABRAM_NAME_TO_IDX: dict[str, int] = {
    name: i for i, name in enumerate(LABRAM_STANDARD_1020)
}

# --------------------------------------------------------------------------- #
# REVE supported electrode names
# REVE uses 3D electrode positions from the reve-positions model.
# The position bank handles arbitrary 10-20/10-10/10-5 names, so we only
# list the common ones used for validation/documentation.
# --------------------------------------------------------------------------- #

REVE_COMMON_CHANNELS: tuple[str, ...] = (
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
)

# --------------------------------------------------------------------------- #
# Aliases for old 10-20 naming conventions
# --------------------------------------------------------------------------- #

_CHANNEL_ALIASES: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

# Non-EEG channel patterns to exclude
_NON_EEG_PATTERNS: tuple[str, ...] = (
    "EMG", "ECG", "EKG", "EOG", "PHOTIC", "IBI", "BURSTS", "SUPPR",
    "RESP", "SPO2", "HR", "DC", "STI", "TRIGGER", "EVENT", "MK",
    "REF", "GND", "STATUS", "ANNOTATIONS", "EDF",
)


def normalise_channel_name(raw_name: str) -> str:
    """Normalise an EDF channel name to standard 10-20 uppercase form.

    Strips common prefixes (EEG, POL), suffixes (-REF, -LE, -CAR, -AVG),
    whitespace, and handles case/alias variations.

    Examples:
        'EEG FP1-REF' -> 'FP1'
        'EEG Fp1-LE'  -> 'FP1'
        'POL T3'      -> 'T3'  (kept as-is, valid LaBraM position)
        'C3'          -> 'C3'
    """
    name = raw_name.strip()

    # Strip common EDF prefixes
    for prefix in ("EEG ", "EEG-", "POL ", "POL-", "BIO "):
        if name.upper().startswith(prefix):
            name = name[len(prefix):]
            break

    # Strip common reference suffixes
    for suffix in ("-REF", "-LE", "-CAR", "-AVG", "-AR", "-LER"):
        if name.upper().endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Strip whitespace that may remain
    name = name.strip()

    # Uppercase for consistent matching
    name = name.upper()

    # Apply aliases (old 10-20 names)
    # Note: only apply aliases for unipolar channels, not if they appear
    # in bipolar montage names (e.g., T3 in LABRAM_STANDARD_1020 is kept as-is
    # because it's a valid position in the 128-channel set)
    if name in _CHANNEL_ALIASES and name not in _LABRAM_NAME_TO_IDX:
        name = _CHANNEL_ALIASES[name]

    return name


def _is_non_eeg(name: str) -> bool:
    """Check if a channel name looks like a non-EEG channel."""
    upper = name.upper()
    for pattern in _NON_EEG_PATTERNS:
        if upper == pattern:
            return True
        # Match pattern at start of name (e.g., EMG1, ECG2)
        if upper.startswith(pattern):
            return True
        # Match as a word boundary (e.g., "LEFT EOG")
        if re.search(rf'\b{pattern}\b', upper):
            return True
    return False


def detect_channels(raw_ch_names: list[str]) -> list[str]:
    """Extract and normalise EEG channel names, filtering out non-EEG channels.

    Normalisation is applied first (stripping prefixes like 'EEG ' and
    suffixes like '-REF'), then the non-EEG check runs on the clean name.
    This prevents false positives from suffixes like '-REF' matching the
    'REF' exclusion pattern.

    Args:
        raw_ch_names: Raw channel names from an EDF file.

    Returns:
        List of normalised EEG channel names.
    """
    eeg_channels = []
    for raw_name in raw_ch_names:
        normalised = normalise_channel_name(raw_name)

        # Skip if normalised name is empty or looks non-EEG
        if not normalised or _is_non_eeg(normalised):
            continue

        eeg_channels.append(normalised)

    return eeg_channels


def map_channels_to_labram(ch_names: list[str]) -> list[int]:
    """Map normalised channel names to LaBraM position embedding indices.

    LaBraM reserves index 0 for the CLS token, so channel i in
    LABRAM_STANDARD_1020 maps to position i+1.

    Args:
        ch_names: Normalised (uppercase) channel names.

    Returns:
        List of position indices (1-based, CLS at 0) for each matched channel.
        Channels not found in LABRAM_STANDARD_1020 are silently skipped.
        Returns indices in the same order as the matched channels.
    """
    indices = []
    for name in ch_names:
        idx = _LABRAM_NAME_TO_IDX.get(name)
        if idx is not None:
            indices.append(idx + 1)  # +1 for CLS token at position 0
    return indices


def get_labram_matched_channels(
    ch_names: list[str],
) -> tuple[list[str], list[int]]:
    """Get the subset of channels that match LaBraM's vocabulary.

    Args:
        ch_names: Normalised (uppercase) channel names.

    Returns:
        Tuple of (matched_channel_names, position_indices).
        Position indices are 1-based (CLS token at 0).
    """
    matched_names = []
    matched_indices = []
    for name in ch_names:
        idx = _LABRAM_NAME_TO_IDX.get(name)
        if idx is not None:
            matched_names.append(name)
            matched_indices.append(idx + 1)
    return matched_names, matched_indices


# --------------------------------------------------------------------------- #
# BENDR 19-channel standard 10-20 order
# BENDR uses old-style names (T3/T4/T5/T6) and expects 20 channels total:
# 19 standard 10-20 EEG + 1 relative amplitude channel computed at runtime.
# --------------------------------------------------------------------------- #

BENDR_STANDARD_1020: tuple[str, ...] = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T3", "C3", "CZ", "C4", "T4",
    "T5", "P3", "PZ", "P4", "T6",
    "O1", "O2",
)

# Pre-built lookup: normalised name -> index in BENDR_STANDARD_1020
_BENDR_NAME_TO_IDX: dict[str, int] = {
    name: i for i, name in enumerate(BENDR_STANDARD_1020)
}

# BENDR uses old-style 10-20 names, so we need reverse aliases
# (modern -> old) for matching channels that have been normalised
_BENDR_REVERSE_ALIASES: dict[str, str] = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}


def get_bendr_matched_channels(
    ch_names: list[str],
) -> tuple[list[str], list[int]]:
    """Get channels matching BENDR's 19 standard 10-20 channels in model order.

    Iterates over BENDR_STANDARD_1020 to ensure returned channels are always
    in the canonical order expected by the pretrained model. Handles both
    old-style (T3/T4/T5/T6) and modern (T7/T8/P7/P8) names.

    Args:
        ch_names: Normalised (uppercase) channel names from the EDF file.

    Returns:
        Tuple of (matched_channel_names, indices_into_ch_names).
        matched_channel_names uses BENDR's expected names (old-style)
        and are ordered to match BENDR_STANDARD_1020.
    """
    # Build lookup from normalised EDF names to their index in ch_names.
    # Map both direct and reverse-aliased names to support old and modern
    # naming conventions.
    name_to_data_idx: dict[str, int] = {}
    for i, name in enumerate(ch_names):
        if name in _BENDR_NAME_TO_IDX:
            name_to_data_idx[name] = i
        elif name in _BENDR_REVERSE_ALIASES:
            name_to_data_idx[_BENDR_REVERSE_ALIASES[name]] = i

    # Iterate in BENDR canonical order so output matches model expectations
    matched_names = []
    matched_indices = []
    for bendr_name in BENDR_STANDARD_1020:
        if bendr_name in name_to_data_idx:
            matched_names.append(bendr_name)
            matched_indices.append(name_to_data_idx[bendr_name])

    return matched_names, matched_indices


def map_channels_to_reve(
    ch_names: list[str], pos_bank
) -> "torch.Tensor":
    """Get 3D electrode positions for REVE using its position bank model.

    Args:
        ch_names: Normalised channel names.
        pos_bank: The REVE position bank model
            (from AutoModel.from_pretrained('brain-bzh/reve-positions')).

    Returns:
        Tensor of shape (n_channels, 3) with 3D electrode positions.
    """
    return pos_bank(ch_names)
