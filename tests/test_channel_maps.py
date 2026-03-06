"""Tests for channel name normalisation and mapping."""

import pytest

from shared.channel_maps import (
    LABRAM_STANDARD_1020,
    detect_channels,
    get_labram_matched_channels,
    map_channels_to_labram,
    normalise_channel_name,
)


class TestNormaliseChannelName:
    """Tests for normalise_channel_name."""

    def test_plain_name(self):
        assert normalise_channel_name("C3") == "C3"

    def test_eeg_prefix_ref_suffix(self):
        assert normalise_channel_name("EEG FP1-REF") == "FP1"

    def test_eeg_prefix_le_suffix(self):
        assert normalise_channel_name("EEG Fp1-LE") == "FP1"

    def test_pol_prefix(self):
        assert normalise_channel_name("POL C3") == "C3"

    def test_uppercase_normalisation(self):
        assert normalise_channel_name("fp1") == "FP1"
        assert normalise_channel_name("Cz") == "CZ"

    def test_whitespace_stripping(self):
        assert normalise_channel_name("  FP1  ") == "FP1"

    def test_t3_not_aliased_when_in_labram(self):
        # T3 exists in LABRAM_STANDARD_1020, so it should NOT be aliased to T7
        assert normalise_channel_name("T3") == "T3"
        assert "T3" in LABRAM_STANDARD_1020

    def test_car_suffix(self):
        assert normalise_channel_name("C3-CAR") == "C3"

    def test_avg_suffix(self):
        assert normalise_channel_name("O1-AVG") == "O1"

    def test_bio_prefix(self):
        assert normalise_channel_name("BIO ECG") == "ECG"


class TestDetectChannels:
    """Tests for detect_channels."""

    def test_filters_non_eeg(self):
        raw_names = ["EEG FP1-REF", "EEG C3-REF", "EMG1", "ECG", "PHOTIC"]
        result = detect_channels(raw_names)
        assert result == ["FP1", "C3"]

    def test_keeps_eeg_with_ref_suffix(self):
        """Regression test: -REF suffix should not cause false positive
        non-EEG detection (the 'REF' pattern)."""
        raw_names = ["EEG FP1-REF", "EEG FP2-REF", "EEG C3-REF"]
        result = detect_channels(raw_names)
        assert result == ["FP1", "FP2", "C3"]

    def test_empty_input(self):
        assert detect_channels([]) == []

    def test_all_non_eeg(self):
        assert detect_channels(["EMG", "ECG", "EOG"]) == []

    def test_deduplication_not_applied(self):
        # If two raw names normalise to the same thing, both appear
        raw_names = ["EEG FP1-REF", "EEG FP1-LE"]
        result = detect_channels(raw_names)
        assert result == ["FP1", "FP1"]

    def test_annotations_filtered(self):
        raw_names = ["FP1", "EDF Annotations"]
        result = detect_channels(raw_names)
        assert result == ["FP1"]

    def test_status_filtered(self):
        raw_names = ["FP1", "Status"]
        result = detect_channels(raw_names)
        assert result == ["FP1"]


class TestMapChannelsToLabram:
    """Tests for LaBraM channel mapping."""

    def test_known_channels(self):
        indices = map_channels_to_labram(["FP1", "FP2", "CZ"])
        # FP1 is at index 0, +1 for CLS = 1
        # FP2 is at index 2, +1 for CLS = 3
        # CZ is at index 41, +1 for CLS = 42
        assert indices == [1, 3, 42]

    def test_unknown_channels_skipped(self):
        indices = map_channels_to_labram(["FP1", "UNKNOWN_CHANNEL", "FP2"])
        assert indices == [1, 3]

    def test_empty_input(self):
        assert map_channels_to_labram([]) == []

    def test_all_unknown(self):
        assert map_channels_to_labram(["X1", "X2"]) == []


class TestGetLabramMatchedChannels:
    """Tests for get_labram_matched_channels."""

    def test_returns_matched_pairs(self):
        names, indices = get_labram_matched_channels(["FP1", "UNKNOWN", "CZ"])
        assert names == ["FP1", "CZ"]
        assert indices == [1, 42]

    def test_empty_input(self):
        names, indices = get_labram_matched_channels([])
        assert names == []
        assert indices == []

    def test_preserves_order(self):
        names, indices = get_labram_matched_channels(["CZ", "FP1"])
        assert names == ["CZ", "FP1"]
        # CZ=42, FP1=1
        assert indices == [42, 1]
