"""Tests for channel name normalisation and mapping."""

import pytest

from shared.channel_maps import (
    BENDR_STANDARD_1020,
    LABRAM_STANDARD_1020,
    detect_channels,
    get_bendr_matched_channels,
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


class TestGetBendrMatchedChannels:
    """Tests for get_bendr_matched_channels."""

    def test_direct_old_style_match(self):
        names, indices = get_bendr_matched_channels(["T3", "T4", "FP1"])
        # Returns in BENDR canonical order: FP1 (idx 0), T3 (idx 7), T4 (idx 11)
        assert names == ["FP1", "T3", "T4"]

    def test_modern_to_old_reverse_alias(self):
        # T7 -> T3, T8 -> T4, P7 -> T5, P8 -> T6
        names, indices = get_bendr_matched_channels(["T7", "T8", "P7", "P8"])
        # Canonical order: T3(7), T4(11), T5(12), T6(16)
        assert names == ["T3", "T4", "T5", "T6"]

    def test_mixed_old_and_modern_names(self):
        names, indices = get_bendr_matched_channels(["T3", "T8", "FP1"])
        assert "FP1" in names
        assert "T3" in names
        assert "T4" in names  # T8 maps to T4 in BENDR

    def test_non_matching_channels_excluded(self):
        names, indices = get_bendr_matched_channels(["FP1", "UNKNOWN", "AF7"])
        assert names == ["FP1"]
        assert len(indices) == 1

    def test_empty_input(self):
        names, indices = get_bendr_matched_channels([])
        assert names == []
        assert indices == []

    def test_returns_bendr_canonical_order(self):
        # Give channels in reverse order - should come back in BENDR order
        reversed_ch = list(reversed(BENDR_STANDARD_1020))
        names, indices = get_bendr_matched_channels(reversed_ch)
        assert names == list(BENDR_STANDARD_1020)

    def test_indices_point_to_correct_source_data(self):
        # Simulate EDF with channels in a specific order
        edf_channels = ["CZ", "FP1", "O2", "T3"]
        names, indices = get_bendr_matched_channels(edf_channels)
        # BENDR order: FP1(0), T3(7), CZ(9), O2(18)
        assert names == ["FP1", "T3", "CZ", "O2"]
        # Indices into edf_channels: FP1=1, T3=3, CZ=0, O2=2
        assert indices == [1, 3, 0, 2]

    def test_all_19_channels_matched(self):
        names, indices = get_bendr_matched_channels(list(BENDR_STANDARD_1020))
        assert len(names) == 19
        assert names == list(BENDR_STANDARD_1020)

    def test_returned_names_use_old_style(self):
        # Even when input uses modern names, output should use BENDR's old-style
        names, _ = get_bendr_matched_channels(["T7"])
        assert names == ["T3"]
