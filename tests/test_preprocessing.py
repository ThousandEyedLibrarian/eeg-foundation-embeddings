"""Tests for EEG preprocessing pipeline."""

import numpy as np
import pytest

from shared.preprocessing import (
    PreprocessingConfig,
    apply_filters,
    create_windows,
    extract_time_window,
    preprocess,
)


@pytest.fixture
def sample_data():
    """Create synthetic EEG data: 3 channels, 20 seconds at 200 Hz."""
    rng = np.random.default_rng(42)
    n_channels = 3
    sr = 200
    duration_sec = 20
    n_samples = sr * duration_sec
    data = rng.standard_normal((n_channels, n_samples)).astype(np.float64)
    return data, sr


class TestApplyFilters:
    """Tests for bandpass and notch filtering."""

    def test_output_shape_matches_input(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(bandpass_low=0.1, bandpass_high=75.0, notch_freq=50.0)
        filtered = apply_filters(data, sr, config)
        assert filtered.shape == data.shape

    def test_no_filter_when_disabled(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(bandpass_low=0.0, bandpass_high=0.0, notch_freq=0.0)
        filtered = apply_filters(data, sr, config)
        np.testing.assert_array_equal(filtered, data)

    def test_bandpass_clamps_high_freq(self, sample_data):
        data, sr = sample_data
        # bandpass_high > Nyquist should be clamped, not crash
        config = PreprocessingConfig(bandpass_low=0.1, bandpass_high=150.0)
        filtered = apply_filters(data, sr, config)
        assert filtered.shape == data.shape

    def test_notch_skipped_above_nyquist(self, sample_data):
        data, sr = sample_data
        # Notch at frequency above Nyquist should be silently skipped
        config = PreprocessingConfig(notch_freq=200.0)
        filtered = apply_filters(data, sr, config)
        assert filtered.shape == data.shape


class TestExtractTimeWindow:
    """Tests for time window extraction (skip/trim/reject)."""

    def test_no_skip_no_trim(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig()
        trimmed, info = extract_time_window(data, sr, config)
        assert trimmed.shape == data.shape
        assert info["used_duration_sec"] == pytest.approx(20.0)

    def test_skip_start(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(skip_start_sec=5.0)
        trimmed, info = extract_time_window(data, sr, config)
        expected_samples = 15 * sr  # 20s - 5s = 15s
        assert trimmed.shape[-1] == expected_samples
        assert info["used_start_sec"] == pytest.approx(5.0)

    def test_max_duration(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(max_duration_sec=10.0)
        trimmed, info = extract_time_window(data, sr, config)
        assert trimmed.shape[-1] == 10 * sr
        assert info["used_duration_sec"] == pytest.approx(10.0)

    def test_skip_plus_max_duration(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(skip_start_sec=5.0, max_duration_sec=10.0)
        trimmed, info = extract_time_window(data, sr, config)
        assert trimmed.shape[-1] == 10 * sr
        assert info["used_start_sec"] == pytest.approx(5.0)
        assert info["used_end_sec"] == pytest.approx(15.0)

    def test_min_duration_reject(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(min_duration_sec=30.0)
        with pytest.raises(ValueError, match="shorter than minimum"):
            extract_time_window(data, sr, config)

    def test_skip_exceeds_duration(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(skip_start_sec=25.0)
        with pytest.raises(ValueError, match="exceeds recording duration"):
            extract_time_window(data, sr, config)


class TestCreateWindows:
    """Tests for windowing with padding."""

    def test_exact_windows(self):
        """Data length is exact multiple of window size."""
        data = np.ones((2, 2000))  # 2 channels, 10 seconds at 200 Hz
        config = PreprocessingConfig(window_size_sec=5.0)
        windows, mask, timestamps = create_windows(data, 200, config)
        assert windows.shape == (2, 2, 1000)  # 2 windows, 2 channels, 1000 samples
        assert mask.all()
        assert timestamps == [0.0, 5.0]

    def test_partial_last_window_marked(self):
        """Data with partial last window should be zero-padded."""
        # 2.5 windows worth of data (12.5 sec at 200 Hz)
        data = np.ones((1, 2500))
        config = PreprocessingConfig(window_size_sec=10.0)
        windows, mask, timestamps = create_windows(data, 200, config)
        assert windows.shape[0] == 2  # 1 full + 1 partial
        # Partial window has 500/2000 = 25% real data, < 50%, so marked invalid
        assert mask[0] is np.bool_(True)
        assert mask[1] is np.bool_(False)

    def test_short_data_single_padded_window(self):
        """Data shorter than one window."""
        data = np.ones((1, 100))  # 0.5 sec
        config = PreprocessingConfig(window_size_sec=10.0)
        windows, mask, timestamps = create_windows(data, 200, config)
        assert windows.shape == (1, 1, 2000)
        # 100/2000 = 5% real data, marked invalid
        assert not mask[0]

    def test_overlap_windows(self):
        """Windows with overlap produce more windows."""
        data = np.ones((1, 4000))  # 20 sec
        config = PreprocessingConfig(window_size_sec=10.0, window_overlap_sec=5.0)
        windows, mask, timestamps = create_windows(data, 200, config)
        # Step = 10-5 = 5 sec. Windows: 0-10, 5-15, 10-20 = 3 windows
        assert windows.shape[0] == 3
        assert timestamps == [0.0, 5.0, 10.0]

    def test_overlap_exceeds_window_raises(self):
        data = np.ones((1, 4000))
        config = PreprocessingConfig(window_size_sec=10.0, window_overlap_sec=10.0)
        with pytest.raises(ValueError, match="must be less than"):
            create_windows(data, 200, config)


class TestPreprocess:
    """Tests for the full preprocessing pipeline."""

    def test_full_pipeline(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(
            bandpass_low=0.1,
            bandpass_high=75.0,
            notch_freq=50.0,
            window_size_sec=10.0,
        )
        windows, mask, metadata = preprocess(data, sr, config)
        assert windows.shape == (2, 3, 2000)  # 20s / 10s = 2 windows
        assert mask.shape == (2,)
        assert metadata["n_windows"] == 2
        assert metadata["n_valid_windows"] == 2

    def test_zscore_normalisation(self, sample_data):
        data, sr = sample_data
        config = PreprocessingConfig(
            normalisation="zscore",
            bandpass_low=0.0,
            bandpass_high=0.0,
            notch_freq=0.0,
            window_size_sec=20.0,
        )
        windows, _, _ = preprocess(data, sr, config)
        # After zscore, each channel should have approx mean=0, std=1
        for ch in range(windows.shape[1]):
            ch_data = windows[0, ch]
            assert abs(np.mean(ch_data)) < 0.1
            assert abs(np.std(ch_data) - 1.0) < 0.1

    def test_scale_factor_applied(self):
        data = np.ones((1, 2000)) * 100.0
        config = PreprocessingConfig(
            scale_factor=0.01,
            bandpass_low=0.0,
            bandpass_high=0.0,
            notch_freq=0.0,
            window_size_sec=10.0,
        )
        windows, _, _ = preprocess(data, 200, config)
        np.testing.assert_allclose(windows[0, 0], 1.0, atol=1e-5)

    def test_clipping(self):
        """Values beyond clip_std should be clamped."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1, 2000))
        # Insert extreme outlier
        data[0, 100] = 100.0
        config = PreprocessingConfig(
            clip_std=3.0,
            bandpass_low=0.0,
            bandpass_high=0.0,
            notch_freq=0.0,
            window_size_sec=10.0,
        )
        windows, _, _ = preprocess(data, 200, config)
        # The extreme value should be clipped
        assert np.max(np.abs(windows)) < 100.0
