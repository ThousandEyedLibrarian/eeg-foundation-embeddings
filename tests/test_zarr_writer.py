"""Tests for Zarr output writer."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from shared.zarr_writer import write_embeddings


@pytest.fixture
def output_dir():
    """Create a temporary directory for test output."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_embeddings():
    """Sample embeddings and padding mask."""
    rng = np.random.default_rng(42)
    n_windows = 10
    emb_dim = 200
    embeddings = rng.standard_normal((n_windows, emb_dim)).astype(np.float32)
    padding_mask = np.ones(n_windows, dtype=bool)
    padding_mask[-2:] = False  # Last 2 windows are padded
    return embeddings, padding_mask


class TestWriteEmbeddings:
    """Tests for write_embeddings."""

    def test_basic_write(self, output_dir, sample_embeddings):
        embeddings, mask = sample_embeddings
        metadata = {"model_name": "test", "edf_filename": "test.edf"}

        path = write_embeddings(
            output_path=output_dir / "test.zarr",
            embeddings=embeddings,
            padding_mask=mask,
            metadata=metadata,
        )

        assert path.exists()
        assert path.suffix == ".zarr"

    def test_zarr_structure(self, output_dir, sample_embeddings):
        embeddings, mask = sample_embeddings
        path = write_embeddings(
            output_path=output_dir / "test.zarr",
            embeddings=embeddings,
            padding_mask=mask,
            metadata={"model_name": "test"},
        )

        root = zarr.open_group(str(path), mode="r")
        assert "embeddings" in root
        assert "padding_mask" in root
        assert "mean_embedding" in root

    def test_embeddings_shape_and_dtype(self, output_dir, sample_embeddings):
        embeddings, mask = sample_embeddings
        path = write_embeddings(
            output_path=output_dir / "test.zarr",
            embeddings=embeddings,
            padding_mask=mask,
            metadata={},
        )

        root = zarr.open_group(str(path), mode="r")
        stored_emb = np.array(root["embeddings"])
        assert stored_emb.shape == (10, 200)
        assert stored_emb.dtype == np.float32

    def test_padding_mask_values(self, output_dir, sample_embeddings):
        embeddings, mask = sample_embeddings
        path = write_embeddings(
            output_path=output_dir / "test.zarr",
            embeddings=embeddings,
            padding_mask=mask,
            metadata={},
        )

        root = zarr.open_group(str(path), mode="r")
        stored_mask = np.array(root["padding_mask"])
        np.testing.assert_array_equal(stored_mask, mask)

    def test_mean_embedding_excludes_padded(self, output_dir, sample_embeddings):
        embeddings, mask = sample_embeddings
        path = write_embeddings(
            output_path=output_dir / "test.zarr",
            embeddings=embeddings,
            padding_mask=mask,
            metadata={},
        )

        root = zarr.open_group(str(path), mode="r")
        stored_mean = np.array(root["mean_embedding"])

        # Expected: mean of first 8 windows (mask[-2:] = False)
        expected_mean = embeddings[:8].mean(axis=0)
        np.testing.assert_allclose(stored_mean, expected_mean, atol=1e-6)

    def test_metadata_stored(self, output_dir, sample_embeddings):
        embeddings, mask = sample_embeddings
        metadata = {
            "model_name": "labram",
            "channel_names": ["FP1", "C3"],
            "original_duration_sec": 300.0,
        }
        path = write_embeddings(
            output_path=output_dir / "test.zarr",
            embeddings=embeddings,
            padding_mask=mask,
            metadata=metadata,
        )

        root = zarr.open_group(str(path), mode="r")
        attrs = dict(root.attrs)
        assert attrs["model_name"] == "labram"
        assert attrs["channel_names"] == ["FP1", "C3"]
        assert attrs["original_duration_sec"] == 300.0
        assert attrs["n_windows"] == 10
        assert attrs["n_valid_windows"] == 8
        assert attrs["embedding_dim"] == 200
        assert "created" in attrs
        assert "software_version" in attrs

    def test_auto_adds_zarr_suffix(self, output_dir, sample_embeddings):
        embeddings, mask = sample_embeddings
        path = write_embeddings(
            output_path=output_dir / "test",
            embeddings=embeddings,
            padding_mask=mask,
            metadata={},
        )
        assert path.suffix == ".zarr"

    def test_no_per_window_option(self, output_dir, sample_embeddings):
        embeddings, mask = sample_embeddings
        path = write_embeddings(
            output_path=output_dir / "test.zarr",
            embeddings=embeddings,
            padding_mask=mask,
            metadata={},
            store_per_window=False,
            store_mean_pooled=True,
        )

        root = zarr.open_group(str(path), mode="r")
        assert "embeddings" not in root
        assert "padding_mask" not in root
        assert "mean_embedding" in root

    def test_numpy_types_in_metadata(self, output_dir, sample_embeddings):
        """Numpy types in metadata should be converted to native Python."""
        embeddings, mask = sample_embeddings
        metadata = {
            "n_something": np.int64(42),
            "some_float": np.float32(3.14),
            "a_bool": np.bool_(True),
            "an_array": np.array([1, 2, 3]),
        }
        path = write_embeddings(
            output_path=output_dir / "test.zarr",
            embeddings=embeddings,
            padding_mask=mask,
            metadata=metadata,
        )

        root = zarr.open_group(str(path), mode="r")
        attrs = dict(root.attrs)
        assert isinstance(attrs["n_something"], int)
        assert isinstance(attrs["some_float"], float)
        assert isinstance(attrs["a_bool"], bool)
        assert attrs["a_bool"] is True  # Not 1
        assert attrs["an_array"] == [1, 2, 3]
