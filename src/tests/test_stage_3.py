"""Tests for Stage 3 — Marine Debris Detection."""

import importlib

import numpy as np
import pytest

detect = importlib.import_module("pipeline.03_detect")


class TestTTAAugmentations:
    """Tests for TTA augmentation / de-augmentation roundtrips."""

    def test_augmentation_roundtrip(self):
        """Applying then reversing an augmentation should recover original."""
        patch = np.random.rand(11, 64, 64).astype(np.float32)

        for aug_type in ["hflip", "vflip", "rot90", "rot180", "rot270"]:
            augmented = detect._apply_augmentation(patch, aug_type)
            reversed_ = detect._reverse_augmentation(augmented, aug_type)
            np.testing.assert_allclose(
                reversed_, patch, atol=1e-6,
                err_msg=f"Roundtrip failed for {aug_type}",
            )


class TestStitching:
    """Tests for probability map stitching."""

    def test_stitch_no_overlap(self):
        """Stitching non-overlapping patches should recover original values."""
        pred1 = np.ones((15, 32, 32), dtype=np.float32) * 0.5
        pred2 = np.ones((15, 32, 32), dtype=np.float32) * 0.8

        patch_index = {
            "patch_0000": {"row_start": 0, "col_start": 0,
                          "actual_h": 32, "actual_w": 32},
            "patch_0001": {"row_start": 0, "col_start": 32,
                          "actual_h": 32, "actual_w": 32},
        }

        patch_ids = ["patch_0000", "patch_0001"]
        result = detect.stitch_patches([pred1, pred2], patch_ids, patch_index, (32, 64))
        assert result.shape == (15, 32, 64)
        np.testing.assert_allclose(result[:, 0, 0], 0.5, atol=1e-3)
        np.testing.assert_allclose(result[:, 0, 32], 0.8, atol=1e-3)


class TestExtractClusters:
    """Tests for debris cluster extraction."""

    def test_no_debris_returns_empty(self):
        """Empty debris mask should return empty GeoDataFrame."""
        from rasterio.transform import Affine

        mask = np.zeros((100, 100), dtype=bool)
        prob = np.zeros((100, 100), dtype=np.float32)
        transform = Affine(10, 0, 0, 0, -10, 0)

        gdf = detect.extract_clusters(mask, prob, transform, "EPSG:4326")
        assert len(gdf) == 0

    def test_single_cluster(self):
        """A single connected component should produce one cluster."""
        from rasterio.transform import Affine

        mask = np.zeros((100, 100), dtype=bool)
        mask[40:50, 40:50] = True

        prob = np.zeros((100, 100), dtype=np.float32)
        prob[40:50, 40:50] = 0.8

        transform = Affine(10, 0, 0, 0, -10, 0)

        gdf = detect.extract_clusters(mask, prob, transform, "EPSG:4326",
                                      min_area_m2=100)
        assert len(gdf) >= 1
        assert gdf.iloc[0]["mean_confidence"] == pytest.approx(0.8)
