"""Tests for Stage 2 — Preprocessing."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

preprocess = importlib.import_module("pipeline.02_preprocess")


class TestPadTo11Bands:
    """Tests for the 8→11 band padding logic."""


class TestNormalize:
    """Tests for scene normalization."""

    def test_normalize_clips_and_zscores(self):
        """Normalization should clip values and apply z-score."""
        image = np.full((11, 32, 32), 0.05, dtype=np.float32)
        normalized, nodata = preprocess.normalize_scene(image)

        assert normalized.shape == (11, 32, 32)
        assert not np.any(np.isnan(normalized))
        assert np.all(normalized >= -5.0)
        assert np.all(normalized <= 5.0)

    def test_nodata_detection(self):
        """All-zero pixels should be detected as nodata."""
        image = np.ones((11, 32, 32), dtype=np.float32) * 0.05
        image[:, :, 0] = 0.0

        _, nodata = preprocess.normalize_scene(image)

        assert nodata.shape == (32, 32)
        assert np.all(nodata[:, 0])
        assert not np.any(nodata[:, 1])


class TestTiling:
    """Tests for scene tiling."""

    def test_tile_produces_correct_patches(self):
        """Tiling should produce patches of the correct size."""
        image = np.random.rand(11, 300, 300).astype(np.float32)
        patches, infos = preprocess.tile_scene(image, patch_size=256, overlap=32)

        assert len(patches) > 0
        assert all(p.shape == (11, 256, 256) for p in patches)
        assert len(patches) == len(infos)

    def test_tile_small_image(self):
        """An image smaller than patch size should produce exactly 1 patch."""
        image = np.random.rand(11, 100, 100).astype(np.float32)
        patches, infos = preprocess.tile_scene(image, patch_size=256, overlap=32)

        assert len(patches) == 1
        assert patches[0].shape == (11, 256, 256)
        assert infos[0]["actual_h"] == 100
        assert infos[0]["actual_w"] == 100
