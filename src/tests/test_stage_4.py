"""Tests for Stage 4 — Polymer Type Classification."""

import importlib

import numpy as np
import pytest

polymer = importlib.import_module("pipeline.04_polymer")


class TestSpectralIndices:
    """Tests for spectral index computation."""

    def test_plastic_index_water(self):
        """Water-like spectrum should have low PI."""
        spectrum = np.array([0.01, 0.02, 0.03, 0.02, 0.01,
                            0.01, 0.01, 0.01, 0.01, 0.005, 0.002],
                           dtype=np.float32)
        indices = polymer.compute_spectral_indices(spectrum)
        assert "pi" in indices
        assert "sr" in indices
        assert "nsi" in indices
        assert "fdi" in indices

    def test_high_nir_gives_positive_pi(self):
        """High NIR relative to Red should give positive PI."""
        spectrum = np.zeros(11, dtype=np.float32)
        spectrum[3] = 0.02  # Red low
        spectrum[7] = 0.10  # NIR high

        indices = polymer.compute_spectral_indices(spectrum)
        assert indices["pi"] > 0


class TestPolymerClassification:
    """Tests for the rule-based classifier."""

    def test_pe_pp_classification(self):
        """High PI + low SR should classify as PE/PP."""
        ptype, is_fp = polymer.classify_polymer({
            "pi": 0.15, "sr": 0.2, "nsi": -0.1, "fdi": 0.1,
        })
        assert "PE/PP" in ptype
        assert not is_fp

    def test_organic_flagged_as_fp(self):
        """High NSI should flag as organic / false positive."""
        ptype, is_fp = polymer.classify_polymer({
            "pi": 0.05, "sr": 0.3, "nsi": 0.4, "fdi": 0.01,
        })
        assert "Organic" in ptype
        assert is_fp

    def test_pet_nylon(self):
        """High PI + high SR should classify as PET/Nylon."""
        ptype, is_fp = polymer.classify_polymer({
            "pi": 0.15, "sr": 0.6, "nsi": -0.05, "fdi": 0.08,
        })
        assert "PET" in ptype
        assert not is_fp
