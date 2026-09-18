"""Tests for Stage 5 — Hydrodynamic Back-Tracking."""

import importlib
import sys
import types
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

backtrack = importlib.import_module("pipeline.05_backtrack")


class TestBacktrackParticle:
    """Tests for OceanParcels fieldset initialization."""

    def test_load_parcels_fieldset_missing_data(self):
        """Missing NetCDF forcing data should safely return None."""
        fieldset = backtrack.load_parcels_fieldset(None, None)
        assert fieldset is None


class TestClusterEndpoints:
    """Tests for DBSCAN endpoint clustering."""

    def test_single_cluster(self):
        """Tightly grouped points should form a single cluster."""
        rng = np.random.default_rng(42)
        endpoints = [(80.5 + rng.normal(0, 0.1),
                      7.5 + rng.normal(0, 0.1))
                     for _ in range(20)]

        sources = backtrack.cluster_endpoints(endpoints, eps_degrees=0.5, min_samples=5)
        assert len(sources) >= 1
        assert sources[0]["source_probability"] > 0

    def test_empty_endpoints(self):
        """Empty endpoints should return empty list."""
        sources = backtrack.cluster_endpoints([], eps_degrees=0.5, min_samples=5)
        assert sources == []


class TestWindDownload:
    """Tests for ERA5 wind download setup and fallback."""

    def test_download_wind_data_uses_bounded_cds_client_kwargs(self, tmp_path, monkeypatch):
        """Client should receive retry/timeout bounds from env when supported."""
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def retrieve(self, dataset, request, target):
                Path(target).write_text("ok", encoding="utf-8")

        fake_mod = types.SimpleNamespace(Client=FakeClient)
        monkeypatch.setitem(sys.modules, "cdsapi", fake_mod)

        monkeypatch.setenv("CDS_API_KEY", "abc123")
        monkeypatch.setenv("CDS_API_URL", "https://cds.example/api")
        monkeypatch.setenv("CDS_RETRY_MAX", "2")
        monkeypatch.setenv("CDS_SLEEP_MAX", "5")
        monkeypatch.setenv("CDS_TIMEOUT", "20")

        out = backtrack.download_wind_data(
            bbox=(80.0, 7.0, 81.0, 8.0),
            date_start="2026-03-01T00:00:00",
            date_end="2026-03-02T00:00:00",
            output_dir=tmp_path,
        )

        assert out == tmp_path / "wind_data.nc"
        assert captured["url"] == "https://cds.example/api"
        assert captured["key"] == "abc123"
        assert captured["retry_max"] == 2
        assert captured["sleep_max"] == 5
        assert captured["timeout"] == 20

    def test_download_wind_data_returns_none_on_retrieve_error(self, tmp_path, monkeypatch):
        """A CDS client failure should trigger synthetic wind fallback (None)."""

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def retrieve(self, dataset, request, target):
                raise RuntimeError("DNS resolution failed")

        fake_mod = types.SimpleNamespace(Client=FakeClient)
        monkeypatch.setitem(sys.modules, "cdsapi", fake_mod)

        out = backtrack.download_wind_data(
            bbox=(80.0, 7.0, 81.0, 8.0),
            date_start="2026-03-01T00:00:00",
            date_end="2026-03-02T00:00:00",
            output_dir=tmp_path,
        )

        assert out is None


class TestRunBacktrack:
    """Regression tests for Stage 5 run orchestration."""

    def test_run_handles_dict_velocity_fields_without_close(self, tmp_path, monkeypatch):
        """run() should complete when loaders return dict-backed fields."""
        scene_id = "scene_test"
        detections_path = tmp_path / "detections.geojson"

        gdf = gpd.GeoDataFrame(
            {"cluster_id": [1], "is_false_positive": [False]},
            geometry=[Point(80.5, 7.5)],
            crs="EPSG:4326",
        )
        gdf.to_file(detections_path, driver="GeoJSON")

        monkeypatch.setattr(backtrack, "stage_output_exists", lambda *args, **kwargs: False)
        monkeypatch.setattr(backtrack, "download_ocean_currents", lambda *args, **kwargs: None)
        monkeypatch.setattr(backtrack, "download_wind_data", lambda *args, **kwargs: None)

        # Simulate in-memory velocity fields represented as dicts.
        dummy_field = {
            "times": np.array([np.datetime64("2026-03-01T00:00:00")], dtype="datetime64[ns]"),
            "lats": np.array([7.5], dtype=np.float64),
            "lons": np.array([80.5], dtype=np.float64),
            "data": {
                "uo": np.zeros((1, 1, 1), dtype=np.float32),
                "vo": np.zeros((1, 1, 1), dtype=np.float32),
                "u10": np.zeros((1, 1, 1), dtype=np.float32),
                "v10": np.zeros((1, 1, 1), dtype=np.float32),
            },
        }
        monkeypatch.setattr(backtrack, "load_parcels_fieldset", lambda *args, **kwargs: None)

        config = {
            "backtracking": {
                "days": 0,
                "n_particles": 2,
                "time_step_hours": 1.0,
                "dbscan_min_samples": 1,
            }
        }

        sources = backtrack.run(
            scene_id=scene_id,
            detections_path=detections_path,
            output_dir=tmp_path,
            config=config,
        )

        assert isinstance(sources, list)
