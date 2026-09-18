"""Tests for Stage 7 — Report Generation."""

import importlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

report = importlib.import_module("pipeline.07_report")


class TestCSVGeneration:
    """Tests for CSV export."""

    def test_empty_detections_csv(self, tmp_path):
        """Empty detections should produce a valid CSV with headers."""
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        out = report.generate_csv(gdf, [], "test_scene", tmp_path / "test.csv")

        assert out.exists()
        df = pd.read_csv(out)
        assert "cluster_id" in df.columns
        assert len(df) == 0

    def test_csv_with_detections(self, tmp_path):
        """Detections should be exported to CSV correctly."""
        gdf = gpd.GeoDataFrame({
            "geometry": [Point(80.5, 7.5)],
            "cluster_id": [0],
            "centroid_lat": [7.5],
            "centroid_lon": [80.5],
            "area_m2": [500],
            "polymer_type": ["PE/PP"],
            "mean_confidence": [0.75],
            "detection_date": ["2024-01-15"],
        }, crs="EPSG:4326")

        attribution = [{
            "debris_cluster_id": 0,
            "source_type": "fishing",
            "location_name": "Sri Lankan Coast",
            "country": "Sri Lanka",
            "attribution_score": 0.72,
        }]

        out = report.generate_csv(gdf, attribution, "test_scene", tmp_path / "test.csv")
        df = pd.read_csv(out)

        assert len(df) == 1
        assert df.iloc[0]["polymer_type"] == "PE/PP"
        assert df.iloc[0]["top_source_type"] == "fishing"


class TestGeoJSONSummary:
    """Tests for GeoJSON summary generation."""

    def test_empty_geojson(self, tmp_path):
        """Empty detections should produce a valid GeoJSON."""
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        out = report.generate_geojson_summary(gdf, [], tmp_path / "test.geojson")

        assert out.exists()
        result_gdf = gpd.read_file(out)
        assert len(result_gdf) == 0


class TestTerminalSummary:
    """Tests for terminal output."""

    def test_print_no_crash_empty(self):
        """Terminal summary should not crash with empty data."""
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        report.print_terminal_summary("test_scene", gdf, [])

    def test_print_no_crash_with_data(self):
        """Terminal summary should not crash with real data."""
        gdf = gpd.GeoDataFrame({
            "geometry": [Point(80.5, 7.5)],
            "cluster_id": [0],
            "area_m2": [500],
            "mean_confidence": [0.75],
            "polymer_type": ["PE/PP"],
            "centroid_lon": [80.5],
            "centroid_lat": [7.5],
        }, crs="EPSG:4326")

        attribution = [{
            "source_rank": 1,
            "source_type": "fishing",
            "attribution_score": 0.72,
            "location_name": "Test Location",
            "explanation": "Test explanation text.",
        }]

        report.print_terminal_summary("test_scene", gdf, attribution)
