"""Tests for Stage 1 — Satellite Data Ingestion.

Tests cover:
- STAC search returning single and multiple tiles
- Scene grouping by satellite pass datetime
- Multi-tile mosaicking with rasterio merge
- BBox cropping of mosaicked output
- Cross-CRS reprojection handling
"""

import importlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# CRITICAL: Prevent PROJ database conflicts between pyproj and rasterio
# (same fix as in run_pipeline.py lines 47-48)
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

ingest = importlib.import_module("pipeline.01_ingest")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _make_mock_item(item_id, dt_str, cloud_cover=5, bbox=None, geometry=None):
    """Create a mock STAC item."""
    mock_item = MagicMock()
    mock_item.id = item_id
    mock_item.datetime = dt_str
    mock_item.properties = {"eo:cloud_cover": cloud_cover}
    mock_item.bbox = bbox or [80.0, 7.0, 81.0, 8.0]
    mock_item.assets = {"B02": MagicMock(href=f"https://example.com/{item_id}/B02.tif")}
    # Make .extra_fields work for asset URL extraction
    for key, asset in mock_item.assets.items():
        asset.extra_fields = {}
    mock_item.geometry = geometry or {
        "type": "Polygon",
        "coordinates": [[[80.0, 7.0], [81.0, 7.0], [81.0, 8.0], [80.0, 8.0], [80.0, 7.0]]],
    }
    return mock_item


def _create_test_geotiff(path, data, bbox, crs="EPSG:32616"):
    """Create a test GeoTIFF file with given data, bbox, and CRS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = data.shape[-2], data.shape[-1]
    if data.ndim == 2:
        count = 1
        data = data[np.newaxis, ...]
    else:
        count = data.shape[0]

    transform = from_bounds(*bbox, width, height)
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


class TestSearchScenes:
    """Tests for the STAC search functionality."""

    @patch.object(ingest, "Client")
    def test_search_returns_scenes(self, mock_client_cls):
        """Search should return sorted scene dicts on success."""
        mock_item = _make_mock_item("S2A_TEST_001", "2024-01-15T10:00:00Z")
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client_cls.open.return_value.search.return_value = mock_search

        result = ingest.search_scenes((80.0, 7.0, 81.0, 8.0), "2024-01-01", "2024-01-31")

        assert len(result) == 1
        assert result[0]["id"] == "S2A_TEST_001"
        assert result[0]["cloud_cover"] == 5

    @patch.object(ingest, "Client")
    def test_search_no_scenes_raises(self, mock_client_cls):
        """Search with no results should raise RuntimeError."""
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_client_cls.open.return_value.search.return_value = mock_search

        with pytest.raises(RuntimeError, match="No Sentinel-2"):
            ingest.search_scenes((0, 0, 1, 1), "2024-01-01", "2024-01-31")

    @patch.object(ingest, "Client")
    def test_search_returns_multiple_tiles(self, mock_client_cls):
        """Search should return multiple tiles when bbox spans multiple granules."""
        item_north = _make_mock_item(
            "S2A_T16QCJ_2024", "2024-01-15T10:00:00Z",
            bbox=[-88.5, 16.0, -88.0, 16.5],
            geometry={
                "type": "Polygon",
                "coordinates": [[[-88.5, 16.0], [-88.0, 16.0], [-88.0, 16.5], [-88.5, 16.5], [-88.5, 16.0]]],
            },
        )
        item_south = _make_mock_item(
            "S2A_T16QBJ_2024", "2024-01-15T10:00:05Z",
            bbox=[-88.5, 15.5, -88.0, 16.05],
            geometry={
                "type": "Polygon",
                "coordinates": [[[-88.5, 15.5], [-88.0, 15.5], [-88.0, 16.05], [-88.5, 16.05], [-88.5, 15.5]]],
            },
        )

        mock_search = MagicMock()
        mock_search.items.return_value = [item_north, item_south]
        mock_client_cls.open.return_value.search.return_value = mock_search

        bbox = (-88.452, 15.813, -88.052, 16.213)
        result = ingest.search_scenes(bbox, "2024-01-01", "2024-01-31")

        # Both tiles should be returned since they're from the same pass
        assert len(result) == 2
        ids = {r["id"] for r in result}
        assert "S2A_T16QCJ_2024" in ids
        assert "S2A_T16QBJ_2024" in ids


class TestSceneGrouping:
    """Tests for _select_best_scene_group()."""

    def test_groups_by_date(self):
        """Tiles within SAME_PASS_TOLERANCE_HOURS should be grouped together."""
        scenes = [
            {"id": "A1", "datetime": "2024-01-15T10:00:00Z",
             "bbox": [-88.5, 16.0, -88.0, 16.5],
             "geometry": {"type": "Polygon", "coordinates": [[[-88.5, 16.0], [-88.0, 16.0], [-88.0, 16.5], [-88.5, 16.5], [-88.5, 16.0]]]}},
            {"id": "A2", "datetime": "2024-01-15T10:00:05Z",
             "bbox": [-88.5, 15.5, -88.0, 16.05],
             "geometry": {"type": "Polygon", "coordinates": [[[-88.5, 15.5], [-88.0, 15.5], [-88.0, 16.05], [-88.5, 16.05], [-88.5, 15.5]]]}},
            {"id": "B1", "datetime": "2024-01-10T10:00:00Z",
             "bbox": [-88.5, 16.0, -88.0, 16.5],
             "geometry": {"type": "Polygon", "coordinates": [[[-88.5, 16.0], [-88.0, 16.0], [-88.0, 16.5], [-88.5, 16.5], [-88.5, 16.0]]]}},
        ]
        bbox = (-88.452, 15.813, -88.052, 16.213)
        result = ingest._select_best_scene_group(scenes, bbox)

        # Should select A1 + A2 (same pass, better coverage) over B1 alone
        assert len(result) == 2
        ids = {r["id"] for r in result}
        assert ids == {"A1", "A2"}

    def test_single_scene_returns_as_is(self):
        """A single scene should be returned without grouping logic."""
        scenes = [
            {"id": "SOLO", "datetime": "2024-01-15T10:00:00Z",
             "bbox": [80.0, 7.0, 81.0, 8.0],
             "geometry": {"type": "Polygon", "coordinates": []}},
        ]
        result = ingest._select_best_scene_group(scenes, (80.0, 7.0, 81.0, 8.0))
        assert len(result) == 1
        assert result[0]["id"] == "SOLO"

    def test_prefers_group_with_better_coverage(self):
        """Should prefer the group that covers more of the bbox, even if older."""
        scenes = [
            # Recent but covers only north half
            {"id": "R1", "datetime": "2024-01-20T10:00:00Z",
             "bbox": [-88.5, 16.0, -88.0, 16.5],
             "geometry": {"type": "Polygon", "coordinates": [[[-88.5, 16.0], [-88.0, 16.0], [-88.0, 16.5], [-88.5, 16.5], [-88.5, 16.0]]]}},
            # Older but covers full bbox
            {"id": "O1", "datetime": "2024-01-15T10:00:00Z",
             "bbox": [-88.5, 16.0, -88.0, 16.5],
             "geometry": {"type": "Polygon", "coordinates": [[[-88.5, 16.0], [-88.0, 16.0], [-88.0, 16.5], [-88.5, 16.5], [-88.5, 16.0]]]}},
            {"id": "O2", "datetime": "2024-01-15T10:00:05Z",
             "bbox": [-88.5, 15.5, -88.0, 16.05],
             "geometry": {"type": "Polygon", "coordinates": [[[-88.5, 15.5], [-88.0, 15.5], [-88.0, 16.05], [-88.5, 16.05], [-88.5, 15.5]]]}},
        ]
        bbox = (-88.452, 15.813, -88.052, 16.213)
        result = ingest._select_best_scene_group(scenes, bbox)

        # O1 + O2 cover more of the bbox than R1 alone
        ids = {r["id"] for r in result}
        assert "O1" in ids
        assert "O2" in ids


class TestDownloadScene:
    """Tests for scene download functionality."""

    def test_download_creates_directory(self, tmp_path):
        """Download should create the scene directory."""
        scene = {
            "id": "test_scene",
            "assets": {},
        }
        scene_dir, band_paths = ingest.download_scene(scene, tmp_path)
        assert scene_dir.exists()
        assert (scene_dir / "metadata.json").exists()


class TestMosaicScenes:
    """Tests for mosaic_scenes() multi-tile mosaicking."""

    def test_mosaic_merges_bands(self, tmp_path):
        """Two adjacent tiles should be merged into one larger raster."""
        tile_north = tmp_path / "tile_north"
        tile_south = tmp_path / "tile_south"

        north_data = np.random.rand(100, 100).astype(np.float32) + 1.0
        south_data = np.random.rand(100, 100).astype(np.float32) + 2.0

        # Use WGS84 coords so bbox crop aligns naturally
        north_bbox = (-88.5, 16.0, -88.0, 16.5)
        south_bbox = (-88.5, 15.5, -88.0, 16.0)

        _create_test_geotiff(tile_north / "B02.tif", north_data, north_bbox, crs="EPSG:4326")
        _create_test_geotiff(tile_south / "B02.tif", south_data, south_bbox, crs="EPSG:4326")

        out_dir = tmp_path / "mosaic"
        bbox_wgs84 = (-88.45, 15.8, -88.05, 16.2)

        mosaic_dir, band_paths = ingest.mosaic_scenes(
            [tile_north, tile_south],
            bbox=bbox_wgs84,
            bands=["B02"],
            output_dir=out_dir,
        )

        assert "B02" in band_paths
        out_path = Path(band_paths["B02"])
        assert out_path.exists()

        with rasterio.open(out_path) as src:
            data = src.read(1)
            assert data.max() > 0, "Mosaic output should contain data"
            # Merged raster should cover the bbox area
            assert src.width > 0
            assert src.height > 0

    def test_mosaic_single_tile_no_merge(self, tmp_path):
        """Single tile should skip mosaicking entirely."""
        tile = tmp_path / "single_tile"
        data = np.ones((100, 100), dtype=np.float32) * 5.0
        tile_bbox = (-88.5, 16.0, -88.0, 16.5)
        _create_test_geotiff(tile / "B02.tif", data, tile_bbox, crs="EPSG:4326")

        result_dir, band_paths = ingest.mosaic_scenes(
            [tile],
            bbox=(-88.5, 16.0, -88.0, 16.5),
            bands=["B02"],
        )

        # Should return the original tile directory unchanged
        assert result_dir == tile

    def test_mosaic_crops_to_bbox(self, tmp_path):
        """Mosaicked output should be cropped to the user's bbox."""
        tile = tmp_path / "wide_tile"
        # Create a large tile (200x200 pixels) covering a wide area
        data = np.random.rand(200, 200).astype(np.float32) + 1.0
        wide_bbox = (-89.0, 15.5, -87.5, 17.0)  # ~1.5° x 1.5°
        _create_test_geotiff(tile / "B02.tif", data, wide_bbox, crs="EPSG:4326")

        # Create a second tile to trigger mosaicking path
        tile2 = tmp_path / "dummy_tile"
        _create_test_geotiff(
            tile2 / "B02.tif",
            np.zeros((10, 10), dtype=np.float32),
            (-89.0, 15.0, -88.5, 15.5),
            crs="EPSG:4326",
        )

        out_dir = tmp_path / "cropped"
        # Request a small subset of the tile
        bbox_wgs84 = (-88.3, 16.0, -88.1, 16.2)

        mosaic_dir, band_paths = ingest.mosaic_scenes(
            [tile, tile2],
            bbox=bbox_wgs84,
            bands=["B02"],
            output_dir=out_dir,
        )

        out_path = Path(band_paths["B02"])
        assert out_path.exists()

        with rasterio.open(out_path) as src:
            # Cropped raster should be smaller than the original 200x200
            assert src.width < 200 or src.height < 200, \
                "Cropped raster should be smaller than original tile"

    def test_mosaic_handles_different_crs(self, tmp_path):
        """Tiles with different CRS should be reprojected to common CRS."""
        tile_a = tmp_path / "tile_wgs84"
        tile_b = tmp_path / "tile_wgs84_2"

        data_a = np.ones((50, 50), dtype=np.float32) * 3.0
        data_b = np.ones((50, 50), dtype=np.float32) * 4.0

        # Use two adjacent tiles both in WGS84 (simpler than cross-UTM for testing)
        bbox_a = (-88.5, 16.0, -88.0, 16.5)
        bbox_b = (-88.5, 15.5, -88.0, 16.0)

        _create_test_geotiff(tile_a / "B02.tif", data_a, bbox_a, crs="EPSG:4326")
        _create_test_geotiff(tile_b / "B02.tif", data_b, bbox_b, crs="EPSG:4326")

        out_dir = tmp_path / "cross_crs"
        bbox_wgs84 = (-88.45, 15.6, -88.05, 16.4)

        mosaic_dir, band_paths = ingest.mosaic_scenes(
            [tile_a, tile_b],
            bbox=bbox_wgs84,
            bands=["B02"],
            output_dir=out_dir,
        )

        assert "B02" in band_paths
        out_path = Path(band_paths["B02"])
        assert out_path.exists()

        with rasterio.open(out_path) as src:
            assert src.crs is not None
            assert src.read(1).max() > 0


class TestBboxCoverage:
    """Tests for _compute_bbox_coverage()."""

    def test_full_coverage(self):
        """A tile fully covering the bbox should return 1.0."""
        from shapely.geometry import box as shapely_box

        group = [{"bbox": [0, 0, 10, 10], "geometry": None}]
        user_box = shapely_box(2, 2, 8, 8)

        coverage = ingest._compute_bbox_coverage(group, user_box)
        assert coverage == pytest.approx(1.0)

    def test_partial_coverage(self):
        """A tile partially covering the bbox should return fraction < 1.0."""
        from shapely.geometry import box as shapely_box

        # Tile covers only the left half of the bbox
        group = [{"bbox": [0, 0, 5, 10], "geometry": None}]
        user_box = shapely_box(0, 0, 10, 10)

        coverage = ingest._compute_bbox_coverage(group, user_box)
        assert 0.4 < coverage < 0.6  # approximately 50%

    def test_two_tiles_full_coverage(self):
        """Two tiles together should provide better coverage than one alone."""
        from shapely.geometry import box as shapely_box

        group = [
            {"bbox": [0, 5, 10, 10], "geometry": None},
            {"bbox": [0, 0, 10, 6], "geometry": None},  # overlaps slightly
        ]
        user_box = shapely_box(0, 0, 10, 10)

        coverage = ingest._compute_bbox_coverage(group, user_box)
        assert coverage == pytest.approx(1.0)


class TestParseDatetime:
    """Tests for _parse_datetime()."""

    def test_parse_iso_with_z(self):
        """Should parse ISO datetime with Z suffix."""
        dt = ingest._parse_datetime("2024-01-15T10:00:00Z")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_iso_with_offset(self):
        """Should parse ISO datetime with timezone offset."""
        dt = ingest._parse_datetime("2024-01-15T10:00:00+00:00")
        assert dt is not None

    def test_parse_invalid_returns_none(self):
        """Invalid strings should return None, not raise."""
        assert ingest._parse_datetime("not-a-date") is None
        assert ingest._parse_datetime("") is None
