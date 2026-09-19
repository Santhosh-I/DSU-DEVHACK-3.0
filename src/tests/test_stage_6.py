"""Tests for Stage 6 — Source Attribution."""

import importlib

import pytest

attribute = importlib.import_module("pipeline.06_attribute")


class TestScoringFunctions:
    """Tests for individual scoring dimensions."""

    def test_fishing_heuristic_no_token(self):
        """Without GFW token, should return a heuristic score."""
        result = attribute.score_fishing(
            source_bbox=(80.0, 7.0, 81.0, 8.0),
            date_start="2024-01-01",
            date_end="2024-01-31",
            gfw_token=None,
        )
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert result["vessel_count"] == 0

    def test_fishing_gfw_api_success(self, monkeypatch):
        """With valid GFW token and successful API response, should parse GFW v3 response."""
        recorded_calls = []

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "total": 2,
                    "entries": [
                        {
                            "public-global-fishing-effort:v4.0": [
                                {"vesselId": "vessel-123", "hours": 4.5, "flag": "ESP"},
                                {"vesselId": "vessel-456", "hours": 12.0, "flag": "ITA"},
                            ]
                        }
                    ]
                }

        def mock_post(url, headers=None, params=None, json=None, timeout=None):
            recorded_calls.append({
                "url": url,
                "headers": headers,
                "params": params,
                "json": json,
                "timeout": timeout,
            })
            return MockResponse()

        import requests
        monkeypatch.setattr(requests, "post", mock_post)

        result = attribute.score_fishing(
            source_bbox=(2.0, 39.0, 4.0, 41.0),
            date_start="2023-01-01",
            date_end="2023-01-31",
            gfw_token="mock_valid_token",
        )

        assert len(recorded_calls) == 1
        call = recorded_calls[0]

        # Verify endpoint and headers
        assert call["url"] == "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
        assert call["headers"]["Authorization"] == "Bearer mock_valid_token"
        assert call["headers"]["Content-Type"] == "application/json"

        # Verify GFW API v3 query parameters
        assert call["params"]["spatial-resolution"] == "LOW"
        assert call["params"]["temporal-resolution"] == "MONTHLY"
        assert call["params"]["group-by"] == "VESSEL_ID"
        assert call["params"]["format"] == "JSON"
        assert call["params"]["datasets[0]"] == "public-global-fishing-effort:latest"

        # Verify GeoJSON polygon in request body
        assert "geojson" in call["json"]
        assert call["json"]["geojson"]["type"] == "FeatureCollection"
        geom = call["json"]["geojson"]["features"][0]["geometry"]
        assert geom["type"] == "Polygon"

        # Verify parsed scores and vessel IDs
        assert result["vessel_count"] == 2
        assert result["vessel_ids"] == ["vessel-123", "vessel-456"]
        assert result["score"] == pytest.approx(2 / 20.0)
        assert result["gfw_unprocessable"] is False

    def test_fishing_gfw_api_error_fallback(self, monkeypatch):
        """When GFW API raises an error, it should gracefully fall back."""
        import requests

        def mock_post_fail(*args, **kwargs):
            mock_resp = requests.Response()
            mock_resp.status_code = 422
            raise requests.exceptions.HTTPError("422 Client Error: Unprocessable Entity", response=mock_resp)

        monkeypatch.setattr(requests, "post", mock_post_fail)

        result = attribute.score_fishing(
            source_bbox=(2.0, 39.0, 4.0, 41.0),
            date_start="2023-01-01",
            date_end="2023-01-31",
            gfw_token="mock_token",
        )

        assert result["score"] == 0.3
        assert result["vessel_count"] == 0
        assert result["vessel_ids"] == []
        assert result["gfw_unprocessable"] is True

    def test_fishing_polar_latitude_skips_gfw(self, monkeypatch):
        """Polar latitudes (>=80 deg) should skip GFW queries."""
        called = []
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **kw: called.append(True))

        result = attribute.score_fishing(
            source_bbox=(-10.0, 82.0, -9.0, 83.0),
            date_start="2023-01-01",
            date_end="2023-01-31",
            gfw_token="mock_token",
        )

        assert len(called) == 0
        assert result["vessel_count"] == 0
        assert result["gfw_unprocessable"] is True
        assert 0 <= result["score"] <= 1


    def test_shipping_heuristic(self):
        """Shipping score should work without reference data."""
        result = attribute.score_shipping(
            source_bbox=(80.0, 7.0, 81.0, 8.0),
        )
        assert "score" in result
        assert 0 <= result["score"] <= 1

    def test_river_scoring(self):
        """River scoring should find nearby rivers for Indian Ocean."""
        result = attribute.score_river(
            source_bbox=(80.0, 7.0, 81.0, 8.0),
        )
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert "nearest_river" in result


class TestCompositeAttribution:
    """Tests for the composite attribution scoring."""

    def test_weighted_score(self):
        """Composite score should be a weighted sum of individual scores."""
        scores = {
            "fishing": {"score": 0.8},
            "industrial": {"score": 0.5},
            "shipping": {"score": 0.3},
            "river": {"score": 0.1},
        }
        weights = {"fishing": 0.4, "industrial": 0.3, "shipping": 0.2, "river": 0.1}

        result = attribute.compute_attribution(scores, weights)

        expected = 0.4 * 0.8 + 0.3 * 0.5 + 0.2 * 0.3 + 0.1 * 0.1
        assert result["attribution_score"] == pytest.approx(expected, abs=1e-6)
        assert result["source_type"] == "fishing"

    def test_empty_scores(self):
        """Missing scores should default to zero."""
        result = attribute.compute_attribution(
            scores={},
            weights={"fishing": 0.5, "industrial": 0.5},
        )
        assert result["attribution_score"] == 0.0


class TestNearbyShips:
    """Tests for locating ships near debris clusters using GFW."""

    def test_haversine_distance(self):
        """Haversine formula should accurately compute distance."""
        # Approx 1 degree of latitude at equator is ~111.19 km
        dist = attribute.haversine_km(0.0, 0.0, 1.0, 0.0)
        assert 110.0 <= dist <= 112.0

    def test_find_ships_near_debris_with_cached_records(self):
        """Should filter by search radius, rank by distance, and select nearest ship."""
        cluster_lat = 16.0
        cluster_lon = -88.0

        records = [
            {
                "shipName": "FAR_SHIP",
                "mmsi": "111111111",
                "vesselType": "CARGO",
                "flag": "PAN",
                "lat": 18.0,
                "lon": -88.0,
                "hours": 5.0,
            },
            {
                "shipName": "CLOSE_SHIP",
                "mmsi": "222222222",
                "vesselType": "FISHING",
                "flag": "HND",
                "lat": 16.08,
                "lon": -88.05,
                "hours": 2.5,
            },
            {
                "shipName": "MID_SHIP",
                "mmsi": "333333333",
                "vesselType": "TANKER",
                "flag": "USA",
                "lat": 16.25,
                "lon": -88.15,
                "hours": 1.0,
            },
        ]

        result = attribute.find_ships_near_debris(
            cluster_lat=cluster_lat,
            cluster_lon=cluster_lon,
            date_start="2020-09-01",
            date_end="2020-09-30",
            search_radius_km=50.0,
            cached_vessel_records=records,
        )

        assert result["nearby_vessel_count"] == 2
        assert len(result["nearby_vessels"]) == 2
        assert result["nearest_ship"] is not None
        assert result["nearest_ship"]["ship_name"] == "CLOSE_SHIP"
        assert result["nearest_ship"]["mmsi"] == "222222222"
        assert result["nearest_ship"]["vessel_type"] == "FISHING"
        assert result["nearest_ship"]["distance_km"] < result["nearby_vessels"][1]["distance_km"]

    def test_find_ships_near_debris_no_token_empty(self):
        """Without token or cached records, should return empty without error."""
        result = attribute.find_ships_near_debris(
            cluster_lat=16.0,
            cluster_lon=-88.0,
            date_start="2020-09-01",
            date_end="2020-09-30",
            gfw_token=None,
        )
        assert result["nearest_ship"] is None
        assert result["nearby_vessels"] == []
        assert result["nearby_vessel_count"] == 0

    def test_run_generates_attribution_with_ships(self, tmp_path, monkeypatch):
        """attribute.run() should include nearest_ship and nearby_vessels in output."""
        import json
        import geopandas as gpd
        from shapely.geometry import Point

        det_file = tmp_path / "detections.geojson"
        gdf = gpd.GeoDataFrame({
            "geometry": [Point(-88.0, 16.0)],
            "cluster_id": [101],
            "area_m2": [1200.0],
            "centroid_lat": [16.0],
            "centroid_lon": [-88.0],
            "polymer_type": ["Marine Debris (Plastic)"],
        }, crs="EPSG:4326")
        gdf.to_file(det_file, driver="GeoJSON")

        mock_records = [{
            "shipName": "SEAS_HUNTER",
            "mmsi": "312456000",
            "vesselType": "FISHING",
            "flag": "BLZ",
            "lat": 16.1,
            "lon": -88.05,
            "hours": 4.2,
        }]

        monkeypatch.setattr(
            attribute,
            "query_gfw_vessels_in_bbox",
            lambda *args, **kwargs: (mock_records, False)
        )

        out_path = attribute.run(
            scene_id="test_scene",
            sources=[],
            detections_path=str(det_file),
            output_dir=str(tmp_path / "attribution"),
            config={"apis": {"gfw_token": "mock_token"}},
            detection_date="2020-09-18",
        )

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)

        assert len(data) == 1
        entry = data[0]
        assert entry["debris_cluster_id"] == 101
        assert entry["nearest_ship"] is not None
        assert entry["nearest_ship"]["ship_name"] == "SEAS_HUNTER"
        assert entry["nearest_ship"]["mmsi"] == "312456000"
        assert entry["nearby_vessel_count"] == 1
        assert "SEAS_HUNTER" in entry["explanation"]

