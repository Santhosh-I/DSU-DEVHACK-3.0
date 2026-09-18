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
