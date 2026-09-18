"""
Plastic-Ledger — Stage 6: Source Attribution
===============================================
Matches candidate source regions to known industrial discharge points,
shipping lanes, fishing zones, and coastal population centres.

Usage (standalone):
    python -m pipeline.06_attribute \\
        --scene_id SCENE_ID \\
        --sources data/attribution/SCENE_ID/backtrack_summary.json \\
        --detections data/detections/SCENE_ID/detections_classified.geojson

Dependencies: geopandas, requests, osmnx, shapely, pandas, numpy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.geo_utils import retry_request, expand_bbox
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)


def _should_retry_gfw_error(exc: Exception) -> bool:
    """Retry transient request failures, but fail fast on permanent 4xx errors."""
    try:
        import requests
    except Exception:
        return True

    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status is not None and 400 <= status < 500 and status != 429:
            return False
    return True


# ─────────────────────────────────────────────
# FISHING VESSEL SCORING
# ─────────────────────────────────────────────
def score_fishing(
    source_bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    gfw_token: Optional[str] = None,
    search_radius_km: float = 10.0,
) -> Dict[str, Any]:
    """Score fishing vessel activity near a source region.

    Args:
        source_bbox: ``(lon_min, lat_min, lon_max, lat_max)`` of the source.
        date_start: Start date ISO string.
        date_end: End date ISO string.
        gfw_token: Global Fishing Watch API token.
        search_radius_km: Search radius in km.

    Returns:
        Dict with ``score`` (0–1), ``vessel_count``, ``vessel_ids``.
    """
    expanded = expand_bbox(source_bbox, search_radius_km / 111.0)
    centroid_lat = (source_bbox[1] + source_bbox[3]) / 2

    # GFW frequently rejects geometries very close to the poles.
    if abs(centroid_lat) >= 80:
        logger.info("  Skipping GFW query at polar latitude (%.2f)", centroid_lat)
        fishing_heuristic = max(0, 1.0 - abs(centroid_lat) / 60.0) * 0.5
        return {
            "score": fishing_heuristic,
            "vessel_count": 0,
            "vessel_ids": [],
            "gfw_unprocessable": True,
        }

    if not gfw_token:
        logger.info("  No GFW token — using heuristic fishing score")
        # Heuristic: coastal areas in high-fishing zones score higher
        # Fishing is more common in tropical/subtropical zones
        fishing_heuristic = max(0, 1.0 - abs(centroid_lat) / 60.0) * 0.5
        return {
            "score": fishing_heuristic,
            "vessel_count": 0,
            "vessel_ids": [],
            "gfw_unprocessable": False,
        }

    try:
        import requests

        def _query_gfw_impl():
            headers = {
                "Authorization": f"Bearer {gfw_token}",
                "Content-Type": "application/json",
            }
            url = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
            params = {
                "spatial-resolution": "LOW",
                "temporal-resolution": "MONTHLY",
                "group-by": "VESSEL_ID",
                "format": "JSON",
                "datasets[0]": "public-global-fishing-effort:latest",
                "date-range": f"{date_start},{date_end}",
            }
            body = {
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [expanded[0], expanded[1]],
                                [expanded[2], expanded[1]],
                                [expanded[2], expanded[3]],
                                [expanded[0], expanded[3]],
                                [expanded[0], expanded[1]],
                            ]],
                        },
                    }],
                }
            }
            resp = requests.post(url, headers=headers, params=params, json=body, timeout=30)
            resp.raise_for_status()
            return resp.json()

        _query_gfw = retry_request(
            _query_gfw_impl,
            retry_if=_should_retry_gfw_error,
        )

        data = _query_gfw()
        vessel_records = []
        for entry in data.get("entries", []):
            if isinstance(entry, dict):
                for key, val in entry.items():
                    if isinstance(val, list):
                        vessel_records.extend(val)
                    elif isinstance(val, dict):
                        vessel_records.append(val)
            elif isinstance(entry, list):
                vessel_records.extend(entry)

        vessel_ids = [v.get("vesselId") for v in vessel_records if isinstance(v, dict) and v.get("vesselId")]
        unique_vessel_ids = list(dict.fromkeys(vessel_ids))
        vessel_count = len(unique_vessel_ids)
        score = min(1.0, vessel_count / 20.0)

        return {
            "score": score,
            "vessel_count": vessel_count,
            "vessel_ids": unique_vessel_ids[:10],
            "gfw_unprocessable": False,
        }

    except Exception as exc:
        logger.warning("GFW query failed: %s", exc)
        is_422 = " 422 " in f" {exc} "
        return {
            "score": 0.3,
            "vessel_count": 0,
            "vessel_ids": [],
            "gfw_unprocessable": is_422,
        }


# ─────────────────────────────────────────────
# INDUSTRIAL SITE SCORING
# ─────────────────────────────────────────────
def score_industrial(
    source_bbox: Tuple[float, float, float, float],
    search_radius_km: float = 10.0,
) -> Dict[str, Any]:
    """Score proximity to coastal industrial/waste sites via OSM.

    Args:
        source_bbox: Source region bounding box.
        search_radius_km: Search radius in km.

    Returns:
        Dict with ``score`` (0–1), ``site_count``, ``site_names``.
    """
    expanded = expand_bbox(source_bbox, search_radius_km / 111.0)
    centroid_lat = (source_bbox[1] + source_bbox[3]) / 2

    # OSM coverage/queries are unreliable near polar regions; use heuristic.
    if abs(centroid_lat) >= 80:
        logger.info("  Skipping OSM query at polar latitude (%.2f)", centroid_lat)
        return {"score": 0.2, "site_count": 0, "site_names": []}

    try:
        import osmnx as ox

        ox.settings.timeout = 20

        tags = {
            "amenity": ["waste_disposal", "waste_transfer_station", "recycling"],
            "landuse": ["industrial"],
            "man_made": ["wastewater_plant"],
        }

        gdf = ox.features_from_bbox(
            bbox=expanded,
            tags=tags,
        )
        site_count = len(gdf)
        site_names = gdf["name"].dropna().tolist()[:10] if "name" in gdf.columns else []

        score = min(1.0, site_count / 10.0)
        return {
            "score": score,
            "site_count": site_count,
            "site_names": site_names[:10],
        }

    except ImportError:
        logger.warning("osmnx not installed — using heuristic industrial score")
        return {"score": 0.2, "site_count": 0, "site_names": []}
    except Exception as exc:
        if "No matching features" in str(exc):
            return {"score": 0.0, "site_count": 0, "site_names": []}
        logger.warning("OSM query failed: %s", exc)
        return {"score": 0.2, "site_count": 0, "site_names": []}


# ─────────────────────────────────────────────
# SHIPPING LANE SCORING
# ─────────────────────────────────────────────
def score_shipping(
    source_bbox: Tuple[float, float, float, float],
    reference_dir: Path = Path("data/reference"),
) -> Dict[str, Any]:
    """Score overlap with major shipping lanes.

    Args:
        source_bbox: Source region bounding box.
        reference_dir: Directory for cached reference data.

    Returns:
        Dict with ``score`` (0–1), ``overlap_area``.
    """
    source_poly = box(*source_bbox)

    # Try to load shipping lane data
    shipping_file = reference_dir / "shipping_lanes.geojson"
    if shipping_file.exists():
        try:
            shipping_gdf = gpd.read_file(shipping_file)
            overlap = shipping_gdf.intersection(source_poly)
            total_overlap = overlap.area.sum()
            source_area = source_poly.area
            score = min(1.0, total_overlap / (source_area + 1e-10))
            return {"score": score, "overlap_area": float(total_overlap)}
        except Exception as exc:
            logger.warning("Shipping lane scoring failed: %s", exc)

    # Heuristic: proximity to known major shipping routes
    centroid_lon = (source_bbox[0] + source_bbox[2]) / 2
    centroid_lat = (source_bbox[1] + source_bbox[3]) / 2

    # Major shipping corridor heuristic (Indian Ocean / SE Asia)
    is_near_shipping = (
        (5 < centroid_lat < 25 and 60 < centroid_lon < 120) or  # Indian Ocean
        (0 < centroid_lat < 40 and -10 < centroid_lon < 40) or  # Mediterranean
        (20 < centroid_lat < 50 and 100 < centroid_lon < 180)   # Pacific
    )
    score = 0.6 if is_near_shipping else 0.2

    return {"score": score, "overlap_area": 0.0}


# ─────────────────────────────────────────────
# RIVER DISCHARGE SCORING
# ─────────────────────────────────────────────
def score_river(
    source_bbox: Tuple[float, float, float, float],
    reference_dir: Path = Path("data/reference"),
    max_distance_km: float = 200.0,
) -> Dict[str, Any]:
    """Score distance to nearest major river mouth.

    Args:
        source_bbox: Source region bounding box.
        reference_dir: Directory for cached reference data.
        max_distance_km: Maximum distance for scoring.

    Returns:
        Dict with ``score`` (0–1), ``nearest_river``, ``distance_km``.
    """
    centroid = Point(
        (source_bbox[0] + source_bbox[2]) / 2,
        (source_bbox[1] + source_bbox[3]) / 2,
    )

    # Try loading river mouths data
    rivers_file = reference_dir / "river_mouths.geojson"
    if rivers_file.exists():
        try:
            rivers_gdf = gpd.read_file(rivers_file)
            distances = rivers_gdf.geometry.distance(centroid)
            min_idx = distances.idxmin()
            min_dist_deg = distances.min()
            min_dist_km = min_dist_deg * 111.0  # approximate

            score = max(0, 1.0 - min_dist_km / max_distance_km)
            river_name = rivers_gdf.loc[min_idx].get("name", "Unknown River")

            return {
                "score": score,
                "nearest_river": river_name,
                "distance_km": float(min_dist_km),
            }
        except Exception as exc:
            logger.warning("River scoring failed: %s", exc)

    # Heuristic: well-known river mouths near major plastic sources
    major_rivers = [
        {"name": "Ganges", "lon": 88.8, "lat": 21.7},
        {"name": "Indus", "lon": 67.5, "lat": 23.9},
        {"name": "Mekong", "lon": 106.7, "lat": 9.8},
        {"name": "Yangtze", "lon": 121.9, "lat": 31.4},
        {"name": "Nile", "lon": 31.5, "lat": 31.5},
        {"name": "Niger", "lon": 6.0, "lat": 4.3},
        {"name": "Amazon", "lon": -50.0, "lat": -0.5},
        {"name": "Pearl", "lon": 113.5, "lat": 22.2},
        {"name": "Mahaweli", "lon": 81.3, "lat": 8.6},
        {"name": "Kelani", "lon": 79.8, "lat": 6.9},
    ]

    min_dist = float("inf")
    nearest = "Unknown"

    for river in major_rivers:
        dist_km = np.sqrt(
            ((centroid.x - river["lon"]) * 111 * np.cos(np.radians(centroid.y))) ** 2
            + ((centroid.y - river["lat"]) * 111) ** 2
        )
        if dist_km < min_dist:
            min_dist = dist_km
            nearest = river["name"]

    score = max(0, 1.0 - min_dist / max_distance_km)
    return {
        "score": score,
        "nearest_river": nearest,
        "distance_km": float(min_dist),
    }


# ─────────────────────────────────────────────
# COMPOSITE ATTRIBUTION
# ─────────────────────────────────────────────
def compute_attribution(
    scores: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Compute weighted composite attribution score.

    Args:
        scores: Dict mapping source type → scoring result.
        weights: Dict mapping source type → weight (should sum to 1).

    Returns:
        Dict with ``attribution_score``, ``source_type``, individual scores.
    """
    composite = 0.0
    for key, weight in weights.items():
        s = scores.get(key, {}).get("score", 0.0)
        composite += weight * s

    # Determine dominant source type
    source_type = max(weights.keys(), key=lambda k: scores.get(k, {}).get("score", 0))

    return {
        "attribution_score": float(composite),
        "source_type": source_type,
        "fishing_score": scores.get("fishing", {}).get("score", 0),
        "industrial_score": scores.get("industrial", {}).get("score", 0),
        "shipping_score": scores.get("shipping", {}).get("score", 0),
        "river_score": scores.get("river", {}).get("score", 0),
    }


def generate_explanation(
    attribution: Dict[str, Any],
    scores: Dict[str, Dict[str, Any]],
    source_region: Dict[str, Any],
) -> str:
    """Generate human-readable attribution explanation.

    Args:
        attribution: Result from :func:`compute_attribution`.
        scores: Individual scoring results.
        source_region: Source region info from backtracking.

    Returns:
        Human-readable explanation string.
    """
    conf = attribution["attribution_score"]
    src_type = attribution["source_type"]
    centroid = source_region.get("source_centroid", (0, 0))
    days = source_region.get("days_to_source", "unknown")

    parts = [
        f"{'High' if conf > 0.6 else 'Moderate' if conf > 0.3 else 'Low'} "
        f"probability ({conf*100:.0f}%): "
    ]

    if src_type == "fishing":
        fishing = scores.get("fishing", {})
        vc = fishing.get("vessel_count", 0)
        parts.append(
            f"Fishing activity near ({centroid[0]:.2f}, {centroid[1]:.2f}). "
            f"{vc} vessel(s) detected in this area ~{days} days before detection."
        )
    elif src_type == "industrial":
        ind = scores.get("industrial", {})
        sc = ind.get("site_count", 0)
        names = ind.get("site_names", [])
        name_str = f" ({', '.join(names[:3])})" if names else ""
        parts.append(
            f"{sc} industrial/waste site(s){name_str} within search radius "
            f"of source region at ({centroid[0]:.2f}, {centroid[1]:.2f})."
        )
    elif src_type == "shipping":
        parts.append(
            f"Major shipping lane overlap near ({centroid[0]:.2f}, {centroid[1]:.2f}). "
            f"Estimated {days} days drift time."
        )
    elif src_type == "river":
        river = scores.get("river", {})
        rname = river.get("nearest_river", "Unknown")
        rdist = river.get("distance_km", 0)
        parts.append(
            f"River discharge from {rname} ({rdist:.0f} km from source region). "
            f"Estimated {days} days drift time."
        )

    return "".join(parts)


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_id: str,
    sources: List[Dict[str, Any]],
    detections_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/attribution",
    config: Optional[Dict] = None,
    detection_date: Optional[str] = None,
) -> Path:
    """Run source attribution for all candidate source regions.

    Args:
        scene_id: Scene identifier.
        sources: Source regions from Stage 5 backtracking.
        detections_path: Path to classified detections GeoJSON.
        output_dir: Root output directory.
        config: Optional config dict.
        detection_date: ISO date string.

    Returns:
        Path to ``attribution_report.json``.
    """
    out_dir = Path(output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "attribution_report.json"

    # Check cache
    if stage_output_exists(out_dir, ["attribution_report.json"]):
        return report_path

    # Settings
    weights = {"fishing": 0.4, "industrial": 0.3, "shipping": 0.2, "river": 0.1}
    search_radius_km = 10.0
    gfw_token = None
    disable_gfw = False
    reference_dir = Path("data/reference")

    if config:
        attr_cfg = config.get("attribution", {})
        w = attr_cfg.get("weights", {})
        if w:
            weights = {k: w.get(k, v) for k, v in weights.items()}
        search_radius_km = attr_cfg.get("search_radius_km", 10.0)
        gfw_token = config.get("apis", {}).get("gfw_token")
        if gfw_token and gfw_token.startswith("your_"):
            gfw_token = None

    # Date range for queries
    if detection_date:
        from datetime import datetime, timedelta
        det_dt = datetime.fromisoformat(detection_date.replace("Z", "+00:00"))
    else:
        from datetime import datetime, timedelta
        det_dt = datetime.now()

    date_start = (det_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    date_end = det_dt.strftime("%Y-%m-%d")

    # Process each source region
    report_entries = []

    if not sources:
        logger.info("No source regions to attribute — writing empty report")
        with open(report_path, "w") as fh:
            json.dump([], fh, indent=2)
        return report_path

    for rank, source in enumerate(sources):
        src_bbox = source.get("source_bbox")
        if not src_bbox:
            continue

        logger.info(
            "Scoring source region %d: bbox=%s", rank + 1, src_bbox,
        )

        # Score each dimension
        fishing_score = score_fishing(
            src_bbox,
            date_start,
            date_end,
            None if disable_gfw else gfw_token,
            search_radius_km,
        )
        if fishing_score.get("gfw_unprocessable"):
            disable_gfw = True

        scores = {
            "fishing": fishing_score,
            "industrial": score_industrial(src_bbox, search_radius_km),
            "shipping": score_shipping(src_bbox, reference_dir),
            "river": score_river(src_bbox, reference_dir),
        }

        attribution = compute_attribution(scores, weights)
        explanation = generate_explanation(attribution, scores, source)

        entry = {
            "debris_cluster_id": source.get("cluster_id", rank),
            "source_rank": rank + 1,
            "source_type": attribution["source_type"],
            "location_name": _get_location_name(source.get("source_centroid", (0, 0))),
            "country": _get_country_name(source.get("source_centroid", (0, 0))),
            "attribution_score": attribution["attribution_score"],
            "confidence": "high" if attribution["attribution_score"] > 0.6
                         else "moderate" if attribution["attribution_score"] > 0.3
                         else "low",
            "explanation": explanation,
            "source_centroid": source.get("source_centroid"),
            "source_bbox": source.get("source_bbox"),
            "source_probability": source.get("source_probability"),
            "days_to_source": source.get("days_to_source"),
            "fishing_score": attribution.get("fishing_score"),
            "industrial_score": attribution.get("industrial_score"),
            "shipping_score": attribution.get("shipping_score"),
            "river_score": attribution.get("river_score"),
            "vessel_ids": scores.get("fishing", {}).get("vessel_ids", []),
        }
        report_entries.append(entry)

    # Sort by attribution score descending
    report_entries.sort(key=lambda x: x["attribution_score"], reverse=True)

    with open(report_path, "w") as fh:
        json.dump(report_entries, fh, indent=2, default=str)

    logger.info(
        "[bold green]Stage 6 complete[/] — %d sources attributed, top: %s (%.1f%%)",
        len(report_entries),
        report_entries[0]["source_type"] if report_entries else "N/A",
        report_entries[0]["attribution_score"] * 100 if report_entries else 0,
    )
    return report_path


def _get_location_name(centroid: Tuple[float, float]) -> str:
    """Get approximate location name from coordinates.

    Args:
        centroid: ``(lon, lat)`` coordinates.

    Returns:
        Location description string.
    """
    lon, lat = centroid
    # Simple grid-based naming (could be enhanced with geocoding API)
    if 79 < lon < 82 and 5 < lat < 10:
        return "Sri Lankan Coast"
    elif 87 < lon < 90 and 20 < lat < 23:
        return "Bay of Bengal (Bangladesh)"
    elif 100 < lon < 110 and 5 < lat < 15:
        return "Gulf of Thailand"
    elif lon > 0 and lat > 0:
        return f"Ocean region ({lon:.1f}°E, {lat:.1f}°N)"
    else:
        return f"Ocean region ({abs(lon):.1f}°{'W' if lon < 0 else 'E'}, " \
               f"{abs(lat):.1f}°{'S' if lat < 0 else 'N'})"


def _get_country_name(centroid: Tuple[float, float]) -> str:
    """Get approximate country from coordinates.

    Args:
        centroid: ``(lon, lat)`` coordinates.

    Returns:
        Country name string.
    """
    lon, lat = centroid
    # Simple heuristic lookup
    if 79 < lon < 82 and 5 < lat < 10:
        return "Sri Lanka"
    elif 87 < lon < 93 and 20 < lat < 27:
        return "Bangladesh"
    elif 68 < lon < 78 and 8 < lat < 24:
        return "India"
    elif 95 < lon < 106 and 5 < lat < 21:
        return "Thailand/Vietnam"
    return "International Waters"


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 6: Source attribution for debris clusters",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--sources", type=str, required=True,
                        help="Path to backtrack_summary.json")
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/attribution")
    parser.add_argument("--detection_date", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.sources) as fh:
        sources = json.load(fh)

    report_path = run(
        scene_id=args.scene_id,
        sources=sources,
        detections_path=args.detections,
        output_dir=args.output_dir,
        config=config,
        detection_date=args.detection_date,
    )
    print(f"\nAttribution report saved to {report_path}")


if __name__ == "__main__":
    main()
