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
import math
import os
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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in kilometers."""
    r = 6371.0  # Earth radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


def query_gfw_vessels_in_bbox(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    gfw_token: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Query Global Fishing Watch 4wings report API for vessels in a bounding box.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max)
        date_start: ISO start date
        date_end: ISO end date
        gfw_token: GFW API Bearer token

    Returns:
        Tuple of (vessel_records, gfw_unprocessable)
    """
    if not gfw_token:
        return [], False

    centroid_lat = (bbox[1] + bbox[3]) / 2.0
    if abs(centroid_lat) >= 80:
        logger.info("  Skipping GFW query at polar latitude (%.2f)", centroid_lat)
        return [], True

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
                                [bbox[0], bbox[1]],
                                [bbox[2], bbox[1]],
                                [bbox[2], bbox[3]],
                                [bbox[0], bbox[3]],
                                [bbox[0], bbox[1]],
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

        return vessel_records, False

    except Exception as exc:
        logger.warning("GFW query failed: %s", exc)
        is_422 = " 422 " in f" {exc} "
        return [], is_422


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
        fishing_heuristic = max(0, 1.0 - abs(centroid_lat) / 60.0) * 0.5
        return {
            "score": fishing_heuristic,
            "vessel_count": 0,
            "vessel_ids": [],
            "gfw_unprocessable": False,
        }

    vessel_records, is_422 = query_gfw_vessels_in_bbox(expanded, date_start, date_end, gfw_token)
    if is_422:
        return {
            "score": 0.3,
            "vessel_count": 0,
            "vessel_ids": [],
            "gfw_unprocessable": True,
        }

    vessel_ids = [v.get("vesselId") for v in vessel_records if isinstance(v, dict) and v.get("vesselId")]
    unique_vessel_ids = list(dict.fromkeys(vessel_ids))
    vessel_count = len(unique_vessel_ids)
    score = min(1.0, vessel_count / 20.0)

    return {
        "score": score,
        "vessel_count": vessel_count,
        "vessel_ids": unique_vessel_ids[:10],
        "vessel_records": vessel_records,
        "gfw_unprocessable": False,
    }


def find_ships_near_debris(
    cluster_lat: float,
    cluster_lon: float,
    date_start: str,
    date_end: str,
    gfw_token: Optional[str] = None,
    search_radius_km: float = 50.0,
    cached_vessel_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Find ships/vessels near a debris cluster location using Global Fishing Watch API.

    Args:
        cluster_lat: Centroid latitude of the debris cluster.
        cluster_lon: Centroid longitude of the debris cluster.
        date_start: Start date (YYYY-MM-DD).
        date_end: End date (YYYY-MM-DD).
        gfw_token: Global Fishing Watch API token.
        search_radius_km: Search radius in km (default 50.0).
        cached_vessel_records: Optional pre-queried vessel records from scene.

    Returns:
        Dict with:
            - nearest_ship: Closest vessel dict or None
            - nearby_vessels: List of nearby vessels sorted by distance
            - nearby_vessel_count: Total count of nearby vessels
    """
    if cached_vessel_records is not None:
        raw_records = cached_vessel_records
    else:
        # Expand bounding box around cluster
        delta_lat = search_radius_km / 111.0
        cos_lat = math.cos(math.radians(cluster_lat))
        delta_lon = search_radius_km / (111.0 * max(0.01, abs(cos_lat)))
        query_bbox = (
            max(-180.0, cluster_lon - delta_lon),
            max(-85.0, cluster_lat - delta_lat),
            min(180.0, cluster_lon + delta_lon),
            min(85.0, cluster_lat + delta_lat),
        )
        raw_records, _ = query_gfw_vessels_in_bbox(query_bbox, date_start, date_end, gfw_token)

    if not raw_records:
        return {
            "nearest_ship": None,
            "nearby_vessels": [],
            "nearby_vessel_count": 0,
        }

    # Group records by vessel identity and find minimum distance to cluster
    vessels_by_id: Dict[str, Dict[str, Any]] = {}

    for rec in raw_records:
        if not isinstance(rec, dict):
            continue

        v_lat = rec.get("lat")
        v_lon = rec.get("lon")
        if v_lat is None or v_lon is None:
            continue

        try:
            v_lat = float(v_lat)
            v_lon = float(v_lon)
        except (ValueError, TypeError):
            continue

        dist_km = haversine_km(cluster_lat, cluster_lon, v_lat, v_lon)
        if dist_km > search_radius_km:
            continue

        vessel_id = rec.get("vesselId") or rec.get("mmsi") or rec.get("shipName") or f"{v_lat}_{v_lon}"
        ship_name = rec.get("shipName")
        mmsi = rec.get("mmsi")
        imo = rec.get("imo")
        callsign = rec.get("callsign")
        flag = rec.get("flag")
        vessel_type = rec.get("vesselType") or rec.get("geartype") or "UNKNOWN"
        geartype = rec.get("geartype") or ""
        hours = float(rec.get("hours", 0) or 0)
        timestamp = rec.get("entryTimestamp") or rec.get("exitTimestamp") or rec.get("date") or ""

        if vessel_id not in vessels_by_id:
            vessels_by_id[vessel_id] = {
                "ship_name": ship_name if ship_name else (f"MMSI {mmsi}" if mmsi else "Unknown Vessel"),
                "mmsi": str(mmsi) if mmsi else "",
                "imo": str(imo) if imo else "",
                "callsign": str(callsign) if callsign else "",
                "flag": str(flag) if flag else "",
                "vessel_type": str(vessel_type),
                "geartype": str(geartype),
                "distance_km": round(dist_km, 2),
                "lat": round(v_lat, 4),
                "lon": round(v_lon, 4),
                "hours": round(hours, 2),
                "last_timestamp": str(timestamp),
                "vessel_id": str(vessel_id),
            }
        else:
            existing = vessels_by_id[vessel_id]
            existing["hours"] = round(existing["hours"] + hours, 2)
            if dist_km < existing["distance_km"]:
                existing["distance_km"] = round(dist_km, 2)
                existing["lat"] = round(v_lat, 4)
                existing["lon"] = round(v_lon, 4)
            if timestamp and (not existing["last_timestamp"] or timestamp > existing["last_timestamp"]):
                existing["last_timestamp"] = str(timestamp)
            if not existing["ship_name"] and ship_name:
                existing["ship_name"] = ship_name

    # Sort vessels by distance ascending
    sorted_vessels = sorted(vessels_by_id.values(), key=lambda v: v["distance_km"])
    nearest_ship = sorted_vessels[0] if sorted_vessels else None

    return {
        "nearest_ship": nearest_ship,
        "nearby_vessels": sorted_vessels,
        "nearby_vessel_count": len(sorted_vessels),
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
    """Run source attribution for candidate source regions and locate ships near debris clusters.

    Args:
        scene_id: Scene identifier.
        sources: Source regions from Stage 5 backtracking.
        detections_path: Path to classified detections GeoJSON or CSV.
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
    debris_search_radius_km = 50.0
    gfw_token = None
    disable_gfw = False
    reference_dir = Path("data/reference")

    if config:
        attr_cfg = config.get("attribution", {})
        w = attr_cfg.get("weights", {})
        if w:
            weights = {k: w.get(k, v) for k, v in weights.items()}
        search_radius_km = attr_cfg.get("search_radius_km", 10.0)
        debris_search_radius_km = attr_cfg.get("debris_search_radius_km", 50.0)
        gfw_token = config.get("apis", {}).get("gfw_token")
        if gfw_token and str(gfw_token).startswith("your_"):
            gfw_token = None

    if not gfw_token:
        env_tok = os.environ.get("GFW_TOKEN")
        if env_tok and not env_tok.startswith("your_"):
            gfw_token = env_tok

    # Date range for queries
    if detection_date:
        from datetime import datetime, timedelta
        det_dt = datetime.fromisoformat(str(detection_date).replace("Z", "+00:00"))
    else:
        from datetime import datetime, timedelta
        det_dt = datetime.now()

    date_start = (det_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    date_end = det_dt.strftime("%Y-%m-%d")

    # Load detections
    detections_gdf = None
    if detections_path:
        det_path = Path(detections_path)
        if det_path.exists():
            try:
                if det_path.suffix.lower() in [".geojson", ".json"]:
                    detections_gdf = gpd.read_file(det_path)
                elif det_path.suffix.lower() == ".csv":
                    det_df = pd.read_csv(det_path)
                    if "lon" in det_df.columns and "lat" in det_df.columns:
                        detections_gdf = gpd.GeoDataFrame(
                            det_df,
                            geometry=gpd.points_from_xy(det_df["lon"], det_df["lat"]),
                            crs="EPSG:4326",
                        )
                    else:
                        detections_gdf = gpd.GeoDataFrame(det_df)
            except Exception as exc:
                logger.warning("Could not load detections from %s: %s", detections_path, exc)

    # Ensure centroid columns in detections_gdf if present
    if detections_gdf is not None and len(detections_gdf) > 0:
        if "centroid_lon" not in detections_gdf.columns or "centroid_lat" not in detections_gdf.columns:
            if "geometry" in detections_gdf.columns:
                try:
                    c = detections_gdf.geometry.centroid
                    detections_gdf["centroid_lon"] = c.x
                    detections_gdf["centroid_lat"] = c.y
                except Exception:
                    pass
            if "lon" in detections_gdf.columns and "centroid_lon" not in detections_gdf.columns:
                detections_gdf["centroid_lon"] = detections_gdf["lon"]
            if "lat" in detections_gdf.columns and "centroid_lat" not in detections_gdf.columns:
                detections_gdf["centroid_lat"] = detections_gdf["lat"]

    # Limit to max_clusters ("n clusters")
    max_clusters = None
    if config and "backtracking" in config:
        max_clusters = config["backtracking"].get("max_clusters")

    target_clusters = []
    if detections_gdf is not None and len(detections_gdf) > 0:
        clusters_df = detections_gdf.copy()
        if "polymer_type" in clusters_df.columns:
            plastics = clusters_df[clusters_df["polymer_type"] == "Marine Debris (Plastic)"]
            if len(plastics) > 0:
                clusters_df = plastics

        if "area_m2" in clusters_df.columns:
            clusters_df = clusters_df.sort_values(by="area_m2", ascending=False)

        if max_clusters is not None and len(clusters_df) > max_clusters:
            clusters_df = clusters_df.iloc[:max_clusters]

        for _, row in clusters_df.iterrows():
            cid = row.get("cluster_id")
            c_lat = float(row.get("centroid_lat", 0))
            c_lon = float(row.get("centroid_lon", 0))
            area = float(row.get("area_m2", 0))
            poly = str(row.get("polymer_type", "Marine Debris (Plastic)"))
            target_clusters.append({
                "cluster_id": cid,
                "centroid_lat": c_lat,
                "centroid_lon": c_lon,
                "area_m2": area,
                "polymer_type": poly,
            })

    # If no clusters found from detections, fall back to sources
    if not target_clusters and sources:
        for idx, s in enumerate(sources):
            cid = s.get("cluster_id", idx)
            cent = s.get("source_centroid", (0, 0))
            target_clusters.append({
                "cluster_id": cid,
                "centroid_lat": cent[1] if len(cent) == 2 else 0.0,
                "centroid_lon": cent[0] if len(cent) == 2 else 0.0,
                "area_m2": 0.0,
                "polymer_type": "Marine Debris (Plastic)",
            })

    if not target_clusters and not sources:
        logger.info("No source regions or debris clusters to attribute — writing empty report")
        with open(report_path, "w") as fh:
            json.dump([], fh, indent=2)
        return report_path

    # Map sources by cluster_id
    sources_by_cluster: Dict[str, Dict[str, Any]] = {}
    for s in (sources or []):
        cid = s.get("cluster_id")
        if cid is not None:
            sources_by_cluster[str(cid)] = s

    # Pre-query GFW API for the scene bounding box to cache results and avoid rate limits
    scene_vessel_records = []
    if gfw_token and not disable_gfw and target_clusters:
        try:
            all_lons = [c["centroid_lon"] for c in target_clusters if c.get("centroid_lon") is not None]
            all_lats = [c["centroid_lat"] for c in target_clusters if c.get("centroid_lat") is not None]
            if all_lons and all_lats:
                pad = debris_search_radius_km / 111.0 + 0.1
                scene_bbox = (
                    max(-180.0, min(all_lons) - pad),
                    max(-85.0, min(all_lats) - pad),
                    min(180.0, max(all_lons) + pad),
                    min(85.0, max(all_lats) + pad),
                )
                logger.info("Querying GFW API for scene bounding box %s", scene_bbox)
                scene_vessel_records, gfw_unproc = query_gfw_vessels_in_bbox(
                    scene_bbox, date_start, date_end, gfw_token
                )
                if gfw_unproc:
                    disable_gfw = True
                logger.info("GFW API returned %d vessel records for scene", len(scene_vessel_records))
        except Exception as exc:
            logger.warning("Scene-level GFW query failed: %s", exc)

    report_entries = []

    # Process each cluster
    for rank, cluster in enumerate(target_clusters):
        cid = cluster["cluster_id"]
        c_lat = cluster["centroid_lat"]
        c_lon = cluster["centroid_lon"]
        source = sources_by_cluster.get(str(cid))

        # Query / match ships near the debris cluster
        ships_info = find_ships_near_debris(
            cluster_lat=c_lat,
            cluster_lon=c_lon,
            date_start=date_start,
            date_end=date_end,
            gfw_token=None if disable_gfw else gfw_token,
            search_radius_km=debris_search_radius_km,
            cached_vessel_records=scene_vessel_records if scene_vessel_records else None,
        )

        nearest_ship = ships_info.get("nearest_ship")
        nearby_vessels = ships_info.get("nearby_vessels", [])
        nearby_vessel_count = ships_info.get("nearby_vessel_count", 0)

        # If we have a backtracked source for this cluster, score the source
        src_bbox = source.get("source_bbox") if source else None
        if src_bbox:
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
            src_centroid = source.get("source_centroid")
            src_prob = source.get("source_probability")
            days_to_src = source.get("days_to_source")
            vessel_ids = scores.get("fishing", {}).get("vessel_ids", [])
        else:
            # Cluster was not backtracked or reached open ocean
            # Attribute based on nearby ships or coastal proximity
            if nearest_ship and "FISHING" in nearest_ship.get("vessel_type", "").upper():
                primary_source = "fishing"
                attr_score = min(0.9, 0.5 + (0.4 if nearest_ship["distance_km"] < 20 else 0.2))
            elif nearest_ship:
                primary_source = "shipping"
                attr_score = min(0.85, 0.4 + (0.4 if nearest_ship["distance_km"] < 20 else 0.2))
            else:
                primary_source = "coastal"
                attr_score = 0.35

            attribution = {
                "source_type": primary_source,
                "attribution_score": attr_score,
                "fishing_score": 0.4 if primary_source == "fishing" else 0.1,
                "industrial_score": 0.2,
                "shipping_score": 0.5 if primary_source == "shipping" else 0.1,
                "river_score": 0.1,
            }
            explanation = f"Attributed to {primary_source} based on local vessel traffic and proximity."
            src_centroid = [c_lon, c_lat]
            src_prob = 1.0
            days_to_src = 0
            vessel_ids = [v["vessel_id"] for v in nearby_vessels[:5] if v.get("vessel_id")]

        # Prepend nearby ship details to the explanation
        if nearest_ship:
            ship_name = nearest_ship.get("ship_name", "Unknown")
            ship_type = nearest_ship.get("vessel_type", "Vessel")
            ship_mmsi = nearest_ship.get("mmsi", "")
            ship_flag = nearest_ship.get("flag", "")
            ship_dist = nearest_ship.get("distance_km", 0)
            id_parts = []
            if ship_mmsi:
                id_parts.append(f"MMSI: {ship_mmsi}")
            if ship_flag:
                id_parts.append(f"Flag: {ship_flag}")
            id_str = f" ({', '.join(id_parts)})" if id_parts else ""
            ship_summary = (
                f"GFW API identified nearby vessel '{ship_name}'{id_str} ({ship_type}) "
                f"{ship_dist:.1f} km from this debris cluster. "
            )
            explanation = f"{ship_summary}{explanation}".strip()

        loc_centroid = src_centroid if src_centroid else (c_lon, c_lat)
        loc_name = _get_location_name(loc_centroid)
        country_name = _get_country_name(loc_centroid)

        entry = {
            "debris_cluster_id": cid if cid is not None else rank,
            "source_rank": rank + 1,
            "source_type": attribution["source_type"],
            "location_name": loc_name,
            "country": country_name,
            "attribution_score": attribution["attribution_score"],
            "confidence": "high" if attribution["attribution_score"] > 0.6
                         else "moderate" if attribution["attribution_score"] > 0.3
                         else "low",
            "explanation": explanation,
            "source_centroid": src_centroid,
            "source_bbox": src_bbox,
            "source_probability": src_prob,
            "days_to_source": days_to_src,
            "fishing_score": attribution.get("fishing_score", 0),
            "industrial_score": attribution.get("industrial_score", 0),
            "shipping_score": attribution.get("shipping_score", 0),
            "river_score": attribution.get("river_score", 0),
            "vessel_ids": vessel_ids,
            "cluster_centroid": [c_lon, c_lat],
            "cluster_area_m2": cluster.get("area_m2", 0),
            "polymer_type": cluster.get("polymer_type", "Marine Debris (Plastic)"),
            "nearest_ship": nearest_ship,
            "nearby_vessels": nearby_vessels,
            "nearby_vessel_count": nearby_vessel_count,
        }
        report_entries.append(entry)

    # Sort by attribution score descending
    report_entries.sort(key=lambda x: x["attribution_score"], reverse=True)

    with open(report_path, "w") as fh:
        json.dump(report_entries, fh, indent=2, default=str)

    logger.info(
        "[bold green]Stage 6 complete[/] — %d clusters attributed, top: %s (%.1f%%)",
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
