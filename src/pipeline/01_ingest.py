"""
Plastic-Ledger — Stage 1: Satellite Data Ingestion
====================================================
Downloads Sentinel-2 L2A imagery from the Copernicus Data Space Ecosystem
using their STAC API.

Usage (standalone):
    python -m pipeline.01_ingest \\
        --bbox "80.0,8.0,82.0,10.0" \\
        --start_date 2024-01-01 \\
        --end_date 2024-01-31 \\
        --output_dir data/raw

Dependencies: pystac-client, requests, rasterio, shapely
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds as window_from_bounds
from pystac_client import Client
from shapely.geometry import box, shape as shapely_shape

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.geo_utils import retry_request
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STAC_ENDPOINT = "https://catalogue.dataspace.copernicus.eu/stac"
REQUIRED_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
COLLECTION = "sentinel-2-l2a"
# Maximum time difference (hours) for tiles to be considered from the same pass
SAME_PASS_TOLERANCE_HOURS = 1


# ─────────────────────────────────────────────
# STAC SEARCH
# ─────────────────────────────────────────────
def search_scenes(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cloud_cover_max: int = 20,
    max_items: int = 50,
) -> List[Dict[str, Any]]:
    """Search the Copernicus STAC catalogue for Sentinel-2 L2A scenes.

    When a bounding box spans multiple Sentinel-2 granules, this function
    returns **all** tiles from the most recent satellite pass so they can
    be mosaicked together for full spatial coverage.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        date_start: ISO date string (e.g. ``2024-01-01``).
        date_end: ISO date string.
        cloud_cover_max: Maximum cloud cover percentage (0–100).
        max_items: Cap on number of items returned from STAC.

    Returns:
        List of STAC item dicts with id, datetime, cloud_cover, assets, bbox.
        All returned scenes belong to the same satellite pass (within
        :data:`SAME_PASS_TOLERANCE_HOURS` of each other).

    Raises:
        ConnectionError: If the STAC endpoint is unreachable.
        RuntimeError: If no scenes are found matching the criteria.
    """
    logger.info(
        "Searching STAC: bbox=%s, dates=%s to %s, cloud≤%d%%",
        bbox, date_start, date_end, cloud_cover_max,
    )

    catalog = Client.open(STAC_ENDPOINT)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        sortby=[{"field": "datetime", "direction": "desc"}],
        max_items=max_items,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L2A scenes found for bbox={bbox}, "
            f"dates={date_start}–{date_end}, cloud<{cloud_cover_max}%. "
            "Try expanding the date range or increasing cloud_cover_max."
        )

    logger.info("Found [bold cyan]%d[/] total scenes from STAC", len(items))

    results = []
    for item in items:
        results.append({
            "id": item.id,
            "datetime": str(item.datetime),
            "cloud_cover": item.properties.get("eo:cloud_cover", None),
            "bbox": item.bbox,
            "assets": {
                k: v.extra_fields.get("alternate", {}).get("https", {}).get("href", v.href)
                for k, v in item.assets.items()
            },
            "geometry": item.geometry,
        })

    # Sort by datetime descending (newest first)
    results.sort(key=lambda x: x.get("datetime", ""), reverse=True)

    # Select the best group of tiles from the same satellite pass
    best_group = _select_best_scene_group(results, bbox)
    logger.info(
        "Selected [bold cyan]%d[/] tile(s) from the most recent pass",
        len(best_group),
    )
    for scene in best_group:
        logger.info("  Tile: %s  (datetime=%s)", scene["id"], scene["datetime"])

    return best_group


def _select_best_scene_group(
    scenes: List[Dict[str, Any]],
    bbox: Tuple[float, float, float, float],
) -> List[Dict[str, Any]]:
    """Group scenes by satellite pass and return the most recent group.

    Tiles captured within :data:`SAME_PASS_TOLERANCE_HOURS` of each other
    are considered part of the same orbital pass.  The newest pass whose
    tiles collectively provide the best spatial coverage of *bbox* is
    selected.

    Args:
        scenes: Sorted list (newest-first) of scene dicts from STAC.
        bbox: User-requested ``(lon_min, lat_min, lon_max, lat_max)``.

    Returns:
        List of scene dicts belonging to the best group.
    """
    if len(scenes) <= 1:
        return scenes

    tolerance = timedelta(hours=SAME_PASS_TOLERANCE_HOURS)
    user_box = box(*bbox)

    # Build groups of scenes from the same pass
    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = [scenes[0]]
    current_anchor = _parse_datetime(scenes[0]["datetime"])

    for scene in scenes[1:]:
        scene_dt = _parse_datetime(scene["datetime"])
        if current_anchor is not None and scene_dt is not None:
            if abs(current_anchor - scene_dt) <= tolerance:
                current_group.append(scene)
                continue
        # New group
        groups.append(current_group)
        current_group = [scene]
        current_anchor = scene_dt

    groups.append(current_group)

    # Score each group by how much of the user bbox it covers
    best_group = groups[0]
    best_coverage = 0.0

    for group in groups:
        coverage = _compute_bbox_coverage(group, user_box)
        if coverage > best_coverage:
            best_coverage = coverage
            best_group = group
        # If we already have near-perfect coverage, no need to look further
        if best_coverage >= 0.99:
            break

    logger.info(
        "Best pass covers %.1f%% of the requested bbox (%d group(s) evaluated)",
        best_coverage * 100,
        len(groups),
    )
    return best_group


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse an ISO datetime string, returning None on failure."""
    try:
        # Handle timezone-aware strings with 'Z' suffix
        clean = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except (ValueError, AttributeError):
        return None


def _compute_bbox_coverage(
    group: List[Dict[str, Any]],
    user_box: "shapely.geometry.Polygon",
) -> float:
    """Compute the fraction of *user_box* covered by the union of scene footprints.

    Args:
        group: List of scene dicts (must have ``geometry`` or ``bbox`` keys).
        user_box: Shapely polygon of the user's requested area.

    Returns:
        Float in ``[0, 1]`` representing fractional coverage.
    """
    from shapely.ops import unary_union

    footprints = []
    for scene in group:
        if scene.get("geometry"):
            try:
                footprints.append(shapely_shape(scene["geometry"]))
            except Exception:
                pass
        elif scene.get("bbox"):
            footprints.append(box(*scene["bbox"]))

    if not footprints:
        return 0.0

    combined = unary_union(footprints)
    intersection = combined.intersection(user_box)
    if user_box.area <= 0:
        return 0.0
    return intersection.area / user_box.area


# ─────────────────────────────────────────────
# MOSAICKING
# ─────────────────────────────────────────────
def mosaic_scenes(
    scene_dirs: List[Path],
    bbox: Tuple[float, float, float, float],
    bands: List[str] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Dict[str, str]]:
    """Mosaic multiple Sentinel-2 tiles into a single set of band GeoTIFFs.

    When a bounding box spans multiple Sentinel-2 granules, this function
    merges the overlapping tiles per band using :func:`rasterio.merge.merge`,
    reprojects to a common CRS if needed, and crops the result to *bbox*.

    Args:
        scene_dirs: List of directories, each containing per-band GeoTIFFs.
        bbox: ``(lon_min, lat_min, lon_max, lat_max)`` in WGS84 (EPSG:4326).
        bands: Band names to mosaic. Defaults to :data:`REQUIRED_BANDS`.
        output_dir: Directory for mosaicked output. Defaults to first
            scene dir's parent / ``mosaic``.

    Returns:
        Tuple of (mosaic directory path, dict mapping band name → file path).

    Raises:
        FileNotFoundError: If no band files are found across any scene.
    """
    bands = bands or REQUIRED_BANDS

    if output_dir is None:
        output_dir = scene_dirs[0].parent / "mosaic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(scene_dirs) == 1:
        # Single tile — no mosaicking needed, just symlink/copy
        logger.info("Single tile — skipping mosaic step")
        return scene_dirs[0], _list_band_paths(scene_dirs[0], bands)

    logger.info(
        "Mosaicking [bold cyan]%d[/] tiles into single raster (bbox=%s)",
        len(scene_dirs), bbox,
    )

    # Determine the target CRS from the majority of tiles
    target_crs = _determine_target_crs(scene_dirs, bands)
    logger.info("Target mosaic CRS: %s", target_crs)

    band_paths = {}
    for band_name in bands:
        # Collect all tile rasters for this band
        tile_paths = []
        for scene_dir in scene_dirs:
            candidates = [
                scene_dir / f"{band_name}.tif",
                scene_dir / f"{band_name.lower()}.tif",
                scene_dir / f"{band_name}_10m.tif",
                scene_dir / f"{band_name}_20m.tif",
            ]
            for cand in candidates:
                if cand.exists():
                    tile_paths.append(cand)
                    break

        if not tile_paths:
            logger.warning(
                "Band %s not found in any tile — will be zero-padded",
                band_name,
            )
            continue

        if len(tile_paths) == 1:
            # Only one tile has this band — crop to bbox directly
            out_path = output_dir / f"{band_name}.tif"
            _crop_to_bbox(tile_paths[0], out_path, bbox, target_crs)
            band_paths[band_name] = str(out_path)
            continue

        # Reproject tiles to common CRS if needed, then merge
        reprojected = []
        for tp in tile_paths:
            reprojected.append(
                _ensure_crs(tp, target_crs, output_dir / f"_tmp_{tp.parent.name}_{tp.name}")
            )

        # Merge tiles
        datasets = [rasterio.open(rp) for rp in reprojected]
        try:
            mosaic_arr, mosaic_transform = rasterio_merge(
                datasets,
                method="first",  # Use first valid pixel
            )
        finally:
            for ds in datasets:
                ds.close()

        # Write full mosaic to a temporary file
        tmp_mosaic = output_dir / f"_tmp_mosaic_{band_name}.tif"
        profile = {
            "driver": "GTiff",
            "dtype": mosaic_arr.dtype,
            "count": mosaic_arr.shape[0],
            "height": mosaic_arr.shape[1],
            "width": mosaic_arr.shape[2],
            "crs": target_crs,
            "transform": mosaic_transform,
            "compress": "lzw",
        }
        with rasterio.open(tmp_mosaic, "w", **profile) as dst:
            dst.write(mosaic_arr)

        # Crop to bbox
        out_path = output_dir / f"{band_name}.tif"
        _crop_to_bbox(tmp_mosaic, out_path, bbox, target_crs)
        band_paths[band_name] = str(out_path)

        # Clean up temporary files
        if tmp_mosaic.exists():
            tmp_mosaic.unlink()
        for rp in reprojected:
            if rp != tile_paths[0] and rp.exists() and "_tmp_" in rp.name:
                rp.unlink()

    if not band_paths:
        raise FileNotFoundError(
            f"No band files found across {len(scene_dirs)} scene directories"
        )

    logger.info(
        "Mosaic complete: %d bands written to %s", len(band_paths), output_dir,
    )
    return output_dir, band_paths


def _list_band_paths(
    scene_dir: Path, bands: List[str],
) -> Dict[str, str]:
    """List available band file paths in a scene directory."""
    result = {}
    for band_name in bands:
        for cand in [
            scene_dir / f"{band_name}.tif",
            scene_dir / f"{band_name.lower()}.tif",
        ]:
            if cand.exists():
                result[band_name] = str(cand)
                break
    return result


def _determine_target_crs(
    scene_dirs: List[Path],
    bands: List[str],
) -> rasterio.crs.CRS:
    """Pick the most common CRS across tiles as the mosaic target.

    Falls back to the CRS of the first readable tile.
    """
    from collections import Counter

    crs_counts: Counter = Counter()
    first_crs = None

    for scene_dir in scene_dirs:
        for band_name in bands:
            cand = scene_dir / f"{band_name}.tif"
            if not cand.exists():
                continue
            try:
                with rasterio.open(cand) as src:
                    if src.crs is not None:
                        crs_counts[str(src.crs)] += 1
                        if first_crs is None:
                            first_crs = src.crs
            except Exception:
                continue
            break  # Only need one band per scene to get the CRS

    if not crs_counts:
        # Fallback to WGS84 UTM estimated from centroid
        return rasterio.crs.CRS.from_epsg(4326)

    most_common_str = crs_counts.most_common(1)[0][0]
    return rasterio.crs.CRS.from_string(most_common_str)


def _ensure_crs(
    src_path: Path,
    target_crs: rasterio.crs.CRS,
    tmp_path: Path,
) -> Path:
    """Reproject a raster to *target_crs* if it differs. Returns path to use."""
    with rasterio.open(src_path) as src:
        if src.crs is not None and src.crs == target_crs:
            return src_path

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
        )

        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

    return tmp_path


def _crop_to_bbox(
    src_path: Path,
    dst_path: Path,
    bbox: Tuple[float, float, float, float],
    target_crs: rasterio.crs.CRS,
) -> Path:
    """Crop a raster to *bbox* (in WGS84), reprojecting bbox coords if needed.

    Args:
        src_path: Input raster path.
        dst_path: Output cropped raster path.
        bbox: ``(lon_min, lat_min, lon_max, lat_max)`` in EPSG:4326.
        target_crs: CRS of the source raster.

    Returns:
        Path to the cropped output file.
    """
    from rasterio.warp import transform_bounds

    with rasterio.open(src_path) as src:
        # Transform bbox from WGS84 to the raster's CRS
        src_crs = src.crs or target_crs
        if str(src_crs) != "EPSG:4326":
            cropped_bounds = transform_bounds(
                rasterio.crs.CRS.from_epsg(4326),
                src_crs,
                *bbox,
            )
        else:
            cropped_bounds = bbox

        # Compute the window for the bbox
        try:
            window = window_from_bounds(*cropped_bounds, transform=src.transform)
        except Exception:
            # If bbox is outside the raster extent, just copy the file
            logger.warning(
                "BBox does not intersect raster %s — copying full raster",
                src_path.name,
            )
            import shutil
            shutil.copy2(src_path, dst_path)
            return dst_path

        # Clamp window to raster dimensions
        try:
            window = window.intersection(rasterio.windows.Window(
                0, 0, src.width, src.height,
            ))
        except Exception:
            # Window doesn't intersect the raster at all
            logger.warning(
                "BBox window does not intersect raster %s — copying full raster",
                src_path.name,
            )
            import shutil
            shutil.copy2(src_path, dst_path)
            return dst_path

        if window.width <= 0 or window.height <= 0:
            logger.warning(
                "Empty intersection for %s — copying full raster",
                src_path.name,
            )
            import shutil
            shutil.copy2(src_path, dst_path)
            return dst_path

        # Read and write cropped data
        cropped_transform = rasterio.windows.transform(window, src.transform)
        data = src.read(window=window)

        profile = src.profile.copy()
        profile.update(
            width=int(window.width),
            height=int(window.height),
            transform=cropped_transform,
            compress="lzw",
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

    return dst_path


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
def _download_file(url: str, dest: Path, session: requests.Session) -> Path:
    """Download a single file with retry logic.

    Args:
        url: Direct download URL.
        dest: Destination file path.
        session: Authenticated requests session.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.HTTPError: If download fails after retries.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("  Already downloaded: %s", dest.name)
        return dest

    @retry_request
    def _do_download():
        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)

    _do_download()
    logger.info("  Downloaded: %s", dest.name)
    return dest


def download_scene(
    scene: Dict[str, Any],
    output_dir: Path,
    bands: List[str] = None,
    session: Optional[requests.Session] = None,
) -> Tuple[Path, Dict[str, str]]:
    """Download band assets for a single STAC scene.

    Args:
        scene: Scene dict from :func:`search_scenes`.
        output_dir: Base output directory (scene subfolder is created).
        bands: Band names to download.  Defaults to :data:`REQUIRED_BANDS`.
        session: Optional pre-authenticated :class:`requests.Session`.

    Returns:
        Tuple of (scene directory path, dict mapping band name → file path).

    Raises:
        KeyError: If a required band is missing from the scene assets.
    """
    bands = bands or REQUIRED_BANDS
    scene_id = scene["id"]
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = requests.Session()

    logger.info("Downloading scene [bold]%s[/] (%d bands)", scene_id, len(bands))

    band_paths = {}
    for band_name in bands:
        # STAC assets may be keyed as "B02_10m", "B02", etc.
        asset_key = None
        for key in scene["assets"]:
            if band_name.lower() in key.lower():
                asset_key = key
                break

        if asset_key is None:
            logger.warning(
                "Band %s not found in scene %s assets. Available: %s",
                band_name, scene_id, list(scene["assets"].keys()),
            )
            continue

        url = scene["assets"][asset_key]
        dest = scene_dir / f"{band_name}.tif"

        try:
            _download_file(url, dest, session)
            band_paths[band_name] = str(dest)
        except Exception as exc:
            logger.error("Failed to download %s: %s", band_name, exc)

    # Save scene metadata
    meta_path = scene_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(scene, fh, indent=2, default=str)

    return scene_dir, band_paths


# ─────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────
def get_copernicus_session(username: str, password: str) -> requests.Session:
    """Authenticate with Copernicus Data Space and return an authorized session.

    Args:
        username: Copernicus account email.
        password: Copernicus account password.

    Returns:
        :class:`requests.Session` with bearer token set.

    Raises:
        RuntimeError: If authentication fails.
    """
    token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/"
        "CDSE/protocol/openid-connect/token"
    )

    @retry_request
    def _get_token():
        resp = requests.post(
            token_url,
            data={
                "grant_type": "password",
                "username": username,
                "password": password,
                "client_id": "cdse-public",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    token = _get_token()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    logger.info("Authenticated with Copernicus Data Space")
    return session


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cloud_cover_max: int = 20,
    output_dir: Union[str, Path] = "data/raw",
    config: Optional[Dict] = None,
) -> Tuple[List[Path], List[Dict]]:
    """Run the full ingestion stage.

    When the bounding box spans multiple Sentinel-2 granules, all
    overlapping tiles from the same satellite pass are downloaded and
    mosaicked into a single set of band GeoTIFFs cropped to *bbox*.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        date_start: Start date ISO string.
        date_end: End date ISO string.
        cloud_cover_max: Max cloud cover %.
        output_dir: Root output directory for raw scenes.
        config: Optional config dict (for API credentials).

    Returns:
        Tuple of (list of scene directory paths, list of scene metadata dicts).
        When multiple tiles are mosaicked, the list contains a single
        mosaic directory.

    Raises:
        RuntimeError: If no scenes found or all downloads fail.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search — returns all tiles from the best satellite pass
    scenes = search_scenes(bbox, date_start, date_end, cloud_cover_max)

    # Authenticate (if credentials available)
    session = None
    if config and config.get("apis", {}).get("copernicus_username"):
        try:
            session = get_copernicus_session(
                config["apis"]["copernicus_username"],
                config["apis"]["copernicus_password"],
            )
        except Exception as exc:
            logger.warning("Auth failed (%s) — downloading without auth", exc)

    # Download each tile
    tile_dirs = []
    scene_metas = []
    for scene in scenes:
        try:
            scene_dir, band_paths = download_scene(
                scene, output_dir, session=session,
            )
            tile_dirs.append(scene_dir)
            scene_metas.append(scene)
            logger.info(
                "Tile %s: %d bands downloaded", scene["id"], len(band_paths),
            )
        except Exception as exc:
            logger.error("Skipping tile %s: %s", scene["id"], exc)

    if not tile_dirs:
        raise RuntimeError("All tile downloads failed.")

    # Mosaic tiles if multiple were downloaded
    if len(tile_dirs) > 1:
        logger.info(
            "Mosaicking %d tiles for full bbox coverage", len(tile_dirs),
        )
        mosaic_dir = output_dir / "mosaic"
        mosaic_dir, mosaic_band_paths = mosaic_scenes(
            tile_dirs, bbox, output_dir=mosaic_dir,
        )
        # Save combined metadata for the mosaic
        mosaic_meta = {
            "id": "mosaic_" + "_".join(s["id"] for s in scene_metas),
            "datetime": scene_metas[0].get("datetime", ""),
            "cloud_cover": max(
                (s.get("cloud_cover", 0) or 0) for s in scene_metas
            ),
            "bbox": list(bbox),
            "source_tiles": [s["id"] for s in scene_metas],
            "assets": mosaic_band_paths,
            "geometry": {"type": "Polygon", "coordinates": [[
                [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                [bbox[2], bbox[3]], [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]]},
        }
        meta_path = mosaic_dir / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(mosaic_meta, fh, indent=2, default=str)

        scene_dirs = [mosaic_dir]
        scene_metas_out = [mosaic_meta]
    else:
        scene_dirs = tile_dirs
        scene_metas_out = scene_metas

    # Save run metadata
    run_meta = {
        "bbox": list(bbox),
        "date_start": date_start,
        "date_end": date_end,
        "cloud_cover_max": cloud_cover_max,
        "scenes": [s["id"] for s in scene_metas],
        "tile_count": len(tile_dirs),
        "mosaicked": len(tile_dirs) > 1,
        "scene_paths": [str(d) for d in scene_dirs],
    }
    with open(output_dir / "ingest_metadata.json", "w") as fh:
        json.dump(run_meta, fh, indent=2)

    logger.info(
        "[bold green]Stage 1 complete[/] — %d tile(s) → %d output scene(s) in %s",
        len(tile_dirs), len(scene_dirs), output_dir,
    )
    return scene_dirs, scene_metas_out


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Download Sentinel-2 L2A scenes from Copernicus STAC",
    )
    parser.add_argument(
        "--bbox", type=str, required=True,
        help="Bounding box as 'lon_min,lat_min,lon_max,lat_max'",
    )
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--cloud_cover", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    assert len(bbox) == 4, "bbox must have exactly 4 values"

    config = load_config(args.config)
    scene_dirs, scene_metas = run(
        bbox=bbox,
        date_start=args.start_date,
        date_end=args.end_date,
        cloud_cover_max=args.cloud_cover,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"\nDownloaded {len(scene_dirs)} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
