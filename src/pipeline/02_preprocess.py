"""
Plastic-Ledger — Stage 2: Preprocessing
==========================================
Converts raw Sentinel-2 scenes into model-ready 256×256 patches
with 11-band reordering, z-score normalization, and overlap tiling.

Usage (standalone):
    python -m pipeline.02_preprocess --scene_dir data/raw/SCENE_ID --output_dir data/processed

Dependencies: rasterio, numpy, shapely
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
# Band mapping: position index → band name
# The model expects 11 bands in this exact order:
# [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
MODEL_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                    "B08", "B8A", "B11", "B12"]
# Bands we actually download (now all 11)
AVAILABLE_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

NUM_BANDS = 11
PATCH_SIZE = 256
OVERLAP = 32

# Per-band normalization statistics from the MARIDA dataset
BAND_MEANS = np.array([0.057, 0.054, 0.046, 0.036, 0.033,
                       0.041, 0.049, 0.043, 0.050, 0.031, 0.019],
                      dtype=np.float32)
BAND_STDS = np.array([0.010, 0.010, 0.013, 0.010, 0.012,
                      0.020, 0.030, 0.020, 0.030, 0.020, 0.013],
                     dtype=np.float32)


# ─────────────────────────────────────────────
# BAND LOADING & REORDERING
# ─────────────────────────────────────────────
def load_and_reorder_bands(
    scene_dir: Path,
) -> Tuple[np.ndarray, dict, Affine, Any]:
    """Load individual band GeoTIFFs and compose an 11-band array.

    Missing bands (B01, B06, B07) are zero-padded.  All bands are
    resampled to match the highest-resolution band's grid.

    Args:
        scene_dir: Directory containing per-band ``*.tif`` files
            (e.g. ``B02.tif``, ``B03.tif``, …).

    Returns:
        Tuple of:
        - ``image``: ``(11, H, W)`` float32 array in model band order
        - ``profile``: rasterio profile dict (from the first band)
        - ``transform``: :class:`Affine` geotransform
        - ``crs``: coordinate reference system

    Raises:
        FileNotFoundError: If *scene_dir* contains none of the required bands.
    """
    scene_dir = Path(scene_dir)

    # Try to load the scene as a single multi-band GeoTIFF first
    multiband_candidates = list(scene_dir.glob("*.tif"))
    single_multiband = [f for f in multiband_candidates
                        if not any(b in f.stem.upper() for b in AVAILABLE_BANDS)]

    if len(single_multiband) == 1:
        return _load_multiband_tif(single_multiband[0])

    # Otherwise load individual band files
    return _load_individual_bands(scene_dir)


def _load_multiband_tif(
    tif_path: Path,
) -> Tuple[np.ndarray, dict, Affine, Any]:
    """Load a single multi-band GeoTIFF that already has all bands.

    Args:
        tif_path: Path to the multi-band GeoTIFF.

    Returns:
        Same as :func:`load_and_reorder_bands`.
    """
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    n_bands = data.shape[0]

    if n_bands == NUM_BANDS:
        # Already 11 bands — assume correct order
        return data, profile, transform, crs

    if n_bands == 8:
        # Legacy 8-band fallback: pad to 11 bands
        image = np.zeros((11, *data.shape[1:]), dtype=np.float32)
        mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8, 6: 9, 7: 10}
        for src_idx, dst_idx in mapping.items():
            image[dst_idx] = data[src_idx]
        return image, profile, transform, crs

    # Try to use first NUM_BANDS bands or pad
    if n_bands > NUM_BANDS:
        data = data[:NUM_BANDS]
    else:
        pad = np.zeros((NUM_BANDS - n_bands, *data.shape[1:]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)

    return data, profile, transform, crs


def _load_individual_bands(
    scene_dir: Path,
) -> Tuple[np.ndarray, dict, Affine, Any]:
    """Load per-band TIF files and assemble into an 11-band array.

    Args:
        scene_dir: Directory containing per-band GeoTIFFs.

    Returns:
        Same as :func:`load_and_reorder_bands`.

    Raises:
        FileNotFoundError: If no band files can be found.
    """
    band_data = {}
    ref_profile = None
    ref_shape = None

    for band_name in AVAILABLE_BANDS:
        # Try multiple naming patterns
        candidates = [
            scene_dir / f"{band_name}.tif",
            scene_dir / f"{band_name.lower()}.tif",
            scene_dir / f"{band_name}_10m.tif",
            scene_dir / f"{band_name}_20m.tif",
        ]
        # Also glob for any file containing the band name
        candidates.extend(scene_dir.glob(f"*{band_name}*.tif"))

        loaded = False
        for cand in candidates:
            if cand.exists():
                with rasterio.open(cand) as src:
                    arr = src.read(1).astype(np.float32)
                    band_data[band_name] = arr
                    
                    # Track the highest resolution profile
                    if ref_shape is None or (arr.shape[0] * arr.shape[1] > ref_shape[0] * ref_shape[1]):
                        ref_profile = src.profile.copy()
                        ref_shape = (src.height, src.width)
                loaded = True
                break

        if not loaded:
            logger.warning("Band %s not found in %s — will be zero-padded", band_name, scene_dir)

    if not band_data:
        raise FileNotFoundError(f"No band files found in {scene_dir}")

    # Determine reference shape (use the largest resolution)
    if ref_shape is None:
        raise FileNotFoundError("Could not determine reference shape")

    # Assemble all available bands in correct order
    available_array = np.zeros((len(AVAILABLE_BANDS), *ref_shape), dtype=np.float32)
    for i, band_name in enumerate(AVAILABLE_BANDS):
        if band_name in band_data:
            arr = band_data[band_name]
            # Resize if needed (upsample 60m/20m to 10m)
            if arr.shape != ref_shape:
                from scipy.ndimage import zoom
                zoom_factors = (ref_shape[0] / arr.shape[0], ref_shape[1] / arr.shape[1])
                arr = zoom(arr, zoom_factors, order=1)
            available_array[i] = arr

    return available_array, ref_profile, ref_profile["transform"], ref_profile.get("crs")





# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
def normalize_scene(
    image: np.ndarray,
    band_means: np.ndarray = None,
    band_stds: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip and z-score normalize an 11-band scene.

    Args:
        image: ``(11, H, W)`` raw Sentinel-2 DN values (0-10000 range).
        band_means: Per-band means (defaults to MARIDA stats).
        band_stds: Per-band stds (defaults to MARIDA stats).

    Returns:
        Tuple of (normalized image, nodata boolean mask of shape ``(H, W)``).
    """
    if band_means is None:
        band_means = BAND_MEANS
    if band_stds is None:
        band_stds = BAND_STDS

    # Detect nodata pixels (all-zero across bands)
    nodata_mask = (image.sum(axis=0) == 0)

    # Sentinel-2 Collection 1 (Baseline >= 04.00) introduced a +1000 DN offset
    # All historical data on Copernicus Data Space was reprocessed to include this.
    # We must subtract 1000 to match the older MARIDA dataset distribution.
    # Subtract 1000 only from valid (non-nodata) pixels to avoid making nodata negative.
    image = np.where(~nodata_mask, image - 1000.0, 0.0)
    
    # Prevent negative reflectance values
    image = np.clip(image, 0.0, None)

    # CRITICAL: Convert DN to reflectance (Sentinel-2 L2A is scaled by 10000)
    # This must match the SegFormer training preprocessing
    image = image / 10000.0

    # [REMOVED]: Do not clip to 0.5, as this destroys bright objects (clouds, plastic).
    # image = np.clip(image, 0.0001, 0.5)

    # Z-score normalize per band using MARIDA training statistics
    # [REMOVED]: The new SegFormer model was trained purely on reflectance values without Z-scoring.
    # for b in range(NUM_BANDS):
    #     image[b] = (image[b] - band_means[b]) / (band_stds[b] + 1e-6)

    # Reset nodata pixels to 0 after normalization
    image[:, nodata_mask] = 0.0

    # Safety clip to prevent numerical issues
    image = np.clip(image, 0.0, 1.0)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

    return image, nodata_mask


def compute_land_mask(
    image: np.ndarray,
    ndwi_threshold: float = 0.15,
    mndwi_threshold: float = 0.05,
    dilate_pixels: int = 3,
) -> np.ndarray:
    """Compute boolean land mask using NDWI and MNDWI spectral indices with shoreline dilation.

    Args:
        image: ``(11, H, W)`` Sentinel-2 array (reflectance range 0..1).
        ndwi_threshold: Threshold above which pixel is classified as water.
        mndwi_threshold: Threshold above which pixel is classified as water.
        dilate_pixels: Number of pixels to dilate land mask into coastline boundary.

    Returns:
        Boolean mask of shape ``(H, W)`` where True = land/shoreline, False = water.
    """
    from scipy.ndimage import binary_dilation

    b03 = image[2].astype(np.float32)  # Green
    b08 = image[7].astype(np.float32)  # NIR
    b11 = image[9].astype(np.float32)  # SWIR 1

    # Avoid division by zero
    ndwi_denom = b03 + b08
    ndwi_denom = np.where(ndwi_denom == 0, 1e-6, ndwi_denom)
    ndwi = (b03 - b08) / ndwi_denom

    mndwi_denom = b03 + b11
    mndwi_denom = np.where(mndwi_denom == 0, 1e-6, mndwi_denom)
    mndwi = (b03 - b11) / mndwi_denom

    # Land pixels: pixels where NDWI <= threshold OR MNDWI <= threshold
    is_water = (ndwi > ndwi_threshold) & (mndwi > mndwi_threshold)
    land_mask = ~is_water

    if dilate_pixels > 0:
        land_mask = binary_dilation(land_mask, iterations=dilate_pixels)

    return land_mask


# ─────────────────────────────────────────────
# TILING
# ─────────────────────────────────────────────
def tile_scene(
    image: np.ndarray,
    patch_size: int = PATCH_SIZE,
    overlap: int = OVERLAP,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Tile a normalized scene into overlapping patches.

    Args:
        image: ``(bands, H, W)`` normalized array.
        patch_size: Patch edge length in pixels.
        overlap: Overlap between adjacent patches in pixels.

    Returns:
        Tuple of:
        - list of ``(bands, patch_size, patch_size)`` patch arrays
        - list of patch info dicts with keys ``row``, ``col``,
          ``row_start``, ``col_start``
    """
    _, h, w = image.shape
    stride = patch_size - overlap

    patches = []
    patch_infos = []

    row = 0
    r_idx = 0
    while row < h:
        col = 0
        c_idx = 0
        while col < w:
            # Handle edge patches — pad if needed
            row_end = min(row + patch_size, h)
            col_end = min(col + patch_size, w)

            patch = np.zeros((image.shape[0], patch_size, patch_size),
                             dtype=np.float32)
            actual_h = row_end - row
            actual_w = col_end - col
            patch[:, :actual_h, :actual_w] = image[:, row:row_end, col:col_end]

            patches.append(patch)
            patch_infos.append({
                "row": r_idx,
                "col": c_idx,
                "row_start": row,
                "col_start": col,
                "actual_h": actual_h,
                "actual_w": actual_w,
            })

            col += stride
            c_idx += 1
            if col_end >= w:
                break

        row += stride
        r_idx += 1
        if row_end >= h:
            break

    return patches, patch_infos


def _extract_crs_from_scene_id(scene_id: str) -> Optional[str]:
    """Fallback: deduce UTM CRS from Sentinel-2 scene ID if missing from raster."""
    match = re.search(r"_T(\d{2})([C-X])[A-Z]{2}_", scene_id)
    if match:
        zone = int(match.group(1))
        lat_band = match.group(2)
        is_north = lat_band >= 'N'
        epsg = 32600 + zone if is_north else 32700 + zone
        return f"EPSG:{epsg}"
    return None

# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_dir: Union[str, Path],
    output_dir: Union[str, Path] = "data/processed",
    config: Optional[Dict] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Run the full preprocessing stage for one scene.

    Args:
        scene_dir: Path to a raw scene directory (from Stage 1).
        output_dir: Root output directory for processed data.
        config: Optional config dict (for custom patch_size/overlap).

    Returns:
        Tuple of (patches directory path, patch index dict).

    Raises:
        FileNotFoundError: If *scene_dir* does not exist or has no band data.
    """
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name

    patch_size = PATCH_SIZE
    overlap = OVERLAP
    patch_storage = "npy"
    patch_dtype = "float32"
    if config:
        pre_cfg = config.get("preprocessing", {})
        patch_size = pre_cfg.get("patch_size", PATCH_SIZE)
        overlap = pre_cfg.get("overlap", OVERLAP)
        patch_storage = str(pre_cfg.get("patch_storage", "npy")).lower()
        patch_dtype = str(pre_cfg.get("patch_dtype", "float32")).lower()

    if patch_storage not in {"npy", "npz"}:
        logger.warning("Unknown patch_storage='%s' — falling back to 'npy'", patch_storage)
        patch_storage = "npy"
    if patch_dtype not in {"float16", "float32"}:
        logger.warning("Unknown patch_dtype='%s' — falling back to 'float32'", patch_dtype)
        patch_dtype = "float32"

    out_dir = Path(output_dir) / scene_id
    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    if stage_output_exists(out_dir, ["patch_index.json"]):
        with open(out_dir / "patch_index.json") as fh:
            patch_index = json.load(fh)
        return patches_dir, patch_index

    # Load and reorder bands
    logger.info("Loading bands from %s", scene_dir)
    image, profile, transform, crs = load_and_reorder_bands(scene_dir)
    logger.info(
        "Scene shape: %s, bands=%d",
        image.shape[1:], image.shape[0],
    )

    # Normalize
    logger.info("Normalizing scene (clip + z-score)")
    image, nodata_mask = normalize_scene(image)

    # Debug: Log statistics about normalized data
    valid_pixels = ~nodata_mask
    if valid_pixels.any():
        logger.info(
            "Normalization stats (valid pixels only): "
            "min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            image[:, valid_pixels].min(),
            image[:, valid_pixels].max(),
            image[:, valid_pixels].mean(),
            image[:, valid_pixels].std(),
        )
        # Per-band stats
        for b in range(min(3, image.shape[0])):  # Log first 3 bands
            band_valid = image[b, valid_pixels]
            logger.debug(
                "Band %d: min=%.4f, max=%.4f, mean=%.4f",
                b, band_valid.min(), band_valid.max(), band_valid.mean(),
            )
    else:
        logger.warning("All pixels marked as nodata! Check input data.")

    # Save nodata mask
    nodata_path = out_dir / "nodata_mask.npy"
    np.save(nodata_path, nodata_mask)

    # Compute and save land mask
    ndwi_thresh  = pre_cfg.get("ndwi_threshold", 0.15) if config else 0.15
    mndwi_thresh = pre_cfg.get("mndwi_threshold", 0.05) if config else 0.05
    dilate_px    = pre_cfg.get("dilate_pixels", 3) if config else 3
    land_mask    = compute_land_mask(image, ndwi_threshold=ndwi_thresh, mndwi_threshold=mndwi_thresh, dilate_pixels=dilate_px)
    land_mask_path = out_dir / "land_mask.npy"
    np.save(land_mask_path, land_mask)
    logger.info("Land mask generated: %d land pixels (%.2f%%)", land_mask.sum(), 100.0 * land_mask.sum() / land_mask.size)

    # Tile
    logger.info(
        "Tiling into %dx%d patches with %d overlap",
        patch_size, patch_size, overlap,
    )
    patches, patch_infos = tile_scene(image, patch_size, overlap)

    # Save patches and build index
    patch_index = {}
    patch_stats = {"min": [], "max": [], "mean": [], "std": []}
    
    for i, (patch, info) in enumerate(zip(patches, patch_infos)):
        patch_id = f"patch_{i:04d}"
        patch_arr = patch.astype(np.float16 if patch_dtype == "float16" else np.float32, copy=False)
        
        # Collect statistics (on a few patches for logging)
        if i < 5 or i == len(patches) - 1:
            valid = patch_arr > -10  # Most valid pixels are > -10
            if valid.any():
                patch_stats["min"].append(patch_arr[valid].min())
                patch_stats["max"].append(patch_arr[valid].max())
                patch_stats["mean"].append(patch_arr[valid].mean())
                patch_stats["std"].append(patch_arr[valid].std())
        
        patch_path = patches_dir / f"{patch_id}.{patch_storage}"
        if patch_storage == "npz":
            np.savez_compressed(patch_path, patch=patch_arr)
        else:
            np.save(patch_path, patch_arr)

        # Build geo_transform for this patch
        if transform is not None:
            patch_transform = rasterio.transform.from_bounds(
                transform.c + info["col_start"] * transform.a,
                transform.f + (info["row_start"] + info["actual_h"]) * transform.e,
                transform.c + (info["col_start"] + info["actual_w"]) * transform.a,
                transform.f + info["row_start"] * transform.e,
                info["actual_w"],
                info["actual_h"],
            )
            geo_transform_list = list(patch_transform)[:6]
        else:
            geo_transform_list = None
            
        # Determine actual string CRS to save
        crs_str = str(crs) if crs else _extract_crs_from_scene_id(scene_id)

        patch_index[patch_id] = {
            "row": info["row"],
            "col": info["col"],
            "row_start": info["row_start"],
            "col_start": info["col_start"],
            "actual_h": info["actual_h"],
            "actual_w": info["actual_w"],
            "patch_file": patch_path.name,
            "geo_transform": geo_transform_list,
            "crs": crs_str,
            "nodata_mask_path": str(nodata_path),
            "land_mask_path": str(land_mask_path),
        }

    # Log patch statistics from sampled patches
    if patch_stats["mean"]:
        logger.info(
            "Patch statistics (sampled %d patches): "
            "min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            len(patch_stats["mean"]),
            min(patch_stats["min"]), max(patch_stats["max"]),
            np.mean(patch_stats["mean"]), np.mean(patch_stats["std"]),
        )

    # Save patch index
    index_path = out_dir / "patch_index.json"
    with open(index_path, "w") as fh:
        json.dump(patch_index, fh, indent=2)

    # Determine final scene CRS
    final_crs = str(crs) if crs else _extract_crs_from_scene_id(scene_id)

    # Save scene metadata
    scene_meta = {
        "scene_id": scene_id,
        "original_shape": list(image.shape),
        "num_patches": len(patches),
        "patch_size": patch_size,
        "overlap": overlap,
        "patch_storage": patch_storage,
        "patch_dtype": patch_dtype,
        "crs": final_crs,
        "transform": list(transform)[:6] if transform else None,
    }
    with open(out_dir / "scene_meta.json", "w") as fh:
        json.dump(scene_meta, fh, indent=2)

    logger.info(
        "[bold green]Stage 2 complete[/] — %d patches saved to %s",
        len(patches), patches_dir,
    )
    return patches_dir, patch_index


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Preprocess raw scenes into model-ready patches",
    )
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to raw scene directory")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    patches_dir, patch_index = run(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"\nPreprocessed {len(patch_index)} patches to {patches_dir}")


if __name__ == "__main__":
    main()
