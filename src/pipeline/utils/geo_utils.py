"""
Plastic-Ledger — Geo Utilities
================================
Shared rasterio / shapely / retry helpers used across pipeline stages.
"""

import time
import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import Affine
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import shape, mapping, box

from pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_geotiff(path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """Load a GeoTIFF and return its data array and full rasterio profile.

    Args:
        path: Path to the GeoTIFF file.

    Returns:
        Tuple of (array with shape ``(bands, height, width)``, rasterio profile dict).

    Raises:
        FileNotFoundError: If *path* does not exist.
        rasterio.errors.RasterioIOError: If the file cannot be opened.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def save_geotiff(
    path: Union[str, Path],
    data: np.ndarray,
    profile: dict,
    **overrides: Any,
) -> Path:
    """Write a numpy array as a GeoTIFF.

    Args:
        path: Output file path.
        data: Array of shape ``(bands, H, W)`` or ``(H, W)`` (single band).
        profile: Base rasterio profile (CRS, transform, etc.).
        **overrides: Extra profile keys to override (e.g. ``dtype``, ``compress``).

    Returns:
        The resolved output :class:`~pathlib.Path`.

    Raises:
        ValueError: If *data* has an unexpected number of dimensions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    if data.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D array, got {data.ndim}-D")

    out_profile = profile.copy()
    out_profile.update(
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
        compress="lzw",
    )
    out_profile.update(overrides)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data)

    return path


def array_to_polygons(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS,
    min_area_m2: float = 0.0,
) -> gpd.GeoDataFrame:
    """Convert a binary uint8 mask to a GeoDataFrame of polygons.

    Args:
        mask: 2-D ``uint8`` array where non-zero pixels are features.
        transform: Affine geotransform of the raster.
        crs: Coordinate reference system.
        min_area_m2: Minimum polygon area in m² to keep.

    Returns:
        :class:`~geopandas.GeoDataFrame` with a ``geometry`` column and a
        ``value`` attribute column.

    Raises:
        ValueError: If *mask* is not 2-D.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {mask.ndim}-D")

    mask = mask.astype(np.uint8)
    results = list(rasterio_shapes(mask, mask=mask > 0, transform=transform))

    if not results:
        return gpd.GeoDataFrame(columns=["geometry", "value"], crs=crs)

    geometries = []
    values = []
    for geom, val in results:
        poly = shape(geom)
        if poly.area >= min_area_m2:
            geometries.append(poly)
            values.append(int(val))

    gdf = gpd.GeoDataFrame({"geometry": geometries, "value": values}, crs=crs)
    return gdf


def retry_request(
    func: Callable,
    retries: int = 3,
    base_delay: float = 2.0,
    exceptions: Tuple = (Exception,),
    retry_if: Optional[Callable[[Exception], bool]] = None,
) -> Callable:
    """Decorator / wrapper for HTTP calls with exponential back-off.

    Args:
        func: The callable to wrap.
        retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds (doubles each retry).
        exceptions: Tuple of exception classes to catch.
        retry_if: Optional predicate receiving an exception and returning
            whether to retry. If ``None``, all caught exceptions are retried.

    Returns:
        Wrapped callable with retry logic.

    Raises:
        The last exception raised after all retries are exhausted.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc = None
        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as exc:
                if retry_if is not None and not retry_if(exc):
                    raise
                last_exc = exc
                if attempt < retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1,
                        retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    return wrapper


def expand_bbox(
    bbox: Tuple[float, float, float, float],
    degrees: float,
) -> Tuple[float, float, float, float]:
    """Expand a bounding box by *degrees* in each direction.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        degrees: Amount to expand.

    Returns:
        Expanded ``(lon_min, lat_min, lon_max, lat_max)``.
    """
    return (
        bbox[0] - degrees,
        bbox[1] - degrees,
        bbox[2] + degrees,
        bbox[3] + degrees,
    )


def bbox_to_shapely(
    bbox: Tuple[float, float, float, float],
) -> "shapely.geometry.Polygon":
    """Convert a ``(lon_min, lat_min, lon_max, lat_max)`` bbox to a Shapely box.

    Args:
        bbox: Bounding box tuple.

    Returns:
        :class:`shapely.geometry.Polygon`.
    """
    return box(*bbox)
