"""
Plastic-Ledger — Stage 5: Hydrodynamic Back-Tracking
======================================================
Implements a Lagrangian particle back-tracking model that traces detected
debris back to likely source points using ocean current and wind data.

Usage (standalone):
    python -m pipeline.05_backtrack \\
        --scene_id SCENE_ID \\
        --detections data/detections/SCENE_ID/detections_classified.geojson \\
        --output_dir data/attribution

Dependencies: copernicusmarine, cdsapi, numpy, scipy, geopandas, sklearn, xarray
"""

import argparse
import json
import inspect
import os
import sys
import warnings
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint, box
from sklearn.cluster import DBSCAN

from parcels import (
    FieldSet,
    ParticleSet,
    JITParticle,
    ScipyParticle,
    AdvectionRK4,
    AdvectionRK45,
    AdvectionEE,
    Variable,
    ParcelsRandom,
)

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.geo_utils import expand_bbox, retry_request
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# OCEAN CURRENT DATA
# ─────────────────────────────────────────────
def download_ocean_currents(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    output_dir: Path,
) -> Optional[Path]:
    """Download CMEMS global ocean surface current data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ocean_currents.nc"

    if out_path.exists():
        logger.info("Ocean currents already cached: %s", out_path)
        return out_path

    try:
        import copernicusmarine as cm

        if os.environ.get("COPERNICUS_USERNAME"):
            os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = os.environ["COPERNICUS_USERNAME"]
        if os.environ.get("COPERNICUS_PASSWORD"):
            os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = os.environ["COPERNICUS_PASSWORD"]

        @retry_request
        def _download():
            dataset_id = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i"
            if date_start < "2020-11-01":
                dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
                
            cm.subset(
                dataset_id=dataset_id,
                variables=["uo", "vo"],
                minimum_longitude=bbox[0],
                minimum_latitude=bbox[1],
                maximum_longitude=bbox[2],
                maximum_latitude=bbox[3],
                start_datetime=date_start,
                end_datetime=date_end,
                output_directory=str(output_dir),
                output_filename="ocean_currents.nc",
            )

        _download()
        logger.info("Downloaded ocean currents to %s", out_path)
        return out_path

    except ImportError:
        logger.error("copernicusmarine not installed — CMEMS download failed")
        return None
    except Exception as exc:
        logger.error("CMEMS download failed: %s", exc)
        return None


# ─────────────────────────────────────────────
# WIND DATA
# ─────────────────────────────────────────────
def download_wind_data(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    output_dir: Path,
) -> Optional[Path]:
    """Download ERA5 10m wind components."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "wind_data.nc"

    if out_path.exists():
        logger.info("Wind data already cached: %s", out_path)
        return out_path

    try:
        import cdsapi

        _cds_kwargs: dict = {}
        cds_key = os.environ.get("CDS_API_KEY")
        cds_url = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
        if cds_key:
            _cds_kwargs = {"url": cds_url, "key": cds_key}

        retry_max = int(os.environ.get("CDS_RETRY_MAX", "3"))
        sleep_max = int(os.environ.get("CDS_SLEEP_MAX", "10"))
        timeout = int(os.environ.get("CDS_TIMEOUT", "60"))

        optional_client_kwargs = {
            "retry_max": max(0, retry_max),
            "sleep_max": max(1, sleep_max),
            "timeout": max(1, timeout),
            "quiet": True,
            "progress": False,
        }

        try:
            sig = inspect.signature(cdsapi.Client.__init__)
            supports_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if supports_var_kwargs:
                supported = optional_client_kwargs
            else:
                supported = {k: v for k, v in optional_client_kwargs.items() if k in sig.parameters}
        except (ValueError, TypeError):
            supported = {}

        cds_client_kwargs = {**_cds_kwargs, **supported}

        @retry_request
        def _download():
            c = cdsapi.Client(**cds_client_kwargs)
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                    "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
                    "date": f"{date_start[:10]}/{date_end[:10]}",
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "format": "netcdf",
                },
                str(out_path),
            )

        _download()
        logger.info("Downloaded wind data to %s", out_path)
        return out_path

    except ImportError:
        logger.error("cdsapi not installed — ERA5 download failed")
        return None
    except Exception as exc:
        logger.error("ERA5 download failed: %s", exc)
        return None


# ─────────────────────────────────────────────
# PLASTIC PARTICLE CLASS & WINDAGE LOOKUP
# ─────────────────────────────────────────────
class PlasticParticle(ScipyParticle):
    """Custom Parcels particle with dynamic plastic-type windage coefficient."""
    windage = Variable('windage', initial=0.03, dtype=np.float32)


def get_windage_for_plastic_type(plastic_type: str, config: Optional[Dict] = None) -> float:
    """Return aerodynamic windage coefficient based on debris type."""
    mapping = {
        "bottle": 0.03,
        "fishing_net": 0.015,
        "rope": 0.02,
        "foam": 0.05,
        "generic": 0.03,
        "marine debris (plastic)": 0.03,
    }
    if config and "backtracking" in config and "plastic_windage" in config["backtracking"]:
        for k, v in config["backtracking"]["plastic_windage"].items():
            mapping[str(k).lower()] = float(v)
    
    key = str(plastic_type).lower()
    for k, v in mapping.items():
        if k in key:
            return float(v)
    return 0.03


def get_integrator_kernel(integrator_name: str):
    """Return appropriate OceanParcels advection scheme."""
    name = str(integrator_name).upper().strip()
    if name in ["RK45", "ADVECTIONRK45"]:
        return AdvectionRK45
    elif name in ["EULER", "EE", "ADVECTIONEE"]:
        return AdvectionEE
    return AdvectionRK4


# ─────────────────────────────────────────────
# OCEANPARCELS SETUP & KERNELS
# ─────────────────────────────────────────────
def load_parcels_fieldset(
    ocean_nc: Optional[Path],
    wind_nc: Optional[Path],
    kh: float = 0.0,
) -> Optional[FieldSet]:
    """Load CMEMS and ERA5 NetCDF files into a combined OceanParcels FieldSet."""
    if not ocean_nc or not ocean_nc.exists() or not wind_nc or not wind_nc.exists():
        logger.error("Missing forcing data for OceanParcels.")
        return None

    try:
        import xarray as xr
        
        # Clear any stale file locks/caches from previous runs in the process
        try:
            xr.backends.file_manager.FILE_CACHE.clear()
        except Exception:
            pass
        
        # Determine time dimension name dynamically
        def get_time_dim(nc_path):
            with xr.open_dataset(nc_path) as ds:
                if "time" in ds.coords: return "time"
                if "valid_time" in ds.coords: return "valid_time"
                return next((c for c in ds.coords if "time" in c.lower()), list(ds.dims)[0])
        
        ocean_time = get_time_dim(ocean_nc)
        wind_time = get_time_dim(wind_nc)

        filenames = {'U': str(ocean_nc), 'V': str(ocean_nc)}
        variables = {'U': 'uo', 'V': 'vo'}
        dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': ocean_time},
                      'V': {'lon': 'longitude', 'lat': 'latitude', 'time': ocean_time}}
        
        # Create base fieldset with ocean currents
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True)

        # Load and add wind data
        wind_filenames = {'U_wind': str(wind_nc), 'V_wind': str(wind_nc)}
        wind_variables = {'U_wind': 'u10', 'V_wind': 'v10'}
        wind_dimensions = {'U_wind': {'lon': 'longitude', 'lat': 'latitude', 'time': wind_time},
                           'V_wind': {'lon': 'longitude', 'lat': 'latitude', 'time': wind_time}}
                           
        wind_fieldset = FieldSet.from_netcdf(wind_filenames, wind_variables, wind_dimensions, allow_time_extrapolation=True)
        fieldset.add_field(wind_fieldset.U_wind)
        fieldset.add_field(wind_fieldset.V_wind)
        
        # Add horizontal diffusion coefficient parameter Kh
        if kh > 0.0:
            fieldset.add_constant('Kh', kh)
        
        return fieldset
    except Exception as e:
        logger.warning(f"Failed to create FieldSet: {e}")
        return None


def DirectWindageKernel(particle, fieldset, time):
    """Applies a direct windage coefficient to floating particles.
    
    This is NOT physical Stokes drift.
    It approximates the aerodynamic drag experienced by floating debris based on 10m wind fields.
    """
    u_wind = fieldset.U_wind[time, particle.depth, particle.lat, particle.lon]
    v_wind = fieldset.V_wind[time, particle.depth, particle.lat, particle.lon]
    
    lat_dist = 111000.0
    lon_dist = 111000.0 * math.cos(particle.lat * math.pi / 180.0)
    
    # Apply particle-specific windage coefficient (default 0.03)
    w_coeff = particle.windage if hasattr(particle, "windage") else 0.03
    particle_dlon += (u_wind * w_coeff / lon_dist) * particle.dt
    particle_dlat += (v_wind * w_coeff / lat_dist) * particle.dt


def BrownianDiffusion2D(particle, fieldset, time):
    """Applies 2D horizontal stochastic random walk diffusion (Kh in m^2/s)."""
    kh = fieldset.Kh
    if kh > 0.0:
        dt_abs = math.fabs(particle.dt)
        lat_dist = 111000.0
        lon_dist = 111000.0 * math.cos(particle.lat * math.pi / 180.0)
        
        # High-quality uniform pseudo-random hash generator (avoids C compiler/GCC requirement)
        s1 = math.sin(particle.id * 12.9898 + time * 78.233) * 43758.5453
        r1 = s1 - math.floor(s1)
        s2 = math.sin((particle.id + 1.0) * 12.9898 + (time + 1.0) * 78.233) * 43758.5453
        r2 = s2 - math.floor(s2)
        
        u1 = max(0.0001, min(0.9999, r1))
        u2 = max(0.0001, min(0.9999, r2))
        
        # Box-Muller transform to Gaussian normal distribution
        r_lon = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        r_lat = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        
        step_scale = math.sqrt(2.0 * kh * dt_abs)
        particle_dlon += (r_lon * step_scale / lon_dist)
        particle_dlat += (r_lat * step_scale / lat_dist)


def compute_ensemble_statistics(lons_array: np.ndarray, lats_array: np.ndarray) -> List[Dict[str, Any]]:
    """Compute per-timestep spatial statistics across all particles."""
    n_trajs, n_obs = lons_array.shape
    stats_list = []
    
    for t in range(n_obs):
        t_lons = lons_array[:, t]
        t_lats = lats_array[:, t]
        valid_mask = ~np.isnan(t_lons) & ~np.isnan(t_lats)
        v_lons = t_lons[valid_mask]
        v_lats = t_lats[valid_mask]
        
        if len(v_lons) == 0:
            continue
            
        m_lon, m_lat = float(np.mean(v_lons)), float(np.mean(v_lats))
        std_lon, std_lat = float(np.std(v_lons)), float(np.std(v_lats))
        
        cov_matrix = np.cov(v_lons, v_lats).tolist() if len(v_lons) > 1 else [[0.0, 0.0], [0.0, 0.0]]
        
        # 95% confidence ellipse semi-axes
        if len(v_lons) > 2:
            cov = np.cov(v_lons, v_lats)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
            # 5.991 for 95% confidence chi-squared 2 d.o.f
            semi_major = float(np.sqrt(5.991 * max(0.0, eigenvals[0])))
            semi_minor = float(np.sqrt(5.991 * max(0.0, eigenvals[1])))
        else:
            semi_major, semi_minor, angle = 0.0, 0.0, 0.0

        # Convex hull area in km2
        convex_hull_area = 0.0
        if len(v_lons) >= 3:
            pts = list(zip(v_lons, v_lats))
            try:
                hull = MultiPoint(pts).convex_hull
                # Approx area conversion deg2 to km2 at mean latitude
                km_per_lat = 111.0
                km_per_lon = 111.0 * math.cos(math.radians(m_lat))
                convex_hull_area = float(hull.area * km_per_lat * km_per_lon)
            except Exception:
                convex_hull_area = 0.0

        # Mean particle spread radius (km) from mean centroid
        km_per_lat = 111.0
        km_per_lon = 111.0 * math.cos(math.radians(m_lat))
        dists = np.sqrt(((v_lons - m_lon) * km_per_lon)**2 + ((v_lats - m_lat) * km_per_lat)**2)
        spread_radius_km = float(np.mean(dists))
        
        stats_list.append({
            "timestep_idx": t,
            "n_active_particles": int(len(v_lons)),
            "mean_lon": m_lon,
            "mean_lat": m_lat,
            "std_lon": std_lon,
            "std_lat": std_lat,
            "covariance_matrix": cov_matrix,
            "ellipse_semi_major_deg": semi_major,
            "ellipse_semi_minor_deg": semi_minor,
            "ellipse_angle_deg": float(angle),
            "convex_hull_area_km2": convex_hull_area,
            "particle_spread_radius_km": spread_radius_km,
        })
        
    return stats_list

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def compute_diagnostics_and_indicators(
    sim_trajectories: List[List[Tuple[float, float]]],
    duration_hours: float,
    forcing_present: bool,
) -> Dict[str, Any]:
    traj_lengths = []
    speeds = []
    
    for traj in sim_trajectories:
        if len(traj) >= 2:
            seg_lens = [
                haversine_km(traj[i][1], traj[i][0], traj[i + 1][1], traj[i + 1][0])
                for i in range(len(traj) - 1)
            ]
            total_len = sum(seg_lens)
            traj_lengths.append(total_len)
            if duration_hours > 0:
                speeds.append((total_len / duration_hours) * (1000.0 / 3600.0))  # m/s
                
    avg_len = float(np.mean(traj_lengths)) if traj_lengths else 0.0
    avg_spd = float(np.mean(speeds)) if speeds else 0.0
    max_spd = float(np.max(speeds)) if speeds else 0.0
    vel_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
    
    smoothness_index = float(vel_std / (avg_spd + 1e-6))
    
    return {
        "forcing_data_completeness_pct": 100.0 if forcing_present else 50.0,
        "ensemble_velocity_std_ms": vel_std,
        "trajectory_smoothness_index": smoothness_index,
        "trajectory_mean_length_km": avg_len,
        "average_drift_speed_ms": avg_spd,
    }

# ─────────────────────────────────────────────
# ENDPOINT CLUSTERING
# ─────────────────────────────────────────────
def cluster_endpoints(
    endpoints: List[Tuple[float, float]],
    eps_degrees: float = 0.5,
    min_samples: int = 5,
) -> List[Dict[str, Any]]:
    """Cluster back-tracked particle endpoints using DBSCAN.

    Args:
        endpoints: List of ``(lon, lat)`` endpoint coordinates.
        eps_degrees: DBSCAN epsilon in degrees.
        min_samples: DBSCAN minimum samples per cluster.

    Returns:
        List of source region dicts with keys ``source_centroid``,
        ``source_bbox``, ``source_probability``, ``n_particles``.
    """
    if len(endpoints) < min_samples:
        # Not enough points for DBSCAN
        if endpoints:
            centroid = np.mean(endpoints, axis=0)
            return [{
                "source_centroid": (float(centroid[0]), float(centroid[1])),
                "source_bbox": (
                    float(min(e[0] for e in endpoints)),
                    float(min(e[1] for e in endpoints)),
                    float(max(e[0] for e in endpoints)),
                    float(max(e[1] for e in endpoints)),
                ),
                "source_probability": 1.0,
                "n_particles": len(endpoints),
            }]
        return []

    coords = np.array(endpoints)
    db = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)

    labels = db.labels_
    unique_labels = set(labels) - {-1}
    n_total = len(endpoints)

    sources = []
    for label in sorted(unique_labels):
        mask = labels == label
        cluster_pts = coords[mask]
        centroid = cluster_pts.mean(axis=0)

        sources.append({
            "source_centroid": (float(centroid[0]), float(centroid[1])),
            "source_bbox": (
                float(cluster_pts[:, 0].min()),
                float(cluster_pts[:, 1].min()),
                float(cluster_pts[:, 0].max()),
                float(cluster_pts[:, 1].max()),
            ),
            "source_probability": float(mask.sum()) / n_total,
            "n_particles": int(mask.sum()),
        })

    # Sort by probability descending
    sources.sort(key=lambda x: x["source_probability"], reverse=True)
    return sources


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_id: str,
    detections_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/attribution",
    config: Optional[Dict] = None,
    detection_date: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, Any]]:
    """Run back-tracking for all confirmed debris clusters.

    Args:
        scene_id: Scene identifier.
        detections_path: Path to classified detections GeoJSON.
        output_dir: Root output directory.
        config: Optional config dict.
        detection_date: ISO date string for the detection.
        bbox: Original search bounding box.

    Returns:
        List of source region dicts per debris cluster.

    Raises:
        FileNotFoundError: If *detections_path* does not exist.
    """
    detections_path = Path(detections_path)
    out_dir = Path(output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings from config
    bt_days = 30
    n_particles = 50
    dt_hours = 1.0
    eps_degrees = 0.5
    min_samples = 5
    integrator_name = "RK4"
    kh = 0.0
    plastic_type = "generic"

    if config:
        bt_cfg = config.get("backtracking", {})
        bt_days = bt_cfg.get("days", 30)
        n_particles = bt_cfg.get("n_particles", 50)
        dt_hours = bt_cfg.get("time_step_hours", 1.0)
        eps_degrees = bt_cfg.get("dbscan_eps_degrees", 0.5)
        min_samples = bt_cfg.get("dbscan_min_samples", 5)
        integrator_name = bt_cfg.get("integrator", "RK4")
        max_clusters = bt_cfg.get("max_clusters", None)
        
        diff_cfg = bt_cfg.get("horizontal_diffusion", {})
        if diff_cfg.get("enabled", False):
            kh = float(diff_cfg.get("Kh", 1.5))
        else:
            kh = 0.0
    else:
        max_clusters = None

    # Check cache
    if stage_output_exists(out_dir, ["backtrack_summary.json"]):
        with open(out_dir / "backtrack_summary.json") as fh:
            return json.load(fh)

    # Load detections
    gdf = gpd.read_file(detections_path)

    # Filter out false positives and non-plastic
    if "polymer_type" in gdf.columns:
        gdf = gdf[gdf["polymer_type"] == "Marine Debris (Plastic)"].reset_index(drop=True)
    elif "is_false_positive" in gdf.columns:
        gdf = gdf[gdf["is_false_positive"] != True].reset_index(drop=True)

    if max_clusters is not None and len(gdf) > max_clusters:
        logger.info("Limiting backtracking to first %d clusters (out of %d)", max_clusters, len(gdf))
        gdf = gdf.iloc[:max_clusters].reset_index(drop=True)

    if len(gdf) == 0:
        logger.info("No confirmed debris clusters — skipping backtracking")
        summary = []
        with open(out_dir / "backtrack_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    # Determine dates
    if detection_date:
        det_dt = datetime.fromisoformat(detection_date.replace("Z", "+00:00"))
    else:
        if "detection_date" in gdf.columns and gdf["detection_date"].iloc[0]:
            try:
                det_dt = datetime.fromisoformat(
                    str(gdf["detection_date"].iloc[0]).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                det_dt = datetime.now()
        else:
            det_dt = datetime.now()

    bt_start = (det_dt - timedelta(days=bt_days)).isoformat()
    bt_end = det_dt.isoformat()

    # Determine bbox for data download
    if bbox is None:
        all_lons = gdf.geometry.centroid.x.tolist()
        all_lats = gdf.geometry.centroid.y.tolist()
        bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))

    expanded_bbox = expand_bbox(bbox, 5.0)

    # Download current and wind data
    project_root = Path(__file__).resolve().parent.parent.parent
    bbox_str = f"{expanded_bbox[0]:.1f}_{expanded_bbox[1]:.1f}_{expanded_bbox[2]:.1f}_{expanded_bbox[3]:.1f}"
    date_str = f"{bt_start[:10]}_{bt_end[:10]}"
    data_dir = project_root / "data" / "cache" / "forcing" / f"{bbox_str}_{date_str}"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ocean current data")
    ocean_path = download_ocean_currents(expanded_bbox, bt_start, bt_end, data_dir)
    logger.info("Downloading wind data")
    wind_path = download_wind_data(expanded_bbox, bt_start, bt_end, data_dir)

    fieldset = load_parcels_fieldset(ocean_path, wind_path, kh=kh)
    if fieldset is None:
        logger.error("Could not load FieldSet. Back-tracking aborted.")
        return []

    # Select integrator scheme
    integrator_class = get_integrator_kernel(integrator_name)

    # Run back-tracking for each cluster
    all_sources = []
    total_hours = bt_days * 24

    for idx, row in gdf.iterrows():
        cluster_id = row.get("cluster_id", idx)
        centroid = row.geometry.centroid
        row_p_type = row.get("polymer_type", plastic_type)
        windage_val = get_windage_for_plastic_type(row_p_type, config)

        logger.info(
            "Back-tracking cluster %s (%d particles, %d days, windage=%.3f, Kh=%.1f)",
            cluster_id, n_particles, bt_days, windage_val, kh,
        )

        rng = np.random.default_rng(int(cluster_id) + 42)
        lons = []
        lats = []
        times = []

        for p in range(n_particles):
            p_lon = centroid.x + rng.normal(0, 0.0008)
            p_lat = centroid.y + rng.normal(0, 0.0008)
            lons.append(p_lon)
            lats.append(p_lat)
            times.append(det_dt)

        pset = ParticleSet.from_list(
            fieldset=fieldset,
            pclass=PlasticParticle,
            lon=lons,
            lat=lats,
            time=times,
            windage=[windage_val] * n_particles,
        )

        # Construct execution kernel
        kernel = pset.Kernel(integrator_class)
        if hasattr(fieldset, 'U_wind'):
            kernel += pset.Kernel(DirectWindageKernel)
        if kh > 0.0:
            kernel += pset.Kernel(BrownianDiffusion2D)

        output_zarr = out_dir / f"backtrack_{cluster_id}.zarr"
        if output_zarr.exists():
            import shutil
            shutil.rmtree(output_zarr)

        pfile = pset.ParticleFile(name=str(output_zarr), outputdt=timedelta(hours=dt_hours))

        try:
            pset.execute(
                kernel,
                runtime=timedelta(hours=total_hours),
                dt=-timedelta(hours=dt_hours),
                output_file=pfile,
            )
        except Exception as e:
            logger.error(f"Parcels execution failed for cluster {cluster_id}: {e}")
            continue

        # Process Zarr trajectories
        import xarray as xr
        import pandas as pd
        endpoints = []
        all_trajectories = []
        
        try:
            with xr.open_zarr(output_zarr) as ds_traj:
                lons_array = ds_traj['lon'].values
                lats_array = ds_traj['lat'].values
                
                # Compute time-series ensemble statistics
                stats_list = compute_ensemble_statistics(lons_array, lats_array)
                if stats_list:
                    pd.DataFrame(stats_list).to_csv(
                        out_dir / f"ensemble_statistics_{cluster_id}.csv", index=False
                    )

                for t_idx in range(lons_array.shape[0]):
                    traj_lons = lons_array[t_idx, :]
                    traj_lats = lats_array[t_idx, :]
                    
                    valid = ~np.isnan(traj_lons) & ~np.isnan(traj_lats)
                    valid_lons = traj_lons[valid]
                    valid_lats = traj_lats[valid]
                    
                    if len(valid_lons) > 0:
                        endpoints.append((float(valid_lons[-1]), float(valid_lats[-1])))
                        all_trajectories.append(list(zip(valid_lons, valid_lats)))
        except Exception as e:
            logger.error(f"Failed to read trajectories from {output_zarr}: {e}")
            continue

        # Cluster endpoints
        sources = cluster_endpoints(endpoints, eps_degrees, min_samples)
        
        diagnostics = compute_diagnostics_and_indicators(all_trajectories, total_hours, fieldset is not None)
        
        for src in sources:
            src["cluster_id"] = int(cluster_id)
            src["days_to_source"] = float(bt_days)
            src["diagnostics"] = diagnostics
            # Also store the original detection location for reference
            src["release_lon"] = float(centroid.x)
            src["release_lat"] = float(centroid.y)

        all_sources.extend(sources)

        # Save trajectory GeoJSON
        traj_features = [LineString(t) for t in all_trajectories if len(t) >= 2]
        if traj_features:
            traj_gdf = gpd.GeoDataFrame(
                {"geometry": traj_features, "cluster_id": [int(cluster_id)] * len(traj_features)},
                crs="EPSG:4326",
            )
            traj_gdf.to_file(out_dir / f"backtrack_{cluster_id}.geojson", driver="GeoJSON")

    # Save summary & run metadata
    with open(out_dir / "backtrack_summary.json", "w") as fh:
        json.dump(all_sources, fh, indent=2, default=str)

    run_meta = {
        "cmems_product": "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
        "era5_product": "reanalysis-era5-single-levels (u10, v10)",
        "ocean_path": str(ocean_path) if ocean_path else None,
        "wind_path": str(wind_path) if wind_path else None,
        "integrator": integrator_name,
        "time_step_hours": dt_hours,
        "horizontal_diffusion_kh": kh,
        "n_particles": n_particles,
        "bt_days": bt_days,
        "kernels": ["Advection" + integrator_name, "DirectWindageKernel", "BrownianDiffusion2D" if kh > 0 else "None"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "run_metadata.json", "w") as fh:
        json.dump(run_meta, fh, indent=2)

    logger.info(
        "[bold green]Stage 5 complete[/] — %d source regions from %d clusters",
        len(all_sources), len(gdf),
    )
    return all_sources

def run_single_cluster(
    scene_id: str,
    cluster_id: int,
    output_dir: Union[str, Path] = "data/attribution",
    config: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir) / scene_id
    
    # Load config settings
    bt_days = 30
    n_particles = 50
    dt_hours = 1.0
    eps_degrees = 0.5
    min_samples = 5
    integrator_name = "RK4"
    kh = 0.0
    plastic_type = "generic"

    if config:
        bt_cfg = config.get("backtracking", {})
        bt_days = bt_cfg.get("days", 30)
        n_particles = bt_cfg.get("n_particles", 50)
        dt_hours = bt_cfg.get("time_step_hours", 1.0)
        eps_degrees = bt_cfg.get("dbscan_eps_degrees", 0.5)
        min_samples = bt_cfg.get("dbscan_min_samples", 5)
        integrator_name = bt_cfg.get("integrator", "RK4")
        diff_cfg = bt_cfg.get("horizontal_diffusion", {})
        if diff_cfg.get("enabled", False):
            kh = float(diff_cfg.get("Kh", 1.5))

    # Load existing backtrack summary
    summary_path = out_dir / "backtrack_summary.json"
    all_sources = []
    if summary_path.exists():
        with open(summary_path) as fh:
            all_sources = json.load(fh)
            
    # Check if cluster is already backtracked
    if any(s.get("cluster_id") == cluster_id for s in all_sources):
        logger.info(f"Cluster {cluster_id} is already backtracked.")
        return all_sources

    # Find cluster from debris summary
    report_dir = Path(output_dir).parent / "reports" / scene_id
    csv_path = report_dir / "debris_summary.csv"
    if not csv_path.exists():
        logger.error(f"Cannot find {csv_path}")
        return all_sources
        
    import pandas as pd
    df = pd.read_csv(csv_path)
    row = df[df["cluster_id"] == cluster_id]
    if row.empty:
        logger.error(f"Cluster {cluster_id} not found in {csv_path}")
        return all_sources
        
    row = row.iloc[0]
    
    # Setup fieldset
    ocean_path = None
    wind_path = None
    run_meta_path = out_dir / "run_metadata.json"
    if run_meta_path.exists():
        with open(run_meta_path) as fh:
            run_meta = json.load(fh)
        if run_meta.get("ocean_path"):
            ocean_path = Path(run_meta["ocean_path"])
        if run_meta.get("wind_path"):
            wind_path = Path(run_meta["wind_path"])
            
    if not ocean_path or not ocean_path.exists() or not wind_path or not wind_path.exists():
        logger.error("Forcing data missing for single cluster backtracking.")
        return all_sources
        
    fieldset = load_parcels_fieldset(ocean_path, wind_path, kh=kh)
    if fieldset is None:
        logger.error("Could not load FieldSet.")
        return all_sources

    integrator_class = get_integrator_kernel(integrator_name)
    total_hours = bt_days * 24
    
    det_dt = datetime.now()
    if "detection_date" in df.columns and row["detection_date"]:
        try:
            det_dt = datetime.fromisoformat(str(row["detection_date"]).replace("Z", "+00:00"))
        except:
            pass

    windage_val = get_windage_for_plastic_type(row.get("polymer_type", plastic_type), config)
    
    logger.info(f"Back-tracking SINGLE cluster {cluster_id} ({n_particles} particles, {bt_days} days)")
    
    rng = np.random.default_rng(int(cluster_id) + 42)
    lons, lats, times = [], [], []
    for p in range(n_particles):
        lons.append(row["lon"] + rng.normal(0, 0.0008))
        lats.append(row["lat"] + rng.normal(0, 0.0008))
        times.append(det_dt)
        
    pset = ParticleSet.from_list(
        fieldset=fieldset, pclass=PlasticParticle, lon=lons, lat=lats, 
        time=times, windage=[windage_val] * n_particles
    )

    kernel = pset.Kernel(integrator_class)
    if hasattr(fieldset, 'U_wind'): kernel += pset.Kernel(DirectWindageKernel)
    if kh > 0.0: kernel += pset.Kernel(BrownianDiffusion2D)

    output_zarr = out_dir / f"backtrack_{cluster_id}.zarr"
    if output_zarr.exists():
        import shutil
        shutil.rmtree(output_zarr)
        
    pfile = pset.ParticleFile(name=str(output_zarr), outputdt=timedelta(hours=dt_hours))

    try:
        pset.execute(kernel, runtime=timedelta(hours=total_hours), dt=-timedelta(hours=dt_hours), output_file=pfile)
    except Exception as e:
        logger.error(f"Parcels execution failed: {e}")
        return all_sources

    import xarray as xr
    endpoints = []
    all_trajectories = []
    
    try:
        with xr.open_zarr(output_zarr) as ds_traj:
            lons_array = ds_traj['lon'].values
            lats_array = ds_traj['lat'].values
            
            stats_list = compute_ensemble_statistics(lons_array, lats_array)
            if stats_list:
                pd.DataFrame(stats_list).to_csv(out_dir / f"ensemble_statistics_{cluster_id}.csv", index=False)

            for t_idx in range(lons_array.shape[0]):
                valid = ~np.isnan(lons_array[t_idx, :]) & ~np.isnan(lats_array[t_idx, :])
                if valid.any():
                    valid_lons, valid_lats = lons_array[t_idx, valid], lats_array[t_idx, valid]
                    if len(valid_lons) > 0:
                        endpoints.append((float(valid_lons[-1]), float(valid_lats[-1])))
                        all_trajectories.append(list(zip(valid_lons, valid_lats)))
    except Exception as e:
        logger.error(f"Failed to read trajectories: {e}")
        return all_sources

    sources = cluster_endpoints(endpoints, eps_degrees, min_samples)
    diagnostics = compute_diagnostics_and_indicators(all_trajectories, total_hours, True)
    
    for src in sources:
        src["cluster_id"] = int(cluster_id)
        src["days_to_source"] = float(bt_days)
        src["diagnostics"] = diagnostics
        src["release_lon"] = float(row["lon"])
        src["release_lat"] = float(row["lat"])

    all_sources.extend(sources)

    # Save trajectories GeoJSON
    traj_features = [LineString(t) for t in all_trajectories if len(t) >= 2]
    if traj_features:
        gpd.GeoDataFrame(
            {"geometry": traj_features, "cluster_id": [int(cluster_id)] * len(traj_features)},
            crs="EPSG:4326"
        ).to_file(out_dir / f"backtrack_{cluster_id}.geojson", driver="GeoJSON")

    with open(summary_path, "w") as fh:
        json.dump(all_sources, fh, indent=2, default=str)
        
    return all_sources
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Hydrodynamic back-tracking of debris clusters",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/attribution")
    parser.add_argument("--detection_date", type=str, default=None)
    parser.add_argument("--bbox", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    bbox = None
    if args.bbox:
        lon_min, lat_min, lon_max, lat_max = (float(x) for x in args.bbox.split(","))
        bbox = (lon_min, lat_min, lon_max, lat_max)

    sources = run(
        scene_id=args.scene_id,
        detections_path=args.detections,
        output_dir=args.output_dir,
        config=config,
        detection_date=args.detection_date,
        bbox=bbox,
    )
    print(f"\nFound {len(sources)} source regions")


if __name__ == "__main__":
    main()
