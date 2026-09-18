"""
Plastic-Ledger — Master Pipeline Runner
==========================================
Orchestrates all 7 stages end-to-end with progress logging,
stage caching, and graceful error handling.

Usage:
    python pipeline/run_pipeline.py \
        --bbox "80.0,8.0,82.0,10.0" \
        --target_date "2024-01-31" \
        --output_dir "data/runs/run_001" \
    --model_path "d:/Plastic-Ledger/models/segformer/runs/segformer_v1/best_model_SegTransformer.pth" \
        --cloud_cover 20 \
        --backtrack_days 30 \
        --skip_stages ""
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is on sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parent.parent)
_src_dir = Path(__file__).resolve().parent.parent   # d:\...\src
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load .env from project root or src/ regardless of CWD
for _candidate_env in [_src_dir.parent / ".env", _src_dir / ".env", Path(".env")]:
    if _candidate_env.exists():
        try:
            from dotenv import load_dotenv as _load_dotenv
            _load_dotenv(_candidate_env, override=True)
            break
        except ImportError:
            pass

# CRITICAL: Prevent external PROJ conflicts (e.g. from pyproj vs rasterio vs global)
# by ensuring neither PROJ_LIB nor PROJ_DATA is forcing a specific path globally.
# This forces each library to fall back to its internal bundled binaries/data.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)
from pipeline.utils.logging_utils import get_logger
from pipeline.utils.cache_utils import load_config

logger = get_logger(__name__)


def _parse_skip_stages(skip_str: str) -> Set[int]:
    """Parse comma-separated stage numbers to skip.

    Args:
        skip_str: e.g. ``"1,5"`` or ``""``.

    Returns:
        Set of stage numbers to skip.
    """
    if not skip_str or not skip_str.strip():
        return set()
    return {int(s.strip()) for s in skip_str.split(",") if s.strip()}


def run_pipeline(
    bbox: Tuple[float, float, float, float],
    target_date: str,
    output_dir: Union[str, Path],
    model_path: Union[str, Path],
    cloud_cover: int = 20,
    backtrack_days: int = 30,
    skip_stages: Set[int] = None,
    cleanup_patches: bool = False,
    config_path: str = None,    # None → resolved relative to this file
    filter_by_bbox: bool = True,
    max_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the complete Plastic-Ledger pipeline.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        target_date: Maximum date to search backwards from (ISO string).
        output_dir: Root output directory for this run.
        model_path: Path to the trained model checkpoint.
        cloud_cover: Maximum cloud cover percentage.
        backtrack_days: Days to back-track particles.
        skip_stages: Set of stage numbers (1–7) to skip.
        cleanup_patches: If True, delete Stage 2 patch cache after Stage 4.
        config_path: Path to YAML config file.
        filter_by_bbox: If True (default), Stage 3 only runs inference on
            patches that intersect the bbox. Set to False to process the
            entire satellite tile (all ~2400 patches).
        max_clusters: Limit backtracking to this many clusters.

    Returns:
        Dict with run summary including output paths and metrics.

    Raises:
        RuntimeError: If critical stages fail.
    """
    if skip_stages is None:
        skip_stages = set()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    # Load config — resolve path relative to src/ so it works from any CWD
    if config_path is None:
        config_path = str(_src_dir / "config" / "config.yaml")
    env_path = str(_src_dir / ".env")
    try:
        config = load_config(config_path, env_path=env_path)
    except FileNotFoundError:
        logger.warning("Config file not found: %s — using defaults", config_path)
        config = {}

    # Override backtracking days in config
    if "backtracking" not in config:
        config["backtracking"] = {}
    config["backtracking"]["days"] = backtrack_days
    if max_clusters is not None:
        config["backtracking"]["max_clusters"] = max_clusters

    # Create subdirectories
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "cache" / "sentinel2"
    processed_dir = output_dir / "processed"
    detections_dir = output_dir / "detections"
    attribution_dir = output_dir / "attribution"
    reports_dir = output_dir / "reports"

    for d in [raw_dir, processed_dir, detections_dir, attribution_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "bbox": list(bbox),
        "target_date": target_date,
        "model_path": str(model_path),
        "stages_completed": [],
        "stages_skipped": list(skip_stages),
        "stages_failed": [],
        "outputs": {},
    }

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel(
            "[bold cyan]🌊 Plastic-Ledger Pipeline[/]\n"
            f"  BBox: {bbox}\n"
            f"  Target Date: {target_date}\n"
            f"  Cloud Cover: ≤{cloud_cover}%\n"
            f"  Back-track: {backtrack_days} days\n"
            f"  Max Clusters: {max_clusters or 'all'}\n"
            f"  Cleanup Patches: {'yes' if cleanup_patches else 'no'}\n"
            f"  Skip Stages: {skip_stages or 'none'}",
            style="cyan",
        ))
    except (ImportError, UnicodeEncodeError, Exception):
        print(f"\n{'='*60}")
        print(f"  Plastic-Ledger Pipeline")
        print(f"  BBox: {bbox}, Target Date: {target_date}")
        print(f"{'='*60}")

    scene_dirs = []
    scene_metas = []
    t_start = time.time()

    # ═══════════════════════════════════════════
    # STAGE 1 — Satellite Data Ingestion
    # ═══════════════════════════════════════════
    if 1 not in skip_stages:
        logger.info("[bold blue]━━━ Stage 1: Satellite Data Ingestion ━━━[/]")
        try:
            ingest_mod = importlib.import_module("pipeline.01_ingest")
            ingest_run = ingest_mod.run
            scene_dirs, scene_metas = ingest_run(
                bbox=bbox,
                date_start="2015-01-01",  # Search all history
                date_end=target_date,     # Up to the target date
                cloud_cover_max=cloud_cover,
                output_dir=str(raw_dir),
                config=config,
            )
            run_summary["stages_completed"].append(1)
            run_summary["outputs"]["raw_scenes"] = [str(d) for d in scene_dirs]
            
            # Store scene dates for summary
            run_summary["scene_dates"] = {
                meta["id"]: meta.get("datetime", "Unknown") 
                for meta in scene_metas
            }
            
        except Exception as exc:
            logger.error("Stage 1 failed: %s", exc)
            run_summary["stages_failed"].append({"stage": 1, "error": str(exc)})
    else:
        logger.info("[dim]Stage 1 skipped[/]")
        # Try to find existing scenes
        if raw_dir.exists():
            scene_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]

    # If no scenes, try to continue with existing processed data
    if not scene_dirs:
        if processed_dir.exists():
            scene_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
            if scene_dirs:
                logger.info(
                    "No raw scenes but found %d processed scenes", len(scene_dirs),
                )
        if not scene_dirs:
            logger.warning("No scenes available — pipeline cannot continue")
            _save_summary(run_summary, output_dir, t_start)
            return run_summary

    # Process each scene through stages 2-7
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        
        # Display the date if we have it
        scene_date = run_summary.get("scene_dates", {}).get(scene_id, "")
        date_str = f" ({scene_date[:10]})" if scene_date else ""
        
        logger.info("\n[bold magenta]Processing scene: %s%s[/]", scene_id, date_str)

        # Scene-level paths
        scene_processed = processed_dir / scene_id
        patches_dir = scene_processed / "patches"
        scene_detections = detections_dir / scene_id
        scene_attribution = attribution_dir / scene_id

        detection_date = None
        meta_path = scene_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
            detection_date = meta.get("datetime")

        # ═══════════════════════════════════════
        # STAGE 2 — Preprocessing
        # ═══════════════════════════════════════
        if 2 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 2: Preprocessing ━━━[/]")
            try:
                preprocess_mod = importlib.import_module("pipeline.02_preprocess")
                preprocess_run = preprocess_mod.run
                patches_dir, patch_index = preprocess_run(
                    scene_dir=str(scene_dir),
                    output_dir=str(processed_dir),
                    config=config,
                )
                run_summary["stages_completed"].append(2)
            except Exception as exc:
                logger.error("Stage 2 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 2, "scene": scene_id, "error": str(exc)},
                )
                continue
        else:
            logger.info("[dim]Stage 2 skipped[/]")

        # ═══════════════════════════════════════
        # STAGE 3 — Marine Debris Detection
        # ═══════════════════════════════════════
        geojson_path = scene_detections / "detections.geojson"
        if 3 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 3: Marine Debris Detection ━━━[/]")
            try:
                detect_mod = importlib.import_module("pipeline.03_detect")
                detect_run = detect_mod.run
                geojson_path = detect_run(
                    scene_id=scene_id,
                    patches_dir=str(patches_dir),
                    model_path=str(model_path),
                    output_dir=str(detections_dir),
                    config=config,
                    bbox=bbox if filter_by_bbox else None,
                )
                run_summary["stages_completed"].append(3)
            except Exception as exc:
                logger.error("Stage 3 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 3, "scene": scene_id, "error": str(exc)},
                )
                continue
        else:
            logger.info("[dim]Stage 3 skipped[/]")

        # ═══════════════════════════════════════
        # STAGE 4 — Polymer Classification
        # ═══════════════════════════════════════
        classified_path = scene_detections / "detections_classified.geojson"
        if 4 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 4: Polymer Classification ━━━[/]")
            try:
                polymer_mod = importlib.import_module("pipeline.04_polymer")
                polymer_run = polymer_mod.run
                classified_path, polymer_counts = polymer_run(
                    scene_id=scene_id,
                    detections_path=str(geojson_path),
                    processed_dir=str(scene_processed),
                    output_dir=str(detections_dir),
                    config=config,
                )
                run_summary["stages_completed"].append(4)
                run_summary["outputs"]["polymer_counts"] = polymer_counts
            except Exception as exc:
                logger.error("Stage 4 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 4, "scene": scene_id, "error": str(exc)},
                )
                classified_path = geojson_path  # fallback to unclassified
        else:
            logger.info("[dim]Stage 4 skipped[/]")
            if not classified_path.exists():
                classified_path = geojson_path

        if cleanup_patches and classified_path.exists():
            reclaimed_mb = _cleanup_scene_patch_cache(scene_processed)
            if reclaimed_mb > 0:
                logger.info(
                    "Cleaned Stage 2 patch cache for %s (freed %.1f MB)",
                    scene_id,
                    reclaimed_mb,
                )

        # ═══════════════════════════════════════
        # STAGE 5 — Hydrodynamic Back-Tracking
        # ═══════════════════════════════════════
        sources = []
        if 5 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 5: Hydrodynamic Back-Tracking ━━━[/]")
            try:
                backtrack_mod = importlib.import_module("pipeline.05_backtrack")
                backtrack_run = backtrack_mod.run
                sources = backtrack_run(
                    scene_id=scene_id,
                    detections_path=str(classified_path),
                    output_dir=str(attribution_dir),
                    config=config,
                    detection_date=detection_date,
                    bbox=bbox,
                )
                run_summary["stages_completed"].append(5)
            except Exception as exc:
                logger.error("Stage 5 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 5, "scene": scene_id, "error": str(exc)},
                )
        else:
            logger.info("[dim]Stage 5 skipped[/]")
            bt_summary = scene_attribution / "backtrack_summary.json"
            if bt_summary.exists():
                with open(bt_summary) as fh:
                    sources = json.load(fh)

        # ═══════════════════════════════════════
        # STAGE 6 — Source Attribution
        # ═══════════════════════════════════════
        attribution_path = scene_attribution / "attribution_report.json"
        if 6 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 6: Source Attribution ━━━[/]")
            try:
                attribute_mod = importlib.import_module("pipeline.06_attribute")
                attribute_run = attribute_mod.run
                attribution_path = attribute_run(
                    scene_id=scene_id,
                    sources=sources,
                    detections_path=str(classified_path),
                    output_dir=str(attribution_dir),
                    config=config,
                    detection_date=detection_date,
                )
                run_summary["stages_completed"].append(6)
            except Exception as exc:
                logger.error("Stage 6 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 6, "scene": scene_id, "error": str(exc)},
                )
        else:
            logger.info("[dim]Stage 6 skipped[/]")

        # ═══════════════════════════════════════
        # STAGE 7 — Report Generation
        # ═══════════════════════════════════════
        if 7 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 7: Report Generation ━━━[/]")
            try:
                report_mod = importlib.import_module("pipeline.07_report")
                report_run = report_mod.run
                report_outputs = report_run(
                    scene_id=scene_id,
                    detections_path=str(classified_path),
                    attribution_path=str(attribution_path),
                    output_dir=str(reports_dir),
                    config=config,
                    bbox=bbox if filter_by_bbox else None,
                )
                run_summary["stages_completed"].append(7)
                run_summary["outputs"]["reports"] = {
                    k: str(v) for k, v in report_outputs.items()
                }
            except Exception as exc:
                logger.error("Stage 7 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 7, "scene": scene_id, "error": str(exc)},
                )
        else:
            logger.info("[dim]Stage 7 skipped[/]")

    _save_summary(run_summary, output_dir, t_start)
    return run_summary


def _save_summary(
    summary: Dict[str, Any],
    output_dir: Path,
    t_start: float,
):
    """Save run summary and print final output.

    Args:
        summary: Run summary dict.
        output_dir: Output directory.
        t_start: Pipeline start time.
    """
    elapsed = time.time() - t_start
    summary["elapsed_seconds"] = round(elapsed, 1)

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Final output
    n_completed = len(set(summary["stages_completed"]))
    n_failed = len(summary["stages_failed"])
    
    # Format dates list if available
    dates_found = summary.get("scene_dates", {})
    dates_str = f"\n  Fetched Dates: {', '.join([d[:10] for d in dates_found.values() if d])}" if dates_found else ""

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel(
            f"[bold green]✅ Pipeline Complete[/]\n"
            f"  Stages completed: {n_completed}\n"
            f"  Stages failed: {n_failed}\n"
            f"  Time elapsed: {elapsed:.1f}s{dates_str}\n"
            f"  Summary: {summary_path}\n"
            f"  Reports: {summary.get('outputs', {}).get('reports', 'N/A')}",
            style="green",
        ))
    except ImportError:
        print(f"\n✅ Pipeline complete: {n_completed} stages, "
              f"{n_failed} failures, {elapsed:.1f}s{dates_str}")
        print(f"   Summary: {summary_path}")


def _cleanup_scene_patch_cache(scene_processed_dir: Path) -> float:
    """Delete heavyweight Stage 2 patch cache for one scene.

    This removes patch tiles and cache markers so Stage 2 can regenerate
    cleanly if needed in a future run.

    Args:
        scene_processed_dir: Path like ``processed/<scene_id>``.

    Returns:
        Reclaimed size in MB.
    """
    reclaimed_bytes = 0

    patches_dir = scene_processed_dir / "patches"
    if patches_dir.exists():
        for pattern in ("*.npy", "*.npz"):
            for p in patches_dir.rglob(pattern):
                try:
                    reclaimed_bytes += p.stat().st_size
                    p.unlink()
                except Exception:
                    continue
        # Remove empty patch directory tree if possible.
        try:
            shutil.rmtree(patches_dir)
        except Exception:
            pass

    for f in ["patch_index.json", "nodata_mask.npy"]:
        fp = scene_processed_dir / f
        if fp.exists():
            try:
                reclaimed_bytes += fp.stat().st_size
                fp.unlink()
            except Exception:
                continue

    return reclaimed_bytes / (1024 * 1024)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Plastic-Ledger: End-to-end marine debris detection pipeline",
    )
    parser.add_argument(
        "--bbox", type=str, required=True,
        help="Bounding box: 'lon_min,lat_min,lon_max,lat_max'",
    )
    parser.add_argument("--date", type=str, required=True, help="Find the last available scene on or before this date")
    parser.add_argument(
        "--output_dir", type=str, default="data/runs/run_001",
        help="Output directory for this run",
    )
    parser.add_argument(
        "--model_path", type=str,
        default="models/production/best_model_SegFormer_v2.pth",
        help="Path to trained segmentation checkpoint (.pth)",
    )
    parser.add_argument("--cloud_cover", type=int, default=20)
    parser.add_argument("--backtrack_days", type=int, default=30)
    parser.add_argument(
        "--cleanup_patches", action="store_true",
        help="Delete Stage 2 patch cache (.npy) after Stage 4 to save disk space",
    )
    parser.add_argument(
        "--skip_stages", type=str, default="",
        help="Comma-separated stage numbers to skip, e.g. '1,5'",
    )
    parser.add_argument(
        "--max_clusters", type=int, default=None,
        help="Maximum number of clusters to run backtracking for",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: src/config/config.yaml)",
    )
    parser.add_argument(
        "--no-bbox-filter", action="store_true", dest="no_bbox_filter",
        help="Disable bbox spatial filtering in Stage 3. When set, the model "
             "runs on ALL patches in the full satellite tile, not just those "
             "within the bbox. Useful for full-scene evaluation.",
    )
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    assert len(bbox) == 4, "bbox must have exactly 4 comma-separated values"

    skip = _parse_skip_stages(args.skip_stages)

    summary = run_pipeline(
        bbox=bbox,
        target_date=args.date,
        output_dir=args.output_dir,
        model_path=args.model_path,
        cloud_cover=args.cloud_cover,
        backtrack_days=args.backtrack_days,
        skip_stages=skip,
        cleanup_patches=args.cleanup_patches,
        config_path=args.config,
        filter_by_bbox=not args.no_bbox_filter,
        max_clusters=args.max_clusters,
    )


if __name__ == "__main__":
    main()
