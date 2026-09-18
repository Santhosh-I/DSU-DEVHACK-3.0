"""
Plastic-Ledger — Stage 7: Report Generation
===============================================
Generates a complete PDF report, GeoJSON summary, CSV export, and
terminal summary for each processed scene.

Usage (standalone):
    python -m pipeline.07_report \\
        --scene_id SCENE_ID \\
        --detections data/detections/SCENE_ID/detections_classified.geojson \\
        --attribution data/attribution/SCENE_ID/attribution_report.json \\
        --output_dir data/reports

Dependencies: fpdf2, matplotlib, geopandas, pandas, rich
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────
def generate_pdf(
    scene_id: str,
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict[str, Any]],
    output_dir: Path,
    detection_map_path: Optional[Path] = None,
    trajectory_map_path: Optional[Path] = None,
    rgb_map_path: Optional[Path] = None,
) -> Path:
    """Generate a PDF report with executive summary, maps, and tables.

    Args:
        scene_id: Scene identifier.
        detections_gdf: GeoDataFrame of classified debris clusters.
        attribution_data: List of attribution report entries.
        output_dir: Output directory.
        detection_map_path: Optional path to a PNG detection map.
        trajectory_map_path: Optional path to a PNG trajectory map.
        rgb_map_path: Optional path to a PNG RGB map.

    Returns:
        Path to the generated PDF file.

    Raises:
        ImportError: If ``fpdf2`` is not installed.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Executive Summary ─────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Plastic-Ledger Report", align="C", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Scene: {scene_id}", align="C", ln=1)
    pdf.ln(10)

    # Summary statistics
    n_clusters = len(detections_gdf)
    total_area = detections_gdf["area_m2"].sum() if n_clusters > 0 and "area_m2" in detections_gdf.columns else 0
    total_area_km2 = total_area / 1e6

    # Dominant polymer
    dominant_polymer = "N/A"
    if n_clusters > 0 and "polymer_type" in detections_gdf.columns:
        polymer_counts = detections_gdf["polymer_type"].value_counts()
        dominant_polymer = polymer_counts.index[0] if len(polymer_counts) > 0 else "N/A"

    # Detection date
    det_date = "Unknown"
    if n_clusters > 0 and "detection_date" in detections_gdf.columns:
        dates = detections_gdf["detection_date"].dropna()
        if len(dates) > 0:
            det_date = str(dates.iloc[0])

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", ln=1)
    pdf.set_font("Helvetica", "", 11)

    summary_lines = [
        f"Detection Date: {det_date}",
        f"Total Debris Clusters: {n_clusters}",
        f"Total Debris Area: {total_area_km2:.4f} sq km  ({total_area:.0f} sq m)",
        f"Dominant Polymer Type: {dominant_polymer}",
    ]
    if attribution_data:
        top_source = attribution_data[0]
        summary_lines.append(
            f"Top Source Attribution: {top_source.get('source_type', 'Unknown')} "
            f"({top_source.get('attribution_score', 0)*100:.0f}% confidence)"
        )

    for line in summary_lines:
        pdf.cell(0, 7, line, ln=1)

    # ── Detection Map ─────────────────────────────
    if detection_map_path and detection_map_path.exists():
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Detection Map", ln=1)
        try:
            pdf.image(str(detection_map_path), w=180)
        except Exception as exc:
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 7, f"(Map image could not be embedded: {exc})", ln=1)
            
    # ── RGB Patch Map ─────────────────────────────
    if rgb_map_path and rgb_map_path.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "RGB Patch Map", ln=1)
        try:
            pdf.image(str(rgb_map_path), w=180)
        except Exception as exc:
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 7, f"(RGB image could not be embedded: {exc})", ln=1)

    # ── Page 2: Polymer Distribution ──────────────
    if n_clusters > 0 and "polymer_type" in detections_gdf.columns:
        # Generate pie chart
        pie_path = output_dir / "polymer_distribution.png"
        _generate_polymer_pie(detections_gdf, pie_path)

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Polymer Distribution", ln=1)
        if pie_path.exists():
            try:
                pdf.image(str(pie_path), w=140)
            except Exception:
                pass

    # ── Source Attribution ─────────────────────────
    if attribution_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Source Attribution - Top 3", ln=1)
        pdf.ln(5)

        for i, attr in enumerate(attribution_data[:3]):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(
                0, 8,
                f"#{i+1}: {attr.get('source_type', 'Unknown')} "
                f"- {attr.get('attribution_score', 0)*100:.0f}% confidence",
                ln=1,
            )
            pdf.set_font("Helvetica", "", 10)
            explanation = attr.get("explanation", "No explanation available.")
            # Word wrap long explanations
            pdf.multi_cell(0, 6, explanation)
            pdf.ln(3)

    # ── Trajectory Map ────────────────────────────
    if trajectory_map_path and trajectory_map_path.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Back-Track Trajectories", ln=1)
        try:
            pdf.image(str(trajectory_map_path), w=180)
        except Exception:
            pass

    # ── Cluster Table ─────────────────────────────
    if n_clusters > 0:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Debris Cluster Details", ln=1)
        pdf.ln(3)

        # Table header
        pdf.set_font("Helvetica", "B", 8)
        col_widths = [10, 18, 12, 28, 18, 18, 18, 18, 25]
        headers = ["ID", "Area", "Conf", "Polymer", "Lat", "Lon", "Src Lat", "Src Lon", "Source"]
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 7)
        
        filtered_gdf = detections_gdf.copy()
        if "polymer_type" in filtered_gdf.columns:
            filtered_gdf = filtered_gdf[filtered_gdf["polymer_type"] == "Marine Debris (Plastic)"]
            
        for _, row in filtered_gdf.iterrows():
            cid = str(row.get("cluster_id", ""))[:4]
            area = f"{row.get('area_m2', 0):.0f}"
            conf = f"{row.get('mean_confidence', 0):.2f}"
            polymer = str(row.get("polymer_type", ""))[:16]
            lat = f"{row.get('centroid_lat', 0):.4f}"
            lon = f"{row.get('centroid_lon', 0):.4f}"

            # Find attribution for this cluster
            src = "N/A"
            src_lat = "N/A"
            src_lon = "N/A"
            if attribution_data:
                for attr in attribution_data:
                    if attr.get("debris_cluster_id") == row.get("cluster_id"):
                        src = attr.get("source_type", "N/A")[:12]
                        s_centroid = attr.get("source_centroid")
                        if s_centroid and len(s_centroid) == 2:
                            src_lon = f"{s_centroid[0]:.4f}"
                            src_lat = f"{s_centroid[1]:.4f}"
                        break

            vals = [cid, area, conf, polymer, lat, lon, src_lat, src_lon, src]
            for w, v in zip(col_widths, vals):
                pdf.cell(w, 6, v, border=1)
            pdf.ln()

    # Save
    pdf_path = output_dir / "final_report.pdf"
    pdf.output(str(pdf_path))
    logger.info("PDF report saved to %s", pdf_path)
    return pdf_path


def _generate_polymer_pie(gdf: gpd.GeoDataFrame, output_path: Path):
    """Generate a polymer distribution pie chart.

    Args:
        gdf: GeoDataFrame with ``polymer_type`` column.
        output_path: Path to save the PNG.
    """
    counts = gdf["polymer_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = ["#E63946", "#2A9D8F", "#F4A261", "#3A86FF", "#8338EC",
              "#57CC99", "#E9C46A", "#FF6B6B"]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors[:len(counts)],
        textprops={"color": "white", "fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color("white")

    ax.set_title("Polymer Type Distribution",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


def _generate_detection_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
):
    """Generate a simple detection map.

    Args:
        gdf: GeoDataFrame of detections.
        output_path: Path to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#1a1a2e")

    plot_gdf = gdf.copy()
    if len(plot_gdf) > 0 and "geometry" in plot_gdf.columns:
        plot_gdf = plot_gdf[plot_gdf.geometry.notna()]
        plot_gdf = plot_gdf[~plot_gdf.geometry.is_empty]
        if "polymer_type" in plot_gdf.columns:
            plot_gdf = plot_gdf[plot_gdf["polymer_type"] == "Marine Debris (Plastic)"]

    if len(plot_gdf) > 0:
        bounds = plot_gdf.total_bounds
        finite_bounds = np.all(np.isfinite(bounds))
        height = bounds[3] - bounds[1] if finite_bounds else np.nan
        width = bounds[2] - bounds[0] if finite_bounds else np.nan

        if not finite_bounds or height <= 0 or width <= 0:
            ax.set_aspect("auto")

        # Plot in raw data coordinates to avoid GeoPandas applying geographic
        # aspect logic to mislabeled or projected geometries.
        plot_gdf = plot_gdf.set_crs(None, allow_override=True)

        plot_gdf.plot(
            ax=ax,
            color="#E63946",
            alpha=0.9,
            edgecolor="none",
            aspect="auto",
        )
        ax.set_title(f"Debris Detections ({len(plot_gdf)} plastic clusters)",
                     color="white", fontsize=13, fontweight="bold")
    else:
        ax.set_title("No Debris Detected",
                     color="gray", fontsize=13, fontweight="bold")
                     
        if bbox is not None:
            # Zoom to bbox
            buffer = 0.02 # about 2km padding in degrees
            min_lon, min_lat, max_lon, max_lat = bbox
            ax.set_xlim(min_lon - buffer, max_lon + buffer)
            ax.set_ylim(min_lat - buffer, max_lat + buffer)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


def _generate_rgb_map(
    scene_id: str,
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
):
    """Generate an RGB map from raw Sentinel-2 B04, B03, B02."""
    from rasterio.plot import show
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "cache" / "sentinel2" / scene_id
    b4 = raw_dir / "B04.tif"
    b3 = raw_dir / "B03.tif"
    b2 = raw_dir / "B02.tif"

    if not (b4.exists() and b3.exists() and b2.exists()):
        # Just return without generating if raw data is missing
        return

    try:
        with rasterio.open(b4) as src4, rasterio.open(b3) as src3, rasterio.open(b2) as src2:
            r = src4.read(1)
            g = src3.read(1)
            b = src2.read(1)
            transform = src4.transform
            src_crs = src4.crs
            extent = [transform[2], transform[2] + transform[0] * src4.width,
                      transform[5] + transform[4] * src4.height, transform[5]]
    except Exception as e:
        logger.warning(f"Could not load raw bands for RGB map: {e}")
        return

    # Normalize to 0-1 using a common visual max (e.g. 3000)
    rgb = np.dstack([r, g, b]) / 3000.0
    rgb = np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb, extent=extent)

    # Plot plastics
    plot_gdf = gdf.copy()
    if len(plot_gdf) > 0 and "geometry" in plot_gdf.columns:
        plot_gdf = plot_gdf[plot_gdf.geometry.notna()]
        plot_gdf = plot_gdf[~plot_gdf.geometry.is_empty]
        if "polymer_type" in plot_gdf.columns:
            plot_gdf = plot_gdf[plot_gdf["polymer_type"] == "Marine Debris (Plastic)"]
            
        if len(plot_gdf) > 0:
            # Reproject to raster CRS so coordinates match
            if plot_gdf.crs and src_crs and plot_gdf.crs != src_crs:
                plot_gdf = plot_gdf.to_crs(src_crs)
                
            plot_gdf.plot(
                ax=ax,
                color="#FF0000",
                alpha=0.9,
                edgecolor="#FF0000", # Using bright red outline to make it easily visible
                linewidth=2,
            )
            
            # Zoom to the detections
            bounds = plot_gdf.total_bounds
            if len(bounds) == 4:
                minx, miny, maxx, maxy = bounds
                # Add 200m buffer
                buffer = 200
                ax.set_xlim(minx - buffer, maxx + buffer)
                ax.set_ylim(miny - buffer, maxy + buffer)
                
    if (len(plot_gdf) == 0 or "geometry" not in plot_gdf.columns) and bbox is not None:
        try:
            from pyproj import Transformer
            if src_crs:
                transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
                bl_x, bl_y = transformer.transform(bbox[0], bbox[1])
                tr_x, tr_y = transformer.transform(bbox[2], bbox[3])
                tl_x, tl_y = transformer.transform(bbox[0], bbox[3])
                br_x, br_y = transformer.transform(bbox[2], bbox[1])
                minx = min(bl_x, tr_x, tl_x, br_x)
                maxx = max(bl_x, tr_x, tl_x, br_x)
                miny = min(bl_y, tr_y, tl_y, br_y)
                maxy = max(bl_y, tr_y, tl_y, br_y)
                buffer = 200
                ax.set_xlim(minx - buffer, maxx + buffer)
                ax.set_ylim(miny - buffer, maxy + buffer)
        except Exception:
            pass

    ax.set_title("RGB Scene with Detected Plastics", fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_interactive_map(
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict],
    output_path: Path,
    attribution_dir: Optional[Path] = None,
):
    """Generate an interactive Leaflet HTML map."""
    if len(detections_gdf) == 0:
        return
        
    filtered_gdf = detections_gdf.copy()
    if "polymer_type" in filtered_gdf.columns:
        filtered_gdf = filtered_gdf[filtered_gdf["polymer_type"] == "Marine Debris (Plastic)"]

    if len(filtered_gdf) == 0:
        return

    features = []
    
    # Add debris points
    for _, row in filtered_gdf.iterrows():
        lat = row.get("centroid_lat")
        lon = row.get("centroid_lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
            
        cid = row.get("cluster_id")
        features.append({
            "type": "Feature",
            "properties": {
                "type": "debris",
                "id": str(cid),
                "area": float(row.get("area_m2", 0))
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        })
        
        # Add backtracking lines
        if attribution_data:
            has_real_trajectories = False
            for attr in attribution_data:
                if str(attr.get("debris_cluster_id")) == str(cid):
                    source_name = attr.get("source_type", "Unknown")
                    
                    # Try to load detailed particle trajectories
                    if attribution_dir is not None:
                        traj_file = attribution_dir / f"backtrack_{cid}.geojson"
                        if traj_file.exists():
                            try:
                                with open(traj_file) as fh:
                                    traj_data = json.load(fh)
                                    for feat in traj_data.get("features", []):
                                        feat["properties"]["type"] = "trajectory"
                                        feat["properties"]["source"] = source_name
                                        features.append(feat)
                                has_real_trajectories = True
                            except Exception as exc:
                                logger.warning("Could not load trajectory for cluster %s: %s", cid, exc)

                    # Fallback to straight line if detailed paths aren't found
                    if not has_real_trajectories:
                        s_cent = attr.get("source_centroid")
                        if s_cent and len(s_cent) == 2:
                            slon, slat = s_cent
                            features.append({
                                "type": "Feature",
                                "properties": {
                                    "type": "trajectory",
                                    "source": source_name
                                },
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": [[lon, lat], [slon, slat]]
                                }
                            })
                    break

    center_lat = filtered_gdf["centroid_lat"].mean()
    center_lon = filtered_gdf["centroid_lon"].mean()

    geojson_str = json.dumps({"type": "FeatureCollection", "features": features})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Plastic-Ledger Hydrodynamic Backtracking Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: sans-serif; }}
        #map {{ width: 100vw; height: 100vh; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 12);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '© OpenStreetMap contributors © CARTO',
            subdomains: 'abcd',
            maxZoom: 19
        }}).addTo(map);

        var data = {geojson_str};

        L.geoJSON(data, {{
            pointToLayer: function (feature, latlng) {{
                return L.circleMarker(latlng, {{
                    radius: 6,
                    fillColor: "#E63946",
                    color: "#ffffff",
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }}).bindPopup("Debris ID: " + feature.properties.id + "<br>Area: " + feature.properties.area + " m²");
            }},
            style: function (feature) {{
                if (feature.properties.type === 'trajectory') {{
                    return {{color: "#38bdf8", weight: 2, dashArray: "5, 5", opacity: 0.2}};
                }}
            }},
            onEachFeature: function (feature, layer) {{
                if (feature.properties.type === 'trajectory') {{
                    layer.bindPopup("Source: " + feature.properties.source);
                }}
            }}
        }}).addTo(map);
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Interactive map saved to %s", output_path)


# ─────────────────────────────────────────────
# GEOJSON & CSV
# ─────────────────────────────────────────────
def generate_geojson_summary(
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate a combined GeoJSON with detections + attributions.

    Args:
        detections_gdf: GeoDataFrame of classified detections.
        attribution_data: List of attribution report entries.
        output_path: Output file path.

    Returns:
        Path to the saved GeoJSON.
    """
    if len(detections_gdf) == 0:
        # Save empty GeoJSON
        empty_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        empty_gdf.to_file(output_path, driver="GeoJSON")
        return output_path

    # Merge attribution data into detections
    gdf = detections_gdf.copy()

    attr_by_cluster = {}
    for attr in attribution_data:
        cid = attr.get("debris_cluster_id")
        if cid is not None:
            attr_by_cluster[cid] = attr

    # Add attribution columns
    for col in ["source_type", "attribution_score", "explanation", "country"]:
        gdf[col] = gdf.apply(
            lambda row: attr_by_cluster.get(
                row.get("cluster_id"), {}
            ).get(col, ""),
            axis=1,
        )

    gdf.to_file(output_path, driver="GeoJSON")
    logger.info("GeoJSON summary saved to %s", output_path)
    return output_path


def generate_csv(
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict[str, Any]],
    scene_id: str,
    output_path: Path,
) -> Path:
    """Generate a flat CSV summary of all detections.

    Args:
        detections_gdf: GeoDataFrame of classified detections.
        attribution_data: List of attribution report entries.
        scene_id: Scene identifier.
        output_path: Output file path.

    Returns:
        Path to the saved CSV.
    """
    if len(detections_gdf) == 0:
        logger.info("Empty GeoDataFrame provided. Returning basic CSV with headers.")
        df = pd.DataFrame(columns=[
            "cluster_id", "lat", "lon", "area_sq_m", "polymer_type",
            "confidence", "top_source_type", "source_lat", "source_lon",
            "top_source_location", "top_source_country", "attribution_score", 
            "detection_date", "scene_id",
        ])
        df.to_csv(output_path, index=False)
        return output_path

    attr_by_cluster = {}
    for attr in attribution_data:
        cid = attr.get("debris_cluster_id")
        if cid is not None:
            attr_by_cluster[cid] = attr

    rows = []
    for _, det in detections_gdf.iterrows():
        cid = det.get("cluster_id", 0)
        attr = attr_by_cluster.get(cid, {})
        s_cent = attr.get("source_centroid", [None, None])

        rows.append({
            "cluster_id": cid,
            "lat": det.get("centroid_lat", 0),
            "lon": det.get("centroid_lon", 0),
            "area_sq_m": det.get("area_m2", 0),
            "polymer_type": det.get("polymer_type", ""),
            "confidence": det.get("mean_confidence", 0),
            "top_source_type": attr.get("source_type", ""),
            "source_lat": s_cent[1] if s_cent and len(s_cent) == 2 else None,
            "source_lon": s_cent[0] if s_cent and len(s_cent) == 2 else None,
            "top_source_location": attr.get("location_name", ""),
            "top_source_country": attr.get("country", ""),
            "attribution_score": attr.get("attribution_score", 0),
            "detection_date": det.get("detection_date", ""),
            "scene_id": scene_id,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("CSV summary saved to %s", output_path)
    return output_path


# ─────────────────────────────────────────────
# TERMINAL SUMMARY
# ─────────────────────────────────────────────
def print_terminal_summary(
    scene_id: str,
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict[str, Any]],
):
    """Print a rich-formatted terminal summary table.

    Args:
        scene_id: Scene identifier.
        detections_gdf: GeoDataFrame of classified detections.
        attribution_data: List of attribution report entries.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Header
        console.print(
            Panel(
                f"[bold cyan]Plastic-Ledger Report[/] — Scene: [bold]{scene_id}[/]",
                style="cyan",
            )
        )

        n_clusters = len(detections_gdf)
        total_area = detections_gdf["area_m2"].sum() if n_clusters > 0 and "area_m2" in detections_gdf.columns else 0

        console.print(f"\n  [bold]Debris Clusters:[/] {n_clusters}")
        console.print(f"  [bold]Total Area:[/] {total_area:.0f} m² ({total_area/1e6:.4f} km²)")

        if n_clusters > 0:
            # Detections table
            table = Table(
                title="Detected Debris Clusters",
                show_lines=True,
                style="cyan",
            )
            table.add_column("ID", style="bold")
            table.add_column("Area (sq m)", justify="right")
            table.add_column("Confidence", justify="right")
            table.add_column("Polymer Type")
            table.add_column("Location")

            class_counts = {}
            for _, row in detections_gdf.iterrows():
                poly = str(row.get("polymer_type", "N/A"))
                class_counts[poly] = class_counts.get(poly, 0) + 1
                
                if "False Positive" in poly:
                    continue
                    
                table.add_row(
                    str(row.get("cluster_id", "")),
                    f"{row.get('area_m2', 0):.0f}",
                    f"{row.get('mean_confidence', 0):.3f}",
                    poly,
                    f"({row.get('centroid_lat', 0):.3f}, "
                    f"{row.get('centroid_lon', 0):.3f})",
                )

            console.print(table)
            
            console.print("\n  [bold]Class Summary:[/]")
            for k, v in sorted(class_counts.items()):
                console.print(f"    - {k}: {v}")

        # Attribution table
        if attribution_data:
            attr_table = Table(
                title="Source Attribution",
                show_lines=True,
                style="green",
            )
            attr_table.add_column("Rank", style="bold")
            attr_table.add_column("Source Type")
            attr_table.add_column("Score", justify="right")
            attr_table.add_column("Location")
            attr_table.add_column("Explanation")

            for i, attr in enumerate(attribution_data[:5]):
                score = attr.get("attribution_score", 0)
                score_color = "green" if score > 0.6 else "yellow" if score > 0.3 else "red"
                attr_table.add_row(
                    f"#{i+1}",
                    attr.get("source_type", "Unknown"),
                    f"[{score_color}]{score*100:.0f}%[/]",
                    attr.get("location_name", "Unknown"),
                    attr.get("explanation", "")[:80] + "...",
                )

            console.print(attr_table)

    except ImportError:
        # Fallback without rich
        print(f"\n{'='*60}")
        print(f"  Plastic-Ledger Report — Scene: {scene_id}")
        print(f"{'='*60}")
        print(f"  Clusters: {len(detections_gdf)}")
        if attribution_data:
            for attr in attribution_data[:3]:
                print(
                    f"  #{attr.get('source_rank', '?')}: "
                    f"{attr.get('source_type', 'Unknown')} "
                    f"({attr.get('attribution_score', 0)*100:.0f}%)"
                )


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_id: str,
    detections_path: Union[str, Path],
    attribution_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/reports",
    config: Optional[Dict] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Path]:
    """Generate all report outputs for a scene.

    Args:
        scene_id: Scene identifier.
        detections_path: Path to classified detections GeoJSON.
        attribution_path: Path to attribution report JSON.
        output_dir: Root output directory.
        config: Optional config dict.

    Returns:
        Dict mapping output type → file path.
    """
    detections_path = Path(detections_path)
    attribution_path = Path(attribution_path)
    out_dir = Path(output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    if stage_output_exists(out_dir, [
        "final_report.pdf", "final_report.geojson", "debris_summary.csv",
    ]):
        return {
            "pdf": out_dir / "final_report.pdf",
            "geojson": out_dir / "final_report.geojson",
            "csv": out_dir / "debris_summary.csv",
        }

    # Load data
    detections_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    if detections_path.exists():
        try:
            detections_gdf = gpd.read_file(detections_path)
        except Exception as exc:
            logger.warning("Could not load detections: %s", exc)

    # Ensure geometry is in EPSG:4326 and recompute centroid lat/lon from
    # the projected geometry so the report always shows geographic coordinates.
    if not detections_gdf.empty:
        if detections_gdf.crs and str(detections_gdf.crs) != "EPSG:4326":
            detections_gdf = detections_gdf.to_crs("EPSG:4326")
        if "geometry" in detections_gdf.columns and detections_gdf.geometry.notna().any():
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                centroids = detections_gdf.geometry.centroid
            detections_gdf["centroid_lon"] = centroids.x
            detections_gdf["centroid_lat"] = centroids.y

    attribution_data = []
    if attribution_path.exists():
        try:
            with open(attribution_path) as fh:
                attribution_data = json.load(fh)
        except Exception as exc:
            logger.warning("Could not load attribution: %s", exc)

    # Load backtrack summary to recover endpoints for unattributed clusters
    backtrack_path = attribution_path.parent / "backtrack_summary.json"
    if backtrack_path.exists():
        try:
            with open(backtrack_path) as fh:
                backtrack_data = json.load(fh)
            
            # Augment attribution data with unattributed endpoints
            attr_cids = {str(a.get("debris_cluster_id")) for a in attribution_data}
            for bt in backtrack_data:
                cid = str(bt.get("cluster_id"))
                if cid not in attr_cids and "source_centroid" in bt:
                    attribution_data.append({
                        "debris_cluster_id": int(cid) if cid.isdigit() else cid,
                        "source_centroid": bt["source_centroid"],
                        "source_type": "N/A",
                        "attribution_score": 0.0,
                    })
                    attr_cids.add(cid)
        except Exception as exc:
            logger.warning("Could not load backtrack summary: %s", exc)

    # Generate detection map
    detection_map = out_dir / "detection_map.png"
    _generate_detection_map(detections_gdf, detection_map, bbox=bbox)

    # Generate RGB map
    rgb_map = out_dir / "rgb_map.png"
    _generate_rgb_map(scene_id, detections_gdf, rgb_map, bbox=bbox)

    # Generate interactive map
    html_map = out_dir / "backtrack_map.html"
    generate_interactive_map(detections_gdf, attribution_data, html_map, attribution_path.parent if attribution_path else None)

    # Generate PDF
    pdf_path = generate_pdf(
        scene_id, detections_gdf, attribution_data, out_dir,
        detection_map_path=detection_map,
        rgb_map_path=rgb_map,
    )

    # Generate GeoJSON summary
    geojson_path = generate_geojson_summary(
        detections_gdf, attribution_data,
        out_dir / "final_report.geojson",
    )

    # Generate CSV
    csv_path = generate_csv(
        detections_gdf, attribution_data, scene_id,
        out_dir / "debris_summary.csv",
    )

    # Print terminal summary
    print_terminal_summary(scene_id, detections_gdf, attribution_data)

    logger.info(
        "[bold green]Stage 7 complete[/] — Reports in %s", out_dir,
    )

    return {
        "pdf": pdf_path,
        "geojson": geojson_path,
        "csv": csv_path,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 7: Generate reports (PDF, GeoJSON, CSV)",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--attribution", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/reports")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = run(
        scene_id=args.scene_id,
        detections_path=args.detections,
        attribution_path=args.attribution,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"\nReports generated:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
