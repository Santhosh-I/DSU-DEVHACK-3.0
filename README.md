# 🌊 Plastic-Ledger

**Autonomous Micro-Plastic Fingerprinting & Source Attribution from Satellite Imagery**

Plastic-Ledger is an end-to-end Python pipeline that detects marine plastic debris in Sentinel-2
satellite imagery, classifies the polymer type, traces debris back to its source using ocean
current simulations, and generates comprehensive attribution reports. It uses a SegFormer deep
learning model trained on the [MARIDA](https://github.com/marine-debris/marine-debris.github.io)
dataset for segmentation, combined with XGBoost spectral analysis, Lagrangian particle tracking, and
multi-source geospatial attribution.

---

## 🔬 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PLASTIC-LEDGER PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Stage 1  │───▶│ Stage 2  │───▶│ Stage 3  │───▶│ Stage 4  │      │
│  │ Ingest   │    │Preprocess│    │ Detect   │    │ Polymer  │      │
│  │          │    │          │    │          │    │ Classify │      │
│  │ Sentinel │    │ Band     │    │ SegFormer│    │ XGBoost  │      │
│  │ 2 STAC   │    │ Reorder  │    │ + TTA    │    │ ML Model │      │
│  │ Download │    │ Offset   │    │ Cluster  │    │          │      │
│  └──────────┘    │ Tile     │    └──────────┘    └─────┬────┘      │
│                  └──────────┘                          │            │
│                                                        ▼            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Stage 7  │◀───│ Stage 6  │◀───│ Stage 5  │◀───│          │      │
│  │ Report   │    │Attribute │    │Backtrack │    │ Debris   │      │
│  │          │    │          │    │          │    │ Clusters │      │
│  │ PDF      │    │ Fishing  │    │ CMEMS +  │    │ + Polymer│      │
│  │ GeoJSON  │    │ Industry │    │ ERA5     │    │ Type     │      │
│  │ CSV      │    │ Shipping │    │ RK4      │    └──────────┘      │
│  │ Terminal │    │ Rivers   │    │ DBSCAN   │                      │
│  └──────────┘    └──────────┘    └──────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Plastic-Ledger.git
cd Plastic-Ledger

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API credentials
cp .env.example .env
# Edit .env with your API keys (see "API Keys" section below)

# 4. Run the pipeline on a test area (Sri Lanka coast)
python pipeline/run_pipeline.py \
    --bbox "80.5,7.5,81.5,8.5" \
    --start_date "2024-01-10" \
    --end_date "2024-01-15" \
    --output_dir "data/runs/test_run" \
    --model_path "ml_training_2.0/SegFormer/training-log/run_1/best_model.pth" \
    --cloud_cover 20 \
    --backtrack_days 7

# 5. Check output
ls data/runs/test_run/reports/
```

---

## 🔑 API Keys Setup

The pipeline requires credentials for external data sources. All keys are stored in `.env`:

| Service | Purpose | Sign Up |
|---------|---------|---------|
| **Copernicus Data Space** | Sentinel-2 satellite imagery download | [dataspace.copernicus.eu](https://dataspace.copernicus.eu) (free) |
| **CMEMS** | Ocean current data (surface velocity) | [marine.copernicus.eu](https://marine.copernicus.eu) (free) |
| **CDS API** | ERA5 wind data for drift calculation | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) (free) |
| **Global Fishing Watch** | Fishing vessel positions | [globalfishingwatch.org](https://globalfishingwatch.org/data/) (free academic) |

```bash
# .env file format:
COPERNICUS_USERNAME=your_email@example.com
COPERNICUS_PASSWORD=your_password
GFW_TOKEN=your_global_fishing_watch_token
CDS_API_KEY=your_climate_data_store_key
# Optional: cap ERA5/CDS retry behavior (Stage 5)
CDS_RETRY_MAX=3
CDS_SLEEP_MAX=10
CDS_TIMEOUT=60
```

> **Note**: The pipeline runs in graceful degradation mode — if API keys are missing, it will
> use heuristic fallbacks for scoring (Stages 5 and 6) and skip data downloads.

---

## 📋 Example CLI Command

```bash
# Full pipeline — Sri Lanka coast, January 2024
python pipeline/run_pipeline.py \
    --bbox "80.0,6.0,82.0,8.0" \
    --start_date "2024-01-01" \
    --end_date "2024-01-31" \
    --output_dir "data/runs/sri_lanka_jan24" \
    --model_path "ml_training_2.0/SegFormer/training-log/run_1/best_model.pth" \
    --cloud_cover 20 \
    --backtrack_days 30

# Skip stages 1 and 2 (if data is already downloaded & preprocessed)
python pipeline/run_pipeline.py \
    --bbox "80.0,6.0,82.0,8.0" \
    --start_date "2024-01-01" \
    --end_date "2024-01-31" \
    --output_dir "data/runs/sri_lanka_jan24" \
    --skip_stages "1,2"
```

Each stage can also be run independently:

```bash
# Run just Stage 3 (detection) on pre-existing patches
python -m pipeline.03_detect \
    --scene_id S2A_MSIL2A_20240115 \
    --patches_dir data/processed/S2A_MSIL2A_20240115/patches \
    --model_path ml_training_2.0/SegFormer/training-log/run_1/best_model.pth
```

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `final_report.pdf` | Executive summary with detection maps, polymer charts, and attribution tables |
| `final_report.geojson` | All detections with attribution data as GeoJSON (for GIS tools) |
| `debris_summary.csv` | Flat CSV with one row per debris cluster |
| `debris_mask.tif` | Binary GeoTIFF mask of detected debris pixels |
| `debris_prob.tif` | Float32 GeoTIFF probability map of debris class |
| `class_mask.tif` | Full 15-class prediction mask (uint8 GeoTIFF) |
| `detections.geojson` | Raw debris detection polygons with area and confidence |
| `detections_classified.geojson` | Detections with polymer type classification |
| `backtrack_*.geojson` | Particle trajectories per debris cluster |
| `attribution_report.json` | Source attribution scores and explanations |
| `run_summary.json` | Pipeline run metadata and timing |

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

Tests use mocked data and do not require API keys, GPU, or downloaded imagery.

---

## 📄 License

This project is for research and educational purposes. The MARIDA dataset and Sentinel-2
imagery are subject to their respective licenses (Copernicus Data Space EULA).
