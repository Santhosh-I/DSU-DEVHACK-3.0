# Plastic-Ledger

## Project Overview

Plastic-Ledger is a satellite-imagery platform for detecting marine debris, identifying likely
polymer classes, tracing debris toward possible source regions, and reviewing the results in a
web dashboard.

The project addresses a practical environmental monitoring problem: plastic debris is difficult to
locate across large coastal areas, and a detection alone does not explain where the debris may have
come from. Plastic-Ledger combines image segmentation, spectral classification, ocean and wind
data, particle backtracking, and source attribution so that analysts can move from a satellite
scene to an evidence-based investigation.

## What The System Does

The complete workflow is implemented as seven pipeline stages:

1. **Ingest**: Find and download Sentinel-2 imagery for a selected bounding box and date.
2. **Preprocess**: Prepare satellite bands and image patches for inference.
3. **Detect**: Use the SegFormer model to segment marine debris and other scene classes.
4. **Polymer classification**: Use spectral features and an XGBoost model to classify detections.
5. **Backtracking**: Simulate particle movement using CMEMS currents, ERA5 wind, RK4 integration,
   diffusion, and clustering.
6. **Attribution**: Score possible fishing, industrial, shipping, and river sources using geographic
   and environmental evidence.
7. **Reporting**: Generate JSON, CSV, GeoJSON, map, and report outputs for each run.

## Pipeline Architecture

```text
+---------------------------------------------------------------------+
|                      PLASTIC-LEDGER PIPELINE                       |
+---------------------------------------------------------------------+
|                                                                     |
|  +----------+    +----------+    +----------+    +----------+       |
|  | Stage 1  |--->| Stage 2  |--->| Stage 3  |--->| Stage 4  |       |
|  | Ingest   |    |Preprocess|    | Detect   |    | Polymer  |       |
|  |          |    |          |    | SegFormer|    | XGBoost  |       |
|  | Sentinel |    | Band     |    | + TTA    |    | ML Model |       |
|  | 2 STAC   |    | Reorder  |    | Cluster  |    |          |       |
|  | Download |    | and Tile |    |          |    |          |       |
|  +----------+    +----------+    +----------+    +-----+----+       |
|                                                       |             |
|                                                       v             |
|  +----------+    +----------+    +----------+    +----------+       |
|  | Stage 7  |<---| Stage 6  |<---| Stage 5  |<---| Debris   |       |
|  | Report   |    |Attribute |    |Backtrack |    | Clusters |       |
|  |          |    | Fishing  |    | CMEMS +  |    | + Polymer|       |
|  | PDF      |    | Industry |    | ERA5     |    | Type     |       |
|  | GeoJSON  |    | Shipping |    | RK4      |    +----------+       |
|  | CSV      |    | Rivers   |    | Diffusion|                     |
|  +----------+    +----------+    +----------+                     |
|                                                                     |
+---------------------------------------------------------------------+
```

The dashboard reads run data through the Django API and also presents the bundled evaluation
artifacts under `frontend/public/asserts/`. The Model Outputs page displays SegFormer evaluation
metrics, Polymer XGBoost metrics, and visual output images.

## Main Features

- Create pipeline runs by drawing or entering a geographic bounding box.
- Configure target date, cloud-cover limit, backtracking duration, cluster limit, and run name.
- Track pipeline status from pending through completed or failed.
- Review aggregate KPIs and all available runs on the dashboard.
- Explore detections on interactive Leaflet maps.
- View polymer classifications, confidence, false-positive information, and debris clusters.
- Animate hydrodynamic backtrack paths with selectable tile layers and speeds from `0.5x` to `4x`.
- Inspect source attribution for fishing, industrial, shipping, and river categories.
- Explore hotspot and contributor views across available runs.
- Review reports and download or inspect GeoJSON, CSV, JSON, and map outputs.
- Compare SegFormer and Polymer XGBoost evaluation metrics in the Model Outputs page.

## Technology Stack

### Machine Learning And Data Processing

- Python 3
- PyTorch, torchvision, and segmentation-models-pytorch
- SegFormer for semantic segmentation
- XGBoost with spectral indices for polymer classification
- NumPy, pandas, SciPy, scikit-learn, h5py, and joblib
- rasterio, GeoPandas, Shapely, pyproj, xarray, and netCDF4
- Sentinel-2 access through Copernicus Data Space and STAC tooling
- CMEMS ocean currents and CDS/ERA5 environmental data
- Matplotlib, Rich, and fpdf2 for visualization and reporting

### Backend

- Django 5
- Django REST Framework
- django-cors-headers
- MySQL through PyMySQL
- REST endpoints for pipeline runs, run status, and cluster backtracking

### Frontend

- React 19 with TypeScript and JSX
- Vite 8
- React Router
- Tailwind CSS
- Leaflet and React Leaflet for maps
- Recharts for charts
- Framer Motion for interface animation
- Lucide React for icons

## Project Structure

```text
DSU-DEVHACK-3.0/
|-- README.md                         Project documentation
|-- requirements.txt                  Shared Python dependency specification
|-- template.env                      Environment-variable template
|-- .env.example                      Example environment configuration
|
|-- frontend/                         React/Vite dashboard
|   |-- package.json                   Frontend scripts and dependencies
|   |-- vite.config.ts                 Vite server and /api, /data proxies
|   |-- src/
|   |   |-- App.tsx                    Router and application shell
|   |   |-- components/                Navbar, error boundary, maps, and tabs
|   |   |-- pages/                     Landing, dashboard, tracking, runs, hotspots, model
|   |   |-- services/                  Run and artifact data loading
|   |   |-- lib/                       API clients and shared utilities
|   |   |-- types/                     Shared frontend data contracts
|   |   `-- index.css                  Tailwind layers and application theme
|   `-- public/
|       |-- data/runs/                 Bundled run artifacts used by the dashboard
|       `-- asserts/                   Evaluation JSON and model visual outputs
|
|-- server/                            Django REST backend
|   |-- manage.py                      Django command entry point
|   |-- requirements.txt               Backend environment lock/specification
|   |-- server/                        Django settings, URLs, and WSGI/ASGI files
|   `-- api/                           Run models, serializers, views, URLs, tests
|
|-- src/                               Python pipeline source
|   |-- config/                        Pipeline configuration
|   |-- pipeline/
|   |   |-- run_pipeline.py            Seven-stage orchestration and CLI
|   |   |-- 01_ingest.py               Sentinel-2 ingestion
|   |   |-- 02_preprocess.py            Band and patch preparation
|   |   |-- 03_detect.py                SegFormer detection
|   |   |-- 04_polymer.py               Polymer classification
|   |   |-- 05_backtrack.py             Particle backtracking
|   |   |-- 06_attribute.py             Source attribution
|   |   |-- 07_report.py                Report generation
|   |   `-- utils/                      Caching, configuration, and logging helpers
|   `-- tests/                          Pipeline tests
|
|-- ml_training/                        Training and evaluation scripts
|   |-- polymer/                        XGBoost training/evaluation workflow
|   |-- segformer/                      SegFormer training/evaluation workflow
|   `-- tools/                          Training utilities
|
|-- models/                             Trained model artifacts and feature metadata
|   |-- polymer/                        XGBoost models, labels, and feature names
|   `-- production/                     Production SegFormer and other deployed artifacts
```

## Setup

### 1. Python Environment

From the repository root:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

The backend also provides a pinned environment specification:

```bash
pip install -r server/requirements.txt
```

### 2. Environment Variables

Copy `template.env` to `.env` and fill in the values required for the services you plan to use:

```bash
copy template.env .env                 # Windows
# cp template.env .env                 # macOS/Linux
```

Important variables include:

| Variable | Used for |
|----------|----------|
| `DJANGO_SECRET_KEY` | Django security configuration |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | MySQL connection |
| `COPERNICUS_USERNAME`, `COPERNICUS_PASSWORD` | Sentinel-2 data access |
| `GFW_TOKEN` | Global Fishing Watch source evidence |
| `CDS_API_KEY` | ERA5 wind data |

Do not commit `.env` or real credentials.

### 3. Start The Backend

From the repository root:

```bash
python server/manage.py migrate
python server/manage.py runserver 8000
```

The frontend Vite configuration proxies `/api` and `/data` requests to `http://localhost:8000`.

### 4. Start The Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

Useful frontend commands:

```bash
npm run build
npm run lint
npm run preview
```

## Running The Pipeline

The current CLI is `src/pipeline/run_pipeline.py` and uses `--date` for the target date:

```bash
python src/pipeline/run_pipeline.py \
    --bbox "80.0,6.0,82.0,8.0" \
    --date "2024-01-31" \
    --output_dir "data/runs/sri_lanka_jan24" \
    --model_path "models/production/best_model_SegFormer_v2.pth" \
    --cloud_cover 20 \
    --backtrack_days 30
```

Optional controls include:

```text
--skip_stages 1,2       Reuse existing ingestion or preprocessing outputs
--max_clusters 10       Limit the number of clusters to backtrack
--cleanup_patches       Remove cached patches after polymer classification
--config path/to/file   Use a custom YAML configuration
--no-bbox-filter        Process the full scene instead of the selected bbox
```

The pipeline expects the bounding box in this order:
`lon_min,lat_min,lon_max,lat_max`.

## Dashboard Routes

| Route | Purpose |
|-------|---------|
| `/` | Landing page and product entry point |
| `/dashboard` | Pipeline runs and aggregate KPIs |
| `/tracking` | Create a run and select an area on a map |
| `/runs/:id` | Detailed run tabs for overview, detection, attribution, analytics, and reports |
| `/hotspots` | Cross-run hotspot and source-contributor analysis |
| `/model` | SegFormer and Polymer evaluation metrics plus visual outputs |

## Run Outputs

Each pipeline run stores artifacts under a run directory, commonly `data/runs/<run_id>/`:

| Output | Purpose |
|--------|---------|
| `run_summary.json` | Target date, stages, model path, outputs, and timing |
| `ingest_metadata.json` | Bounding box, date range, cloud-cover limit, and scenes |
| `detections/` | Raw and classified detection GeoJSON files |
| `attribution/` | Backtrack trajectories, source scores, and metadata |
| `reports/` | Final GeoJSON, debris CSV, maps, and generated reports |
| `backtrack_*.geojson` | Per-cluster particle trajectories |
| `attribution_report.json` | Ranked source attribution results |
| `debris_summary.csv` | Flat cluster-level detection summary |

The repository includes a sample run under `frontend/public/data/runs/run_001/` for dashboard
development and demonstration.

## Model Evaluation Assets

The Model Outputs page reads these checked-in files:

- `frontend/public/asserts/segformer_eval_results/evaluation_results.json`
- `frontend/public/asserts/segformer_eval_results/evaluation_report.md`
- `frontend/public/asserts/polymer/polymer_xgb_model_eval.json`
- `frontend/public/asserts/vis_output/*.png`

The SegFormer data contains overall mIoU, marine-debris precision/recall/F1, and per-class IoU.
The Polymer data contains accuracy, macro and weighted averages, and per-class precision, recall,
F1 score, and support.

## Testing

Backend and pipeline tests are located in `server/api/tests.py` and `src/tests/`. Run the available
Python tests from the repository root with:

```bash
python -m pytest server/api src/tests -v
```

For frontend validation:

```bash
cd frontend
npm run build
npm run lint
```

## Data And Credentials

Sentinel-2, CMEMS, ERA5/CDS, and Global Fishing Watch data are supplied by external services and
have their own access terms. Keep credentials out of source control and verify the applicable
licenses before redistributing imagery, trained weights, or derived datasets.

## License And Intended Use

Plastic-Ledger is intended for research, environmental analysis, and demonstration. Detection and
source-attribution outputs should be treated as decision support rather than definitive proof of
responsibility or origin.
