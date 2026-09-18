export interface PipelineRun {
  id: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";
  bbox: string;
  target_date: string;
  cloud_cover: number;
  backtrack_days: number;
  max_clusters: number;
  process_all_patches: boolean;
  run_name: string;
  output_dir: string | null;
  created_at: string;
  completed_at: string | null;
  summary: RunSummary | null;
  error_message: string | null;
}

// ─── Run Summary (run_summary.json) ───
export interface RunSummary {
  bbox: [number, number, number, number];
  target_date: string;
  model_path: string;
  stages_completed: number[];
  stages_skipped: number[];
  stages_failed: number[];
  outputs: {
    raw_scenes: string[];
    polymer_counts: Record<string, number>;
    reports: {
      pdf: string;
      geojson: string;
      csv: string;
    };
  };
  scene_dates: Record<string, string>;
  elapsed_seconds: number;
}

// ─── GeoJSON Detection Feature ───
export interface DetectionProperties {
  area_m2: number;
  mean_confidence: number;
  centroid_lon: number;
  centroid_lat: number;
  detection_date: string;
  cluster_id: number;
  polymer_type: string;
  pi_value: number;
  sr_value: number;
  nsi_value: number;
  fdi_value: number;
  is_false_positive: boolean;
  rf_confidence: number;
  source_type?: string;
  attribution_score?: string;
  explanation?: string;
  country?: string;
}

export interface DetectionFeature {
  type: "Feature";
  properties: DetectionProperties;
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
}

export interface DetectionFeatureCollection {
  type: "FeatureCollection";
  features: DetectionFeature[];
}

// ─── Attribution ───
export interface AttributionEntry {
  debris_cluster_id: number;
  source_rank: number;
  source_type: string;
  location_name: string;
  country: string;
  attribution_score: number;
  confidence: string;
  explanation: string;
  source_centroid: [number, number];
  source_bbox: [number, number, number, number];
  source_probability: number;
  days_to_source: number;
  fishing_score: number;
  industrial_score: number;
  shipping_score: number;
  river_score: number;
  vessel_ids: string[];
}

// ─── Backtrack Summary ───
export interface BacktrackEntry {
  source_centroid: [number, number];
  source_bbox: [number, number, number, number];
  source_probability: number;
  n_particles: number;
  cluster_id: number;
  days_to_source: number;
}

// ─── Run Metadata (attribution config) ───
export interface RunMetadata {
  cmems_product: string;
  era5_product: string;
  integrator: string;
  time_step_hours: number;
  horizontal_diffusion_kh: number;
  n_particles: number;
  bt_days: number;
  kernels: string[];
  timestamp_utc: string;
}

// ─── CSV Row from debris_summary.csv ───
export interface DebrisSummaryRow {
  cluster_id: number;
  lat: number;
  lon: number;
  area_sq_m: number;
  polymer_type: string;
  confidence: number;
  top_source_type: string;
  source_lat: number;
  source_lon: number;
  top_source_location: string;
  top_source_country: string;
  attribution_score: number;
  detection_date: string;
  scene_id: string;
}

// ─── Ingest Metadata ───
export interface IngestMetadata {
  bbox: [number, number, number, number];
  date_start: string;
  date_end: string;
  cloud_cover_max: number;
  scenes: string[];
  scene_paths: string[];
}

// ─── Pipeline stages ───
export const PIPELINE_STAGES = [
  { id: 1, name: "Scene Download", short: "Ingest" },
  { id: 2, name: "Preprocessing", short: "Preprocess" },
  { id: 3, name: "Detection", short: "Detect" },
  { id: 4, name: "Polymer Classification", short: "Polymer" },
  { id: 5, name: "False Positive Filter", short: "Filter" },
  { id: 6, name: "Source Attribution", short: "Attribute" },
  { id: 7, name: "Report Generation", short: "Report" },
] as const;

// ─── Polymer type colors ───
export const POLYMER_COLORS: Record<string, string> = {
  "Marine Debris (Plastic)": "#EF4444",
  "False Positive (Wake)": "#6B7280",
  "False Positive (Water)": "#3B82F6",
  "False Positive (Cloud)": "#9CA3AF",
  "False Positive (Turbid Water)": "#8B5CF6",
  "False Positive (Shallow Water)": "#06B6D4",
  "False Positive (Ship)": "#F59E0B",
  "False Positive (Cloud Shadow)": "#4B5563",
  "Organic Matter (Foam)": "#10B981",
};

// ─── Source type icons ───
export const SOURCE_ICONS: Record<string, string> = {
  fishing: "🎣",
  industrial: "🏭",
  shipping: "🚢",
  river: "🏞️",
};
