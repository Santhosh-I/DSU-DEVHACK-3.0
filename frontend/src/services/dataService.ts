import type {
  RunSummary,
  DetectionFeatureCollection,
  AttributionEntry,
  BacktrackEntry,
  RunMetadata,
  DebrisSummaryRow,
  IngestMetadata,
} from "@/types";

const getBase = (runId: string) => `/data/runs/${runId}`;

export function getSceneId(summary: RunSummary): string | null {
  const scenePath = summary.outputs.raw_scenes?.[0];
  if (!scenePath) return null;
  return scenePath.split(/[\\/]/).pop() || null;
}

async function fetchJson<T>(path: string, fallback?: T): Promise<T> {
  try {
    const res = await fetch(path);
    if (!res.ok) {
      if (fallback !== undefined) return fallback;
      throw new Error(`Failed to fetch ${path}: ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    if (fallback !== undefined) return fallback;
    throw err;
  }
}

export async function loadRunSummary(runId: string): Promise<RunSummary> {
  return fetchJson<RunSummary>(`${getBase(runId)}/run_summary.json`);
}

export async function loadDetections(runId: string): Promise<DetectionFeatureCollection> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<DetectionFeatureCollection>(`${getBase(runId)}/detections/${sceneId}/detections_classified.geojson`, { type: "FeatureCollection", features: [] } as any);
}

export async function loadFinalReport(runId: string): Promise<DetectionFeatureCollection> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<DetectionFeatureCollection>(`${getBase(runId)}/reports/${sceneId}/final_report.geojson`, { type: "FeatureCollection", features: [] } as any);
}

export async function loadAttribution(runId: string): Promise<AttributionEntry[]> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<AttributionEntry[]>(`${getBase(runId)}/attribution/${sceneId}/attribution_report.json`, []);
}

export async function loadBacktrackSummary(runId: string): Promise<BacktrackEntry[]> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<BacktrackEntry[]>(`${getBase(runId)}/attribution/${sceneId}/backtrack_summary.json`, []);
}

export async function loadRunMetadata(runId: string): Promise<RunMetadata | null> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<RunMetadata | null>(`${getBase(runId)}/attribution/${sceneId}/run_metadata.json`, null);
}

export async function loadIngestMetadata(runId: string): Promise<IngestMetadata> {
  return fetchJson<IngestMetadata>(`${getBase(runId)}/raw/ingest_metadata.json`);
}

export async function loadBacktrackGeoJson(runId: string, sceneId: string, clusterId: number): Promise<any> {
  return fetchJson<any>(`${getBase(runId)}/attribution/${sceneId}/backtrack_${clusterId}.geojson`);
}

export async function loadAllBacktrackGeoJsons(clusterIds: number[], runId: string): Promise<Record<number, any>> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  if (!sceneId) return {};

  const results: Record<number, any> = {};
  await Promise.all(
    clusterIds.map(async (id) => {
      try {
        const geojson = await loadBacktrackGeoJson(runId, sceneId, id);
        results[id] = geojson;
      } catch (err) {
        console.warn(`Could not load backtrack GeoJSON for cluster ${id}`, err);
      }
    })
  );
  return results;
}

export async function loadDebrisSummaryCsv(runId: string): Promise<DebrisSummaryRow[]> {
  try {
    const summary = await loadRunSummary(runId);
    const sceneId = getSceneId(summary);
    if (!sceneId) return [];
    const res = await fetch(`${getBase(runId)}/reports/${sceneId}/debris_summary.csv`);
    if (!res.ok) return [];
    const text = await res.text();
    const lines = text.trim().split(/\r?\n/);
    if (lines.length <= 1) return [];
    const headers = lines[0].split(",").map((h) => h.trim());
  
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: any = {};
    headers.forEach((h, i) => {
      const v = values[i]?.trim() ?? "";
      if (["cluster_id", "lat", "lon", "area_sq_m", "confidence", "attribution_score", "source_lat", "source_lon"].includes(h)) {
        row[h] = v === "" ? 0 : parseFloat(v);
      } else {
        row[h] = v;
      }
    });
    return row as DebrisSummaryRow;
  });
  } catch {
    return [];
  }
}

// Report file paths generator
export const getReportFiles = (runId: string, sceneId: string) => ({
  pdf: `${getBase(runId)}/reports/${sceneId}/final_report.pdf`,
  geojson: `${getBase(runId)}/reports/${sceneId}/final_report.geojson`,
  csv: `${getBase(runId)}/reports/${sceneId}/debris_summary.csv`,
  backtrackMap: `${getBase(runId)}/reports/${sceneId}/backtrack_map.html`,
  detectionMap: `${getBase(runId)}/reports/${sceneId}/detection_map.png`,
  polymerDist: `${getBase(runId)}/reports/${sceneId}/polymer_distribution.png`,
  rgbMap: `${getBase(runId)}/reports/${sceneId}/rgb_map.png`,
});
