import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import * as turf from "@turf/turf";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { getPipelineRuns } from "@/lib/api";
import { loadFinalReport, loadBacktrackSummary, loadAttribution, loadRunSummary, getSceneId } from "@/services/dataService";
import { PipelineRun, AttributionEntry, BacktrackEntry } from "@/types";
import { CloudHeatmapLayer, HeatmapPoint } from "@/components/CloudHeatmapLayer";
import {
  Flame, Filter, SlidersHorizontal, X, MapPin, Calendar,
  ChevronDown, ChevronUp, Crosshair, Activity, TrendingUp,
  RefreshCw, AlertTriangle, Layers, Target, Compass, Anchor,
  Eye, EyeOff, Radio, Navigation, Award, Zap, CheckCircle2,
  CalendarRange, ArrowRight, RotateCcw, Map as MapIcon, Waves,
  GitCommit, ArrowUpRight,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface GeographicAreaInfo {
  name: string;
  type: "Marina Coast" | "River Estuary" | "Bay Waters" | "Harbor / Port" | "Reef Lagoon" | "Marine Reserve" | "Coastal Sector";
  badgeColor: string;
}

interface DetectionPoint {
  lat: number;
  lng: number;
  confidence: number;
  polymer_type: string;
  is_false_positive: boolean;
  detection_date: string; // YYYY-MM-DD
  area_m2: number;
  run_id: string;
  run_name: string;
  cluster_id: number;
  area_name: string;
  area_type: string;
}

export interface AttributedSourcePoint {
  id: string;
  lat: number;
  lng: number;
  location_name: string;
  country: string;
  source_type: string;
  attribution_score: number;
  confidence: string;
  explanation: string;
  days_to_source: number;
  cluster_id: number;
  run_id: string;
  run_name: string;
  detection_date: string; // YYYY-MM-DD
}

interface Hotspot {
  id: number;
  label: string;
  center: [number, number]; // [lat, lng]
  detection_count: number;
  run_count: number;
  avg_confidence: number;
  total_area_m2: number;
  first_seen: string;
  last_seen: string;
  points: DetectionPoint[];
  area_name: string;
  area_type: string;
  area_badge_color: string;
}

export interface AreaContributor {
  id: string;
  areaName: string;
  areaType: string;
  badgeColor: string;
  detectionCount: number;
  percentOfTotal: number;
  clusterCount: number;
  clusters: Hotspot[];
  center: [number, number];
  bounds: [[number, number], [number, number]];
  totalAreaM2: number;
  avgConfidence: number;
  runCount: number;
}

export interface SourceContributor {
  id: string;
  sourceName: string;
  sourceType: string;
  country: string;
  badgeColor: string;
  center: [number, number];
  detectionCount: number;
  percentOfTotal: number;
  clusterCount: number;
  clusters: Hotspot[];
  attributionScore: number;
  daysToSource: number;
  explanation: string;
  runCount: number;
}

export interface AttributionLine {
  id: string;
  clusterId: number;
  clusterLabel: string;
  clusterCenter: [number, number];
  sourceId: string;
  sourceName: string;
  sourceType: string;
  sourceCenter: [number, number];
  score: number;
  days: number;
  distanceKm: number;
}

interface ActivityTimelinePoint {
  date: string;
  detections: number;
  label: string;
}

// ─── Geographic Area Geocoder ────────────────────────────────────────────────

export function getGeographicArea(lat: number, lon: number): GeographicAreaInfo {
  // Gulf of Honduras / Belize / Guatemala / Honduras coastal zones
  if (lat >= 15.60 && lat <= 15.88 && lon >= -88.25 && lon <= -88.05) {
    return { name: "Motagua River Marina & Estuary", type: "River Estuary", badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" };
  }
  if (lat >= 15.85 && lat <= 16.12 && lon >= -88.30 && lon <= -88.02) {
    return { name: "Omoa Bay & Puerto Cortés Marina", type: "Marina Coast", badgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30" };
  }
  if (lat >= 15.70 && lat <= 15.95 && lon >= -88.45 && lon < -88.25) {
    return { name: "Puerto Barrios Harbor & Marina", type: "Harbor / Port", badgeColor: "bg-blue-500/20 text-blue-300 border-blue-500/30" };
  }
  if (lat >= 15.75 && lat <= 16.05 && lon >= -88.75 && lon <= -88.45) {
    return { name: "Livingston & Rio Dulce Marina Inflow", type: "Marina Coast", badgeColor: "bg-teal-500/20 text-teal-300 border-teal-500/30" };
  }
  if (lat >= 15.85 && lat <= 16.15 && lon >= -88.45 && lon <= -88.20) {
    return { name: "Amatique Bay Coastal Waters", type: "Bay Waters", badgeColor: "bg-sky-500/20 text-sky-300 border-sky-500/30" };
  }
  if (lat >= 15.70 && lat <= 16.15 && lon >= -88.15 && lon <= -87.80) {
    return { name: "Puerto Cortés Coastal Strip", type: "Marina Coast", badgeColor: "bg-indigo-500/20 text-indigo-300 border-indigo-500/30" };
  }
  if (lat >= 16.00 && lat <= 16.35 && lon >= -88.60 && lon <= -88.25) {
    return { name: "Port Honduras Marine Reserve", type: "Marine Reserve", badgeColor: "bg-amber-500/20 text-amber-300 border-amber-500/30" };
  }
  if (lat >= 16.05 && lat <= 16.45 && lon >= -88.25 && lon <= -88.00) {
    return { name: "Sapodilla Cayes Marina Channel", type: "Reef Lagoon", badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/30" };
  }
  if (lat >= 16.35 && lat <= 16.70 && lon >= -88.50 && lon <= -88.15) {
    return { name: "Placencia Peninsula Marina Coast", type: "Marina Coast", badgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30" };
  }
  if (lat >= 16.35 && lat <= 16.70 && lon >= -88.15 && lon <= -87.85) {
    return { name: "South Water Caye Marine Zone", type: "Reef Lagoon", badgeColor: "bg-violet-500/20 text-violet-300 border-violet-500/30" };
  }
  if (lat >= 16.70 && lat <= 17.30 && lon >= -88.40 && lon <= -88.05) {
    return { name: "Belize Barrier Reef Lagoon", type: "Reef Lagoon", badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/30" };
  }
  if (lat >= 16.70 && lat <= 17.30 && lon >= -88.05 && lon <= -87.75) {
    return { name: "Turneffe Atoll Marine Barrier", type: "Reef Lagoon", badgeColor: "bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-500/30" };
  }
  if (lat >= 5.3 && lat <= 6.0 && lon >= -0.5 && lon <= 1.0) {
    return { name: "Gulf of Guinea / Accra Marina Coast", type: "Marina Coast", badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" };
  }
  if (lat >= 7.0 && lat <= 10.5 && lon >= 79.0 && lon <= 82.5) {
    return { name: "Sri Lanka Coastal Waters", type: "Marina Coast", badgeColor: "bg-sky-500/20 text-sky-300 border-sky-500/30" };
  }

  const ew = lon < 0 ? "W" : "E";
  const ns = lat >= 0 ? "N" : "S";
  return {
    name: `Coastal Sector (${Math.abs(lat).toFixed(1)}°${ns}, ${Math.abs(lon).toFixed(1)}°${ew})`,
    type: "Coastal Sector",
    badgeColor: "bg-slate-500/20 text-slate-300 border-slate-500/30",
  };
}

// ─── Tile Configuration (Watermark-free, reliable providers) ──────────────────

type BaseMapMode = "satellite" | "dark" | "light";

const TILE_LAYERS: Record<BaseMapMode, { label: string; url: string; attribution: string }> = {
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
  },
  dark: {
    label: "Dark",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
  },
  light: {
    label: "Light",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
  },
};

const CLUSTER_RADIUS_KM = 0.5; // km radius for hotspot clustering

function intensityColor(normalized: number): string {
  if (normalized > 0.75) return "#ef4444";
  if (normalized > 0.5) return "#f97316";
  if (normalized > 0.25) return "#facc15";
  return "#06b6d4";
}

function formatDate(d: string) {
  if (!d) return "—";
  try {
    return new Date(d).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  } catch {
    return d;
  }
}

/** Spatial clustering using Turf */
function clusterPoints(points: DetectionPoint[], radiusKm: number): Hotspot[] {
  if (points.length === 0) return [];

  const features = points.map((p, i) =>
    turf.point([p.lng, p.lat], { idx: i })
  );

  const clustered: { centroid: [number, number]; pts: DetectionPoint[] }[] = [];
  const used = new Set<number>();

  features.forEach((feat, i) => {
    if (used.has(i)) return;
    const group: DetectionPoint[] = [points[i]];
    used.add(i);

    features.forEach((other, j) => {
      if (used.has(j)) return;
      const dist = turf.distance(feat, other, { units: "kilometers" });
      if (dist <= radiusKm) {
        group.push(points[j]);
        used.add(j);
      }
    });

    const avgLat = group.reduce((s, p) => s + p.lat, 0) / group.length;
    const avgLng = group.reduce((s, p) => s + p.lng, 0) / group.length;
    clustered.push({ centroid: [avgLat, avgLng], pts: group });
  });

  // Sort by detection count descending
  clustered.sort((a, b) => b.pts.length - a.pts.length);

  return clustered.map((c, idx) => {
    const dates = c.pts.map(p => p.detection_date).filter(Boolean).sort();
    const runs = new Set(c.pts.map(p => p.run_id));
    const avgConf =
      c.pts.reduce((s, p) => s + p.confidence, 0) / (c.pts.length || 1);
    const totalArea = c.pts.reduce((s, p) => s + p.area_m2, 0);
    const areaInfo = getGeographicArea(c.centroid[0], c.centroid[1]);

    return {
      id: idx + 1,
      label: `Cluster ${String.fromCharCode(65 + idx)}`,
      center: c.centroid,
      detection_count: c.pts.length,
      run_count: runs.size,
      avg_confidence: avgConf,
      total_area_m2: totalArea,
      first_seen: dates[0] || "",
      last_seen: dates[dates.length - 1] || "",
      points: c.pts,
      area_name: areaInfo.name,
      area_type: areaInfo.type,
      area_badge_color: areaInfo.badgeColor,
    };
  });
}

/** Build activity timeline data from points within range */
function buildActivityTimeline(points: DetectionPoint[]): ActivityTimelinePoint[] {
  if (points.length === 0) return [];

  const countsByDate: Record<string, number> = {};
  points.forEach(p => {
    if (!p.detection_date) return;
    const d = p.detection_date;
    countsByDate[d] = (countsByDate[d] || 0) + 1;
  });

  const sortedDates = Object.keys(countsByDate).sort();
  return sortedDates.map(date => ({
    date,
    detections: countsByDate[date],
    label: formatDate(date),
  }));
}

// ─── Map Controller ───────────────────────────────────────────────────────────

function MapController({ flyTo }: { flyTo: [number, number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (flyTo) {
      map.flyTo([flyTo[0], flyTo[1]], flyTo[2], { duration: 1.2 });
    }
  }, [flyTo, map]);
  return null;
}

// ─── Main Page ────────────────────────────────────────────────────────────────

const HotspotsPage: React.FC = () => {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [allDetections, setAllDetections] = useState<DetectionPoint[]>([]);
  const [allSources, setAllSources] = useState<AttributedSourcePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState("");

  // Base map mode - DEFAULT IS SATELLITE
  const [tileKey, setTileKey] = useState<BaseMapMode>("satellite");

  // Historical bounds (overall available dates)
  const [earliestDate, setEarliestDate] = useState<string>("");
  const [latestDate, setLatestDate] = useState<string>("");

  // Date Range state
  const [fromDateInput, setFromDateInput] = useState<string>("");
  const [toDateInput, setToDateInput] = useState<string>("");
  const [appliedRange, setAppliedRange] = useState<{ start: string; end: string } | null>(null);

  // Locations View Mode: Grouped Coastal Areas vs Grouped by Source Attribution
  const [activeLocationsView, setActiveLocationsView] = useState<"coastal_areas" | "source_attribution">("coastal_areas");

  // Filters
  const [minConf, setMinConf] = useState(0);
  const [showFP, setShowFP] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [typeFilter, setTypeFilter] = useState("all");

  // Cloud Heatmap appearance controls
  const [showCloudHeatmap, setShowCloudHeatmap] = useState(true);
  const [cloudRadius, setCloudRadius] = useState(34);
  const [cloudOpacity, setCloudOpacity] = useState(0.85);

  // Source attribution controls & Attribution Lines Toggle
  const [showSources, setShowSources] = useState(true);
  const [showAttributionLines, setShowAttributionLines] = useState(true);

  // UI state
  const [selectedHotspot, setSelectedHotspot] = useState<Hotspot | null>(null);
  const [selectedSource, setSelectedSource] = useState<AttributedSourcePoint | null>(null);
  const [flyTo, setFlyTo] = useState<[number, number, number] | null>(null);
  const [expandedTable, setExpandedTable] = useState(false);

  // ── Load data from all completed runs ──────────────────────────────────────
  useEffect(() => {
    async function loadAllData() {
      setLoading(true);
      try {
        const allRuns = await getPipelineRuns();
        const completed = allRuns.filter(r => r.status === "COMPLETED");
        setRuns(completed);

        const detections: DetectionPoint[] = [];
        const sources: AttributedSourcePoint[] = [];

        for (const run of completed) {
          setLoadingProgress(`Loading ${run.run_name || run.id.substring(0, 8)}…`);
          try {
            const [reportGeoJson, summary, attributionList, btSummary] = await Promise.all([
              loadFinalReport(run.id).catch(() => ({ type: "FeatureCollection", features: [] })),
              loadRunSummary(run.id).catch(() => null),
              loadAttribution(run.id).catch(() => [] as AttributionEntry[]),
              loadBacktrackSummary(run.id).catch(() => [] as BacktrackEntry[]),
            ]);

            // Get detection date from scene dates or run target_date
            const sceneId = summary ? getSceneId(summary) : null;
            let rawDetectionDate =
              sceneId && summary?.scene_dates
                ? summary.scene_dates[sceneId] || summary?.target_date
                : summary?.target_date;
            if (!rawDetectionDate && run.created_at) {
              rawDetectionDate = run.created_at;
            }

            const cleanDate = (rawDetectionDate || "2020-09-18").substring(0, 10);

            // Extract detection points
            if (reportGeoJson?.features) {
              reportGeoJson.features.forEach(feat => {
                const p = feat.properties;
                const coords = feat.geometry.type === "Polygon"
                  ? feat.geometry.coordinates[0][0] as [number, number]
                  : [0, 0] as [number, number];
                
                const lat = p.centroid_lat || coords[1];
                const lng = p.centroid_lon || coords[0];
                const pointDate = (p.detection_date || cleanDate).substring(0, 10);
                const areaInfo = getGeographicArea(lat, lng);

                detections.push({
                  lat,
                  lng,
                  confidence: p.mean_confidence || 0,
                  polymer_type: p.polymer_type || "Unknown",
                  is_false_positive: p.is_false_positive || false,
                  detection_date: pointDate,
                  area_m2: p.area_m2 || 0,
                  run_id: run.id,
                  run_name: run.run_name || run.id.substring(0, 8),
                  cluster_id: p.cluster_id || 0,
                  area_name: areaInfo.name,
                  area_type: areaInfo.type,
                });
              });
            }

            // Extract rich Source Attributed Places (Backtracking origin)
            if (attributionList && attributionList.length > 0) {
              attributionList.forEach(entry => {
                if (entry.source_centroid && entry.source_centroid.length === 2) {
                  sources.push({
                    id: `${run.id}-${entry.debris_cluster_id}-${entry.source_rank || 1}`,
                    lat: entry.source_centroid[1],
                    lng: entry.source_centroid[0],
                    location_name: entry.location_name || `Source Region (${entry.source_centroid[1].toFixed(2)}°N, ${entry.source_centroid[0].toFixed(2)}°W)`,
                    country: entry.country || "International / Coastal",
                    source_type: entry.source_type || "land-outflow",
                    attribution_score: entry.attribution_score ?? entry.source_probability ?? 0.5,
                    confidence: entry.confidence || "medium",
                    explanation: entry.explanation || "Identified source through oceanographic reverse trajectory tracking.",
                    days_to_source: entry.days_to_source ?? 7.0,
                    cluster_id: entry.debris_cluster_id,
                    run_id: run.id,
                    run_name: run.run_name || run.id.substring(0, 8),
                    detection_date: cleanDate,
                  });
                }
              });
            } else if (btSummary && btSummary.length > 0) {
              btSummary.forEach((bt, idx) => {
                if (bt.source_centroid && bt.source_centroid.length === 2) {
                  sources.push({
                    id: `${run.id}-bt-${bt.cluster_id || idx}`,
                    lat: bt.source_centroid[1],
                    lng: bt.source_centroid[0],
                    location_name: `Backtrack Origin (${bt.source_centroid[1].toFixed(2)}°N, ${bt.source_centroid[0].toFixed(2)}°W)`,
                    country: "Attributed Coastal / Marine Area",
                    source_type: "hydrodynamic-drift",
                    attribution_score: bt.source_probability || 0.75,
                    confidence: "medium",
                    explanation: `Backtracked reverse drift trajectory ~${bt.days_to_source || 7} days prior to detection.`,
                    days_to_source: bt.days_to_source || 7.0,
                    cluster_id: bt.cluster_id,
                    run_id: run.id,
                    run_name: run.run_name || run.id.substring(0, 8),
                    detection_date: cleanDate,
                  });
                }
              });
            }

          } catch (e) {
            console.warn(`Could not load full report for run ${run.id}:`, e);
          }
        }

        setAllDetections(detections);
        setAllSources(sources);

        // Find chronological bounds
        const dates = detections.map(d => d.detection_date).filter(Boolean).sort();
        if (dates.length > 0) {
          const earliest = dates[0];
          const latest = dates[dates.length - 1];
          setEarliestDate(earliest);
          setLatestDate(latest);
          setFromDateInput(earliest);
          setToDateInput(latest);
          setAppliedRange({ start: earliest, end: latest });
        }
      } catch (err) {
        console.error("Failed to load historical data:", err);
      } finally {
        setLoading(false);
        setLoadingProgress("");
      }
    }

    loadAllData();
  }, []);

  // ── Handle Date Range Apply & Reset ────────────────────────────────────────
  const handleApplyDateRange = () => {
    if (!fromDateInput && !toDateInput) {
      setAppliedRange(null);
      return;
    }
    const start = fromDateInput || earliestDate;
    const end = toDateInput || latestDate;
    setAppliedRange({ start, end });
  };

  const handleResetDateRange = () => {
    setFromDateInput(earliestDate);
    setToDateInput(latestDate);
    setAppliedRange({ start: earliestDate, end: latestDate });
  };

  // ── Filter detections strictly by applied date range and user filters ──────
  const filteredDetections = useMemo(() => {
    return allDetections.filter(p => {
      // Date filter
      if (appliedRange) {
        if (appliedRange.start && p.detection_date < appliedRange.start) return false;
        if (appliedRange.end && p.detection_date > appliedRange.end) return false;
      }
      // FP, Conf, Polymer filter
      if (!showFP && p.is_false_positive) return false;
      if (p.confidence < minConf) return false;
      if (typeFilter !== "all" && p.polymer_type !== typeFilter) return false;
      return true;
    });
  }, [allDetections, appliedRange, showFP, minConf, typeFilter]);

  // ── Filter sources by date range ───────────────────────────────────────────
  const filteredSources = useMemo(() => {
    if (!appliedRange) return allSources;
    return allSources.filter(s => {
      if (appliedRange.start && s.detection_date < appliedRange.start) return false;
      if (appliedRange.end && s.detection_date > appliedRange.end) return false;
      return true;
    });
  }, [allSources, appliedRange]);

  // ── Hotspot clusters for the filtered period ───────────────────────────────
  const hotspots = useMemo(() => clusterPoints(filteredDetections, CLUSTER_RADIUS_KM), [filteredDetections]);

  // ── Summary KPIs ───────────────────────────────────────────────────────────
  const totalDetections = filteredDetections.length;
  const activeHotspotsCount = hotspots.length;
  const totalAffectedAreaM2 = useMemo(() => {
    return filteredDetections.reduce((sum, p) => sum + p.area_m2, 0);
  }, [filteredDetections]);
  const backtrackedSourcesCount = filteredSources.length;

  // ── 1. Group by Geographic Coastal Area ─────────────────────────────────────
  const topAreaContributors = useMemo<AreaContributor[]>(() => {
    if (hotspots.length === 0) return [];

    const map = new Map<string, {
      areaType: string;
      badgeColor: string;
      clusters: Hotspot[];
      detectionCount: number;
      totalAreaM2: number;
      allLats: number[];
      allLngs: number[];
      runs: Set<string>;
      weightedConfSum: number;
    }>();

    hotspots.forEach(h => {
      const existing = map.get(h.area_name);
      if (!existing) {
        map.set(h.area_name, {
          areaType: h.area_type,
          badgeColor: h.area_badge_color,
          clusters: [h],
          detectionCount: h.detection_count,
          totalAreaM2: h.total_area_m2,
          allLats: [h.center[0]],
          allLngs: [h.center[1]],
          runs: new Set(h.points.map(p => p.run_id)),
          weightedConfSum: h.avg_confidence * h.detection_count,
        });
      } else {
        existing.clusters.push(h);
        existing.detectionCount += h.detection_count;
        existing.totalAreaM2 += h.total_area_m2;
        existing.allLats.push(h.center[0]);
        existing.allLngs.push(h.center[1]);
        h.points.forEach(p => existing.runs.add(p.run_id));
        existing.weightedConfSum += h.avg_confidence * h.detection_count;
      }
    });

    const list: AreaContributor[] = [];
    map.forEach((val, areaName) => {
      const minLat = Math.min(...val.allLats);
      const maxLat = Math.max(...val.allLats);
      const minLng = Math.min(...val.allLngs);
      const maxLng = Math.max(...val.allLngs);
      const avgLat = val.allLats.reduce((a, b) => a + b, 0) / val.allLats.length;
      const avgLng = val.allLngs.reduce((a, b) => a + b, 0) / val.allLngs.length;

      list.push({
        id: areaName,
        areaName,
        areaType: val.areaType,
        badgeColor: val.badgeColor,
        detectionCount: val.detectionCount,
        percentOfTotal: (val.detectionCount / (totalDetections || 1)) * 100,
        clusterCount: val.clusters.length,
        clusters: val.clusters.sort((a, b) => b.detection_count - a.detection_count),
        center: [avgLat, avgLng],
        bounds: [[minLat - 0.05, minLng - 0.05], [maxLat + 0.05, maxLng + 0.05]],
        totalAreaM2: val.totalAreaM2,
        avgConfidence: val.weightedConfSum / (val.detectionCount || 1),
        runCount: val.runs.size,
      });
    });

    return list.sort((a, b) => b.detectionCount - a.detectionCount);
  }, [hotspots, totalDetections]);

  // Primary top area contributor
  const primaryAreaContributor = topAreaContributors.length > 0 ? topAreaContributors[0] : null;

  // ── 2. Source Attribution Trajectory Lines (Between Clusters and Sources) ──
  const attributionLines = useMemo<AttributionLine[]>(() => {
    if (hotspots.length === 0 || filteredSources.length === 0) return [];
    const lines: AttributionLine[] = [];

    hotspots.forEach(h => {
      let matchedSource: AttributedSourcePoint | null = null;

      // Try exact matching by run_id and cluster_id
      for (const pt of h.points) {
        const found = filteredSources.find(s => s.run_id === pt.run_id && s.cluster_id === pt.cluster_id);
        if (found) {
          matchedSource = found;
          break;
        }
      }

      // Fallback: match to nearest source by spatial distance
      if (!matchedSource && filteredSources.length > 0) {
        let minDist = Infinity;
        filteredSources.forEach(s => {
          try {
            const dist = turf.distance([h.center[1], h.center[0]], [s.lng, s.lat], { units: "kilometers" });
            if (dist < minDist) {
              minDist = dist;
              matchedSource = s;
            }
          } catch { /* ignore */ }
        });
      }

      if (
        matchedSource &&
        Number.isFinite(h.center[0]) && Number.isFinite(h.center[1]) &&
        Number.isFinite(matchedSource.lat) && Number.isFinite(matchedSource.lng)
      ) {
        const distKm = turf.distance([h.center[1], h.center[0]], [matchedSource.lng, matchedSource.lat], { units: "kilometers" });
        lines.push({
          id: `${h.id}-${matchedSource.id}`,
          clusterId: h.id,
          clusterLabel: h.label,
          clusterCenter: [h.center[0], h.center[1]],
          sourceId: matchedSource.id,
          sourceName: matchedSource.location_name,
          sourceType: matchedSource.source_type,
          sourceCenter: [matchedSource.lat, matchedSource.lng],
          score: matchedSource.attribution_score,
          days: matchedSource.days_to_source,
          distanceKm: distKm,
        });
      }
    });

    return lines;
  }, [hotspots, filteredSources]);

  // ── 3. Grouped by Source Attribution Contributors ─────────────────────────
  const topSourceContributors = useMemo<SourceContributor[]>(() => {
    if (filteredSources.length === 0) return [];

    const map = new Map<string, {
      sourceName: string;
      sourceType: string;
      country: string;
      badgeColor: string;
      center: [number, number];
      linkedClusters: Set<Hotspot>;
      scores: number[];
      transitDays: number[];
      explanations: string[];
      runs: Set<string>;
    }>();

    filteredSources.forEach(src => {
      const key = src.location_name || `${src.lat.toFixed(2)},${src.lng.toFixed(2)}`;
      const badgeColor =
        src.source_type === "fishing"
          ? "bg-purple-500/20 text-purple-300 border-purple-500/30"
          : src.source_type === "river" || src.source_type === "land-outflow"
          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
          : src.source_type === "shipping"
          ? "bg-blue-500/20 text-blue-300 border-blue-500/30"
          : "bg-amber-500/20 text-amber-300 border-amber-500/30";

      if (!map.has(key)) {
        map.set(key, {
          sourceName: src.location_name,
          sourceType: src.source_type,
          country: src.country,
          badgeColor,
          center: [src.lat, src.lng],
          linkedClusters: new Set<Hotspot>(),
          scores: [src.attribution_score],
          transitDays: [src.days_to_source],
          explanations: src.explanation ? [src.explanation] : [],
          runs: new Set([src.run_id]),
        });
      } else {
        const existing = map.get(key)!;
        existing.scores.push(src.attribution_score);
        existing.transitDays.push(src.days_to_source);
        if (src.explanation) existing.explanations.push(src.explanation);
        existing.runs.add(src.run_id);
      }
    });

    // Link clusters from attributionLines
    attributionLines.forEach(line => {
      map.forEach(entry => {
        if (entry.sourceName === line.sourceName) {
          const cluster = hotspots.find(h => h.id === line.clusterId);
          if (cluster) entry.linkedClusters.add(cluster);
        }
      });
    });

    const list: SourceContributor[] = [];
    map.forEach((val, key) => {
      const clustersArr = Array.from(val.linkedClusters);
      const detectionCount = clustersArr.reduce((sum, c) => sum + c.detection_count, 0);
      const avgScore = val.scores.reduce((a, b) => a + b, 0) / (val.scores.length || 1);
      const avgDays = val.transitDays.reduce((a, b) => a + b, 0) / (val.transitDays.length || 1);

      list.push({
        id: key,
        sourceName: val.sourceName,
        sourceType: val.sourceType,
        country: val.country,
        badgeColor: val.badgeColor,
        center: val.center,
        detectionCount,
        percentOfTotal: (detectionCount / (totalDetections || 1)) * 100,
        clusterCount: clustersArr.length,
        clusters: clustersArr.sort((a, b) => b.detection_count - a.detection_count),
        attributionScore: avgScore,
        daysToSource: avgDays,
        explanation: val.explanations[0] || "Reverse drift hydrodynamic trajectory.",
        runCount: val.runs.size,
      });
    });

    // Sort descending by detection count, fallback to attribution score
    return list.sort((a, b) => b.detectionCount - a.detectionCount || b.attributionScore - a.attributionScore);
  }, [filteredSources, hotspots, attributionLines, totalDetections]);

  // ── Activity Over Time Timeline ───────────────────────────────────────────
  const activityTimeline = useMemo(() => buildActivityTimeline(filteredDetections), [filteredDetections]);

  // Peak activity period
  const peakActivityPoint = useMemo(() => {
    if (activityTimeline.length === 0) return null;
    return [...activityTimeline].sort((a, b) => b.detections - a.detections)[0];
  }, [activityTimeline]);

  // ── Key Insights ───────────────────────────────────────────────────────────
  const keyInsights = useMemo(() => {
    if (hotspots.length === 0) return null;

    // Largest hotspot by area
    const largestHotspot = [...hotspots].sort((a, b) => b.total_area_m2 - a.total_area_m2)[0];
    
    // Highest confidence hotspot
    const highestConfHotspot = [...hotspots].sort((a, b) => b.avg_confidence - a.avg_confidence)[0];

    // Recurring hotspots (runs > 1)
    const recurringHotspotsCount = hotspots.filter(h => h.run_count > 1).length;

    // Dominant polymer type
    const polymerCount: Record<string, number> = {};
    filteredDetections.forEach(d => {
      polymerCount[d.polymer_type] = (polymerCount[d.polymer_type] || 0) + 1;
    });
    const topPolymer = Object.entries(polymerCount).sort((a, b) => b[1] - a[1])[0];

    return {
      topArea: primaryAreaContributor,
      topSource: topSourceContributors.length > 0 ? topSourceContributors[0] : null,
      peakActivity: peakActivityPoint,
      largest: largestHotspot,
      highestConf: highestConfHotspot,
      recurringCount: recurringHotspotsCount,
      topPolymer: topPolymer ? `${topPolymer[0]} (${topPolymer[1]} detections)` : "N/A",
    };
  }, [hotspots, filteredDetections, peakActivityPoint, primaryAreaContributor, topSourceContributors]);

  // ── Continuous Cloud Heatmap points ────────────────────────────────────────
  const heatmapPoints = useMemo<HeatmapPoint[]>(() => {
    if (filteredDetections.length === 0) return [];
    return filteredDetections.map(p => ({
      lat: p.lat,
      lng: p.lng,
      intensity: Math.max(0.2, Math.min(1.0, (p.confidence || 0.6) * 1.1)),
    }));
  }, [filteredDetections]);

  // ── Polymer types for filter ───────────────────────────────────────────────
  const polymerTypes = useMemo(() => {
    const types = new Set(allDetections.map(p => p.polymer_type));
    return Array.from(types).sort();
  }, [allDetections]);

  // ── Map center ─────────────────────────────────────────────────────────────
  const mapCenter = useMemo<[number, number]>(() => {
    if (filteredDetections.length > 0) {
      const lat = filteredDetections.reduce((s, p) => s + p.lat, 0) / filteredDetections.length;
      const lng = filteredDetections.reduce((s, p) => s + p.lng, 0) / filteredDetections.length;
      return [lat, lng];
    }
    return [16.1, -88.4];
  }, [filteredDetections]);

  const selectHotspot = useCallback((h: Hotspot) => {
    setSelectedHotspot(h);
    setSelectedSource(null);
    setFlyTo([h.center[0], h.center[1], 13]);
  }, []);

  const selectArea = useCallback((area: AreaContributor) => {
    setFlyTo([area.center[0], area.center[1], 11]);
    if (area.clusters.length > 0) {
      setSelectedHotspot(area.clusters[0]);
    }
  }, []);

  const selectSourceGroup = useCallback((srcGroup: SourceContributor) => {
    setFlyTo([srcGroup.center[0], srcGroup.center[1], 12]);
    const matching = allSources.find(s => s.location_name === srcGroup.sourceName);
    if (matching) {
      setSelectedSource(matching);
      setSelectedHotspot(null);
    } else if (srcGroup.clusters.length > 0) {
      setSelectedHotspot(srcGroup.clusters[0]);
    }
  }, [allSources]);

  const selectSource = useCallback((s: AttributedSourcePoint) => {
    setSelectedSource(s);
    setSelectedHotspot(null);
    setFlyTo([s.lat, s.lng, 13]);
  }, []);

  const tooltipStyle = {
    background: "hsl(220 30% 8%)",
    border: "1px solid hsl(215 20% 16%)",
    borderRadius: "8px",
    fontSize: "12px",
  };

  // ── Loading screen ─────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-foreground font-medium">Aggregating Historical & Coastal Area Plastic Data</p>
          <p className="text-muted-foreground text-sm mt-1">{loadingProgress}</p>
        </div>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="glass-card p-10 text-center max-w-sm">
          <AlertTriangle className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
          <h2 className="font-heading font-bold text-xl mb-2">No Completed Runs</h2>
          <p className="text-muted-foreground text-sm">
            Complete at least one pipeline run to view hotspot and coastal area analysis.
          </p>
        </div>
      </div>
    );
  }

  const currentTile = TILE_LAYERS[tileKey];
  const maxDetections = hotspots.length > 0 ? hotspots[0].detection_count : 1;

  return (
    <div className="min-h-screen bg-background pt-14 flex flex-col">
      {/* ── Page Header & Title ── */}
      <div className="max-w-[1600px] mx-auto w-full px-4 sm:px-6 pt-5 pb-3">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="font-heading text-2xl font-bold flex items-center gap-2.5">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Flame className="w-5 h-5 text-primary" />
              </div>
              Hotspot Locations & Source Attribution Analysis
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Geographic coastal zone grouping, reverse drift attribution trajectories, and period-specific debris concentrations across {runs.length} pipeline runs.
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 max-w-[1600px] mx-auto w-full px-4 sm:px-6 pb-12 flex flex-col gap-6">

        {/* ═══════════════════════════════════════════════════════════
            1. DATE RANGE ANALYSIS SELECTOR
            [ From Date ] → [ To Date ] [ Apply ]
        ═══════════════════════════════════════════════════════════ */}
        <div className="glass-card p-4 border border-border/40 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
              <CalendarRange className="w-4 h-4 text-primary" />
              <span>Date Range Analysis:</span>
            </div>

            {/* From Date input */}
            <div className="flex items-center gap-1.5 bg-black/30 border border-border/40 rounded-lg px-2.5 py-1.5">
              <span className="text-[11px] text-muted-foreground uppercase tracking-wide">From</span>
              <input
                type="date"
                value={fromDateInput}
                min={earliestDate}
                max={toDateInput || latestDate}
                onChange={e => setFromDateInput(e.target.value)}
                className="bg-transparent text-xs text-foreground font-mono focus:outline-none"
              />
            </div>

            <ArrowRight className="w-4 h-4 text-muted-foreground hidden sm:block" />

            {/* To Date input */}
            <div className="flex items-center gap-1.5 bg-black/30 border border-border/40 rounded-lg px-2.5 py-1.5">
              <span className="text-[11px] text-muted-foreground uppercase tracking-wide">To</span>
              <input
                type="date"
                value={toDateInput}
                min={fromDateInput || earliestDate}
                max={latestDate}
                onChange={e => setToDateInput(e.target.value)}
                className="bg-transparent text-xs text-foreground font-mono focus:outline-none"
              />
            </div>

            {/* Apply Button */}
            <button
              onClick={handleApplyDateRange}
              className="px-4 py-1.5 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary/90 transition-all flex items-center gap-1.5 shadow-sm"
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              Apply Range
            </button>

            {/* Reset / All Time Button */}
            <button
              onClick={handleResetDateRange}
              className="px-3 py-1.5 bg-muted/30 hover:bg-muted/50 text-xs text-muted-foreground hover:text-foreground rounded-lg transition-colors flex items-center gap-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              All Time
            </button>
          </div>

          {/* Applied range indicator pill */}
          <div className="text-xs text-muted-foreground flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>Analyzing:</span>
            <span className="font-semibold text-foreground">
              {appliedRange ? `${formatDate(appliedRange.start)} → ${formatDate(appliedRange.end)}` : "All Time"}
            </span>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════
            2. KEY SUMMARY (Compact 5 KPIs with highlighted Most Active Area)
        ═══════════════════════════════════════════════════════════ */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
          {/* Total plastic detections */}
          <div className="glass-card p-4 border border-border/30 flex flex-col justify-between">
            <div className="flex items-center justify-between text-muted-foreground mb-1">
              <span className="text-xs font-medium">Total Detections</span>
              <Crosshair className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <p className="text-2xl font-bold font-heading text-foreground">{totalDetections}</p>
              <p className="text-[11px] text-muted-foreground mt-0.5">verified plastic debris points</p>
            </div>
          </div>

          {/* Most active coastal area (CLEARLY HIGHLIGHTED) */}
          <div className="glass-card p-4 border-2 border-primary/60 bg-primary/10 shadow-lg shadow-primary/10 flex flex-col justify-between relative overflow-hidden">
            <div className="absolute top-0 right-0 px-2 py-0.5 bg-primary text-primary-foreground text-[9px] font-bold uppercase rounded-bl tracking-wider">
              Top Contributor
            </div>
            <div className="flex items-center justify-between text-primary mb-1">
              <span className="text-xs font-semibold flex items-center gap-1">
                <Flame className="w-3.5 h-3.5 text-primary" /> Top Contributing Area
              </span>
            </div>
            <div>
              <p className="text-sm font-bold font-heading text-foreground truncate" title={primaryAreaContributor?.areaName || "None"}>
                {primaryAreaContributor ? primaryAreaContributor.areaName : "None"}
              </p>
              <p className="text-xs text-primary font-medium mt-0.5">
                {primaryAreaContributor
                  ? `${primaryAreaContributor.detectionCount} detections (${primaryAreaContributor.percentOfTotal.toFixed(0)}% across ${primaryAreaContributor.clusterCount} clusters)`
                  : "No detections in range"}
              </p>
            </div>
          </div>

          {/* Number of active hotspots */}
          <div className="glass-card p-4 border border-border/30 flex flex-col justify-between">
            <div className="flex items-center justify-between text-muted-foreground mb-1">
              <span className="text-xs font-medium">Active Hotspots</span>
              <Target className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold font-heading text-foreground">{activeHotspotsCount}</p>
              <p className="text-[11px] text-muted-foreground mt-0.5">in {topAreaContributors.length} coastal zones</p>
            </div>
          </div>

          {/* Total affected area */}
          <div className="glass-card p-4 border border-border/30 flex flex-col justify-between">
            <div className="flex items-center justify-between text-muted-foreground mb-1">
              <span className="text-xs font-medium">Total Affected Area</span>
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <p className="text-2xl font-bold font-heading text-foreground">
                {totalAffectedAreaM2 > 10000 ? `${(totalAffectedAreaM2 / 1000).toFixed(1)}k` : Math.round(totalAffectedAreaM2)} <span className="text-sm font-normal text-muted-foreground">m²</span>
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">estimated marine surface area</p>
            </div>
          </div>

          {/* Backtracked source locations */}
          <div className="glass-card p-4 border border-border/30 flex flex-col justify-between">
            <div className="flex items-center justify-between text-muted-foreground mb-1">
              <span className="text-xs font-medium">Backtracked Sources</span>
              <Anchor className="w-4 h-4 text-purple-400" />
            </div>
            <div>
              <p className="text-2xl font-bold font-heading text-foreground">{backtrackedSourcesCount}</p>
              <p className="text-[11px] text-muted-foreground mt-0.5">{attributionLines.length} active trajectory lines</p>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════
            3. HOTSPOT MAP (Interactive Leaflet Map with Cloud Heatmap & Attribution Lines)
        ═══════════════════════════════════════════════════════════ */}
        <div className="glass-card overflow-hidden relative rounded-xl border border-border/40" style={{ height: "620px" }}>
          <MapContainer
            center={mapCenter}
            zoom={9}
            className="w-full h-full"
            style={{ background: tileKey === "light" ? "#f1f5f9" : "#060d1a" }}
            zoomControl={false}
          >
            <TileLayer url={currentTile.url} attribution={currentTile.attribution} />
            <MapController flyTo={flyTo} />

            {/* Continuous Cloud Heatmap (Gaussian Density Field) */}
            {showCloudHeatmap && (
              <CloudHeatmapLayer
                points={heatmapPoints}
                radius={cloudRadius}
                blur={24}
                opacity={cloudOpacity}
              />
            )}

            {/* ── Source Attribution Trajectory Lines (Between Clusters and Backtracked Sources) ── */}
            {showAttributionLines && attributionLines.map((line) => {
              const isSelected = selectedHotspot?.id === line.clusterId || selectedSource?.id === line.sourceId;
              return (
                <Polyline
                  key={`attr-line-${line.id}`}
                  positions={[line.clusterCenter, line.sourceCenter]}
                  pathOptions={{
                    color: isSelected ? "#f43f5e" : "#c084fc",
                    weight: isSelected ? 3.5 : 2,
                    dashArray: isSelected ? "6, 5" : "4, 6",
                    opacity: isSelected ? 1.0 : 0.75,
                  }}
                >
                  <Popup>
                    <div className="text-xs space-y-1.5 p-1 min-w-[210px]" style={{ color: "#0f172a" }}>
                      <div className="flex items-center justify-between gap-2 border-b border-slate-200 pb-1">
                        <span className="font-bold text-[12px] text-purple-700 flex items-center gap-1">
                          ↔ Source Attribution Trajectory
                        </span>
                        <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-purple-100 text-purple-800">
                          {line.sourceType}
                        </span>
                      </div>
                      <div className="font-bold text-slate-900 text-sm leading-tight">
                        {line.clusterLabel} ➔ {line.sourceName}
                      </div>
                      <div className="grid grid-cols-2 gap-2 bg-slate-50 p-2 rounded border border-slate-200 text-[11px] my-1">
                        <div>
                          <span className="text-slate-500 block text-[10px]">Attribution Score</span>
                          <span className="font-bold text-emerald-700 text-xs">{(line.score * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">Reverse Transit</span>
                          <span className="font-bold text-slate-800 text-xs">{line.days.toFixed(1)} days</span>
                        </div>
                      </div>
                      <div className="text-[10px] text-slate-500 pt-0.5 flex justify-between">
                        <span>Reverse Drift: {line.distanceKm.toFixed(1)} km</span>
                        <span className="text-purple-700 font-semibold">{line.sourceType}</span>
                      </div>
                    </div>
                  </Popup>
                </Polyline>
              );
            })}

            {/* Source Attributed Place Markers (Backtracked Origins) */}
            {showSources && filteredSources.map((src) => {
              const isSelected = selectedSource?.id === src.id;
              return (
                <CircleMarker
                  key={`src-${src.id}`}
                  center={[src.lat, src.lng]}
                  radius={isSelected ? 10 : 7}
                  pathOptions={{
                    fillColor: "#c084fc",
                    fillOpacity: 0.95,
                    color: "#ffffff",
                    weight: isSelected ? 3 : 1.5,
                  }}
                  eventHandlers={{
                    click: () => selectSource(src),
                  }}
                >
                  <Popup>
                    <div className="text-xs space-y-1.5 p-1 min-w-[220px]" style={{ color: "#0f172a" }}>
                      <div className="flex items-center justify-between gap-2 border-b border-slate-200 pb-1">
                        <span className="font-bold text-[12px] text-purple-700 flex items-center gap-1">
                          ⚓ Source Attributed Place
                        </span>
                        <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-purple-100 text-purple-800">
                          {src.source_type}
                        </span>
                      </div>
                      <div className="font-bold text-slate-900 text-sm leading-tight">{src.location_name}</div>
                      <div className="text-slate-600 font-medium text-[11px]">📍 {src.country}</div>
                      
                      <div className="grid grid-cols-2 gap-2 bg-slate-50 p-2 rounded border border-slate-200 text-[11px] my-1">
                        <div>
                          <span className="text-slate-500 block text-[10px]">Attribution Score</span>
                          <span className="font-bold text-emerald-700 text-xs">{(src.attribution_score * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">Backtrack Transit</span>
                          <span className="font-bold text-slate-800 text-xs">{src.days_to_source.toFixed(1)} days</span>
                        </div>
                      </div>

                      {src.explanation && (
                        <div className="text-[11px] text-slate-600 italic bg-purple-50/60 p-1.5 rounded border border-purple-100 leading-snug">
                          "{src.explanation}"
                        </div>
                      )}

                      <div className="text-[10px] text-slate-400 pt-0.5 flex justify-between">
                        <span>Cluster #{src.cluster_id}</span>
                        <span>{src.run_name}</span>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}

            {/* Hotspot Centroid Rings with Coastal Area Popups */}
            {hotspots.map((h) => {
              const norm = h.detection_count / maxDetections;
              const isSelected = selectedHotspot?.id === h.id;
              return (
                <CircleMarker
                  key={`hs-${h.id}`}
                  center={h.center}
                  radius={isSelected ? 16 : Math.max(9, 8 + norm * 12)}
                  pathOptions={{
                    fillColor: intensityColor(norm),
                    fillOpacity: isSelected ? 0.95 : 0.8,
                    color: "#ffffff",
                    weight: isSelected ? 2.5 : 1.5,
                  }}
                  eventHandlers={{ click: () => selectHotspot(h) }}
                >
                  <Popup>
                    <div className="text-xs space-y-1.5 min-w-[200px]" style={{ color: "#0f172a" }}>
                      <div className="flex items-center justify-between gap-1 border-b border-slate-200 pb-1">
                        <span className="font-bold text-sm text-slate-900">{h.label}</span>
                        <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-cyan-100 text-cyan-800">
                          {h.area_type}
                        </span>
                      </div>
                      <div className="font-semibold text-xs text-primary">{h.area_name}</div>
                      <div className="text-slate-700 font-medium">{h.detection_count} detections across {h.run_count} run(s)</div>
                      <div className="text-slate-600">Avg confidence: {(h.avg_confidence * 100).toFixed(1)}%</div>
                      <div className="text-slate-600">Area: {h.total_area_m2 > 1000 ? `${(h.total_area_m2 / 1000).toFixed(1)}k` : Math.round(h.total_area_m2)} m²</div>
                      <div className="text-slate-500 text-[10px] pt-1 border-t border-slate-200">
                        {formatDate(h.first_seen)} → {formatDate(h.last_seen)}
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>

          {/* ── Map overlay: Controls & Filters ── */}
          <div className="absolute top-3 left-3 z-[1000] flex flex-col gap-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-1.5 px-3 py-2 glass rounded-lg text-xs font-medium hover:bg-white/10 transition-colors shadow-lg"
            >
              <SlidersHorizontal className="w-3.5 h-3.5 text-primary" />
              Map Controls & Base Layers
            </button>
            {showFilters && (
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass p-4 rounded-xl space-y-3.5 w-[260px] shadow-2xl backdrop-blur-md border border-border/40"
              >
                {/* Base Map Switcher: SATELLITE (DEFAULT), DARK, LIGHT */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <p className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground">Base Map Style</p>
                    <span className="text-[10px] font-mono text-primary uppercase">{tileKey}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-1 p-1 bg-black/30 rounded-lg border border-border/30">
                    {(["satellite", "dark", "light"] as const).map(mode => (
                      <button
                        key={mode}
                        onClick={() => setTileKey(mode)}
                        className={`px-2 py-1.5 text-xs rounded-md capitalize font-medium transition-all ${
                          tileKey === mode
                            ? "bg-primary text-primary-foreground shadow-sm font-semibold"
                            : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Cloud Heatmap Controls */}
                <div className="pt-2 border-t border-border/30 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-foreground flex items-center gap-1.5">
                      <Flame className="w-3.5 h-3.5 text-primary" /> Cloud Heatmap
                    </span>
                    <button
                      onClick={() => setShowCloudHeatmap(!showCloudHeatmap)}
                      className={`text-xs px-2 py-0.5 rounded transition-colors ${
                        showCloudHeatmap ? "bg-primary/20 text-primary" : "bg-muted/40 text-muted-foreground"
                      }`}
                    >
                      {showCloudHeatmap ? "On" : "Off"}
                    </button>
                  </div>

                  {showCloudHeatmap && (
                    <>
                      <div>
                        <div className="flex justify-between text-[11px] text-muted-foreground mb-1">
                          <span>Cloud Opacity</span>
                          <span className="text-foreground font-mono">{Math.round(cloudOpacity * 100)}%</span>
                        </div>
                        <input
                          type="range" min={0.3} max={1.0} step={0.05}
                          value={cloudOpacity}
                          onChange={e => setCloudOpacity(parseFloat(e.target.value))}
                          className="w-full accent-primary h-1.5"
                        />
                      </div>
                      <div>
                        <div className="flex justify-between text-[11px] text-muted-foreground mb-1">
                          <span>Dispersion Radius</span>
                          <span className="text-foreground font-mono">{cloudRadius}px</span>
                        </div>
                        <input
                          type="range" min={20} max={60} step={2}
                          value={cloudRadius}
                          onChange={e => setCloudRadius(parseInt(e.target.value))}
                          className="w-full accent-primary h-1.5"
                        />
                      </div>
                    </>
                  )}
                </div>

                {/* Source Attribution & Lines Toggle */}
                <div className="pt-2 border-t border-border/30 space-y-2">
                  <label className="flex items-center justify-between cursor-pointer">
                    <span className="text-xs text-foreground flex items-center gap-1.5">
                      <Anchor className="w-3.5 h-3.5 text-purple-400" /> Attributed Sources
                    </span>
                    <button
                      onClick={() => setShowSources(!showSources)}
                      className={`w-9 h-5 rounded-full transition-colors relative ${showSources ? "bg-purple-600" : "bg-muted/50"}`}
                    >
                      <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-all ${showSources ? "left-4" : "left-0.5"}`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <span className="text-xs text-foreground flex items-center gap-1.5">
                      <GitCommit className="w-3.5 h-3.5 text-purple-400" /> Attribution Lines
                    </span>
                    <button
                      onClick={() => setShowAttributionLines(!showAttributionLines)}
                      className={`w-9 h-5 rounded-full transition-colors relative ${showAttributionLines ? "bg-purple-600" : "bg-muted/50"}`}
                    >
                      <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-all ${showAttributionLines ? "left-4" : "left-0.5"}`} />
                    </button>
                  </label>
                </div>

                {/* Minimum Confidence & False Positives */}
                <div className="pt-2 border-t border-border/30 space-y-2">
                  <label className="flex items-center justify-between cursor-pointer">
                    <span className="text-xs text-muted-foreground">Show False Positives</span>
                    <button
                      onClick={() => setShowFP(!showFP)}
                      className={`w-9 h-5 rounded-full transition-colors relative ${showFP ? "bg-primary" : "bg-muted/50"}`}
                    >
                      <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-all ${showFP ? "left-4" : "left-0.5"}`} />
                    </button>
                  </label>

                  <div>
                    <label className="text-xs text-muted-foreground block mb-1">
                      Min Confidence: <span className="text-foreground font-medium">{(minConf * 100).toFixed(0)}%</span>
                    </label>
                    <input
                      type="range" min={0} max={1} step={0.05}
                      value={minConf}
                      onChange={e => setMinConf(parseFloat(e.target.value))}
                      className="w-full accent-primary h-1.5"
                    />
                  </div>

                  <div>
                    <label className="text-xs text-muted-foreground block mb-1">Polymer Type</label>
                    <select
                      value={typeFilter}
                      onChange={e => setTypeFilter(e.target.value)}
                      className="w-full px-2.5 py-1.5 bg-muted/50 border border-border/50 rounded-lg text-xs text-foreground"
                    >
                      <option value="all">All Types</option>
                      {polymerTypes.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </div>
                </div>
              </motion.div>
            )}
          </div>

          {/* ── Map Legend (With Interactive Attribution Lines Toggle) ── */}
          <div className="absolute bottom-3 left-3 z-[1000] glass px-3.5 py-2.5 rounded-lg shadow-lg min-w-[220px] max-w-[260px]">
            <p className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground mb-1.5">Visualization Legend</p>
            
            {/* Cloud Heatmap Gradient */}
            <div className="mb-2">
              <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                <span>Cloud Density</span>
                <span className="text-primary font-medium">Plume</span>
              </div>
              <div className="h-2 w-full rounded-full" style={{ background: "linear-gradient(to right, #0284c7, #06b6d4, #10b981, #facc15, #f97316, #ef4444)" }} />
            </div>

            {/* Legend Markers */}
            <div className="space-y-1.5 pt-1.5 border-t border-border/20 text-[11px]">
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-3 h-3 rounded-full bg-purple-400 border border-white shadow-sm flex-shrink-0" />
                <span className="text-foreground font-medium">Source Attributed Place</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-3 h-3 rounded-full border-2 border-white bg-orange-500 flex-shrink-0" />
                <span>Hotspot Centroid</span>
              </div>

              {/* ── Interactive Attribution Lines Toggle in Legend ── */}
              <div
                className="flex items-center justify-between gap-2 p-1.5 mt-1 rounded-md hover:bg-white/10 cursor-pointer transition-colors border border-purple-500/20 bg-purple-950/20"
                onClick={() => setShowAttributionLines(!showAttributionLines)}
                title="Click to toggle source attribution trajectory lines on/off"
              >
                <div className="flex items-center gap-2">
                  <div className="w-5 h-0.5 border-b-2 border-dashed border-purple-400 flex-shrink-0" />
                  <span className="text-foreground font-semibold text-xs">Attribution Lines</span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowAttributionLines(!showAttributionLines);
                  }}
                  className={`w-7 h-4 rounded-full transition-colors relative flex-shrink-0 ${
                    showAttributionLines ? "bg-purple-600" : "bg-muted/60"
                  }`}
                >
                  <div
                    className={`absolute top-0.5 w-3 h-3 bg-white rounded-full shadow transition-all ${
                      showAttributionLines ? "left-3.5" : "left-0.5"
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════
            4. TOP ACTIVE LOCATIONS: COASTAL AREAS vs SOURCE ATTRIBUTION
            Options: Grouped Coastal Area OR Grouped by Source Attribution
        ═══════════════════════════════════════════════════════════ */}
        <div className="glass-card p-5 border border-border/40">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
            <div>
              <h2 className="text-base font-semibold font-heading flex items-center gap-2 text-foreground">
                <Award className="w-4 h-4 text-primary" />
                {activeLocationsView === "coastal_areas" ? "Top Contributing Coastal Areas" : "Top Source Attribution Origins"}
              </h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                {activeLocationsView === "coastal_areas"
                  ? "Coordinates resolved into coastal marine zones (Marina, Estuary, Bay, Reef) and grouped by total activity."
                  : "Reverse drift backtracking origins grouped by attributed source location and maritime activity."}
              </p>
            </div>

            {/* View Mode Toggle: Grouped Coastal Areas vs Grouped by Source Attribution */}
            <div className="flex items-center gap-2">
              <div className="flex p-0.5 bg-black/40 border border-border/40 rounded-lg text-xs">
                <button
                  onClick={() => setActiveLocationsView("coastal_areas")}
                  className={`px-3 py-1.5 rounded-md font-medium transition-all flex items-center gap-1.5 ${
                    activeLocationsView === "coastal_areas"
                      ? "bg-primary text-primary-foreground font-semibold shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                  }`}
                >
                  <Waves className="w-3.5 h-3.5" />
                  Grouped Coastal Areas ({topAreaContributors.length})
                </button>
                <button
                  onClick={() => setActiveLocationsView("source_attribution")}
                  className={`px-3 py-1.5 rounded-md font-medium transition-all flex items-center gap-1.5 ${
                    activeLocationsView === "source_attribution"
                      ? "bg-purple-600 text-white font-semibold shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                  }`}
                >
                  <Anchor className="w-3.5 h-3.5" />
                  Grouped by Source Attribution ({topSourceContributors.length})
                </button>
              </div>
            </div>
          </div>

          {/* VIEW 1: GROUPED BY COASTAL AREA */}
          {activeLocationsView === "coastal_areas" && (
            topAreaContributors.length === 0 ? (
              <div className="p-6 text-center text-sm text-muted-foreground border border-dashed border-border/40 rounded-lg">
                No coastal area activity recorded within the selected date range.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
                {topAreaContributors.slice(0, 6).map((area, index) => {
                  const isLeading = index === 0;

                  return (
                    <motion.div
                      key={area.id}
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      onClick={() => selectArea(area)}
                      className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between ${
                        isLeading
                          ? "border-primary/60 bg-gradient-to-br from-primary/15 via-primary/5 to-transparent shadow-lg shadow-primary/5"
                          : "border-border/30 bg-muted/10 hover:bg-muted/20 hover:border-border/60"
                      }`}
                    >
                      <div>
                        {/* Header: Rank + Area Name + Type Badge */}
                        <div className="flex items-start justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-bold ${
                              isLeading ? "bg-primary text-primary-foreground" : "bg-muted/50 text-muted-foreground font-mono"
                            }`}>
                              {index + 1}
                            </span>
                            <span className="font-semibold text-sm text-foreground leading-tight">{area.areaName}</span>
                          </div>
                          <span className={`px-2 py-0.5 text-[10px] font-semibold uppercase rounded-full border flex-shrink-0 ${area.badgeColor}`}>
                            {area.areaType}
                          </span>
                        </div>

                        {/* Volume & Share */}
                        <div className="flex items-center justify-between text-xs mt-3 mb-1">
                          <span className="font-bold text-base text-foreground">
                            {area.detectionCount} <span className="text-xs font-normal text-muted-foreground">detections</span>
                          </span>
                          <span className="text-primary font-semibold">{area.percentOfTotal.toFixed(1)}% of period</span>
                        </div>

                        {/* Progress Bar */}
                        <div className="w-full h-1.5 bg-black/40 rounded-full overflow-hidden mb-3">
                          <div
                            className="h-full bg-gradient-to-r from-cyan-500 to-primary rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, area.percentOfTotal)}%` }}
                          />
                        </div>

                        {/* Metrics Grid */}
                        <div className="grid grid-cols-2 gap-2 text-[11px] text-muted-foreground bg-black/20 p-2.5 rounded-lg border border-border/20">
                          <div>
                            <span className="text-muted-foreground block text-[10px] uppercase">Clusters Aggregated</span>
                            <span className="font-semibold text-foreground">{area.clusterCount} cluster{area.clusterCount !== 1 ? "s" : ""}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-[10px] uppercase">Total Surface Area</span>
                            <span className="font-semibold text-foreground">
                              {area.totalAreaM2 > 1000 ? `${(area.totalAreaM2 / 1000).toFixed(1)}k m²` : `${Math.round(area.totalAreaM2)} m²`}
                            </span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-[10px] uppercase">Avg Confidence</span>
                            <span className="font-semibold text-emerald-400">{(area.avgConfidence * 100).toFixed(1)}%</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-[10px] uppercase">Contributing Runs</span>
                            <span className="font-semibold text-foreground">{area.runCount} run{area.runCount !== 1 ? "s" : ""}</span>
                          </div>
                        </div>

                        {/* Included Clusters Badges */}
                        <div className="mt-2.5 flex items-center gap-1.5 flex-wrap">
                          <span className="text-[10px] text-muted-foreground">Includes:</span>
                          {area.clusters.slice(0, 4).map(c => (
                            <span key={c.id} className="text-[10px] bg-muted/40 text-muted-foreground px-1.5 py-0.2 rounded font-mono">
                              {c.label} ({c.detection_count})
                            </span>
                          ))}
                          {area.clusters.length > 4 && (
                            <span className="text-[10px] text-muted-foreground font-mono">+{area.clusters.length - 4} more</span>
                          )}
                        </div>
                      </div>

                      <div className="mt-3 pt-2.5 border-t border-border/20 flex items-center justify-between text-[11px]">
                        <span className="font-mono text-[10px] text-muted-foreground">
                          {area.center[0].toFixed(2)}°, {area.center[1].toFixed(2)}°
                        </span>
                        <span className="text-primary font-medium hover:underline flex items-center gap-0.5">
                          Focus Area on Map →
                        </span>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )
          )}

          {/* VIEW 2: GROUPED BY SOURCE ATTRIBUTION */}
          {activeLocationsView === "source_attribution" && (
            topSourceContributors.length === 0 ? (
              <div className="p-6 text-center text-sm text-muted-foreground border border-dashed border-border/40 rounded-lg">
                No backtracked source origins attributed in the selected date range.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
                {topSourceContributors.slice(0, 6).map((srcGroup, index) => {
                  const isTopSource = index === 0;

                  return (
                    <motion.div
                      key={srcGroup.id}
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      onClick={() => selectSourceGroup(srcGroup)}
                      className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between ${
                        isTopSource
                          ? "border-purple-500/60 bg-gradient-to-br from-purple-950/30 via-purple-900/10 to-transparent shadow-lg shadow-purple-900/10"
                          : "border-border/30 bg-muted/10 hover:bg-muted/20 hover:border-purple-500/40"
                      }`}
                    >
                      <div>
                        {/* Header: Rank + Source Name + Type Badge */}
                        <div className="flex items-start justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-bold ${
                              isTopSource ? "bg-purple-600 text-white" : "bg-muted/50 text-muted-foreground font-mono"
                            }`}>
                              {index + 1}
                            </span>
                            <span className="font-semibold text-sm text-foreground leading-tight">{srcGroup.sourceName}</span>
                          </div>
                          <span className={`px-2 py-0.5 text-[10px] font-semibold uppercase rounded-full border flex-shrink-0 ${srcGroup.badgeColor}`}>
                            {srcGroup.sourceType}
                          </span>
                        </div>

                        {/* Country Tag */}
                        <div className="text-[11px] text-muted-foreground mb-2 flex items-center gap-1">
                          <span>📍</span> {srcGroup.country}
                        </div>

                        {/* Volume & Share */}
                        <div className="flex items-center justify-between text-xs mt-2 mb-1">
                          <span className="font-bold text-base text-foreground">
                            {srcGroup.detectionCount} <span className="text-xs font-normal text-muted-foreground">attributed detections</span>
                          </span>
                          <span className="text-purple-400 font-semibold">{srcGroup.percentOfTotal.toFixed(1)}% of period</span>
                        </div>

                        {/* Progress Bar */}
                        <div className="w-full h-1.5 bg-black/40 rounded-full overflow-hidden mb-3">
                          <div
                            className="h-full bg-gradient-to-r from-purple-500 to-violet-400 rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, srcGroup.percentOfTotal)}%` }}
                          />
                        </div>

                        {/* Metrics Grid */}
                        <div className="grid grid-cols-2 gap-2 text-[11px] text-muted-foreground bg-black/20 p-2.5 rounded-lg border border-border/20">
                          <div>
                            <span className="text-muted-foreground block text-[10px] uppercase">Attribution Score</span>
                            <span className="font-semibold text-emerald-400">{(srcGroup.attributionScore * 100).toFixed(1)}%</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-[10px] uppercase">Reverse Transit</span>
                            <span className="font-semibold text-foreground">{srcGroup.daysToSource.toFixed(1)} days</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-[10px] uppercase">Clusters Attributed</span>
                            <span className="font-semibold text-purple-300">{srcGroup.clusterCount} cluster{srcGroup.clusterCount !== 1 ? "s" : ""}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-[10px] uppercase">Pipeline Runs</span>
                            <span className="font-semibold text-foreground">{srcGroup.runCount} run{srcGroup.runCount !== 1 ? "s" : ""}</span>
                          </div>
                        </div>

                        {/* Linked Clusters */}
                        <div className="mt-2.5 flex items-center gap-1.5 flex-wrap">
                          <span className="text-[10px] text-muted-foreground">Connected:</span>
                          {srcGroup.clusters.slice(0, 4).map(c => (
                            <span key={c.id} className="text-[10px] bg-purple-950/40 border border-purple-800/30 text-purple-300 px-1.5 py-0.2 rounded font-mono">
                              {c.label} ({c.detection_count})
                            </span>
                          ))}
                          {srcGroup.clusters.length > 4 && (
                            <span className="text-[10px] text-muted-foreground font-mono">+{srcGroup.clusters.length - 4} more</span>
                          )}
                        </div>
                      </div>

                      <div className="mt-3 pt-2.5 border-t border-border/20 flex items-center justify-between text-[11px]">
                        <span className="font-mono text-[10px] text-muted-foreground">
                          {srcGroup.center[0].toFixed(2)}°, {srcGroup.center[1].toFixed(2)}°
                        </span>
                        <span className="text-purple-400 font-medium hover:underline flex items-center gap-0.5">
                          Focus Source & Lines →
                        </span>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )
          )}
        </div>

        {/* ═══════════════════════════════════════════════════════════
            5. ACTIVITY OVER TIME
            ONE simple line chart showing plastic activity over the selected date range
            Answering: "When was plastic activity highest?"
        ═══════════════════════════════════════════════════════════ */}
        <div className="glass-card p-5 border border-border/40">
          <div className="flex items-start justify-between gap-4 flex-wrap mb-4">
            <div>
              <h2 className="text-base font-semibold font-heading flex items-center gap-2 text-foreground">
                <TrendingUp className="w-4 h-4 text-primary" />
                Plastic Activity Over Time
              </h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                Detection volume chronology for the selected period answering when plastic activity peaked.
              </p>
            </div>

            {/* Answer banner to: "When was plastic activity highest?" */}
            {peakActivityPoint && (
              <div className="px-3 py-1.5 bg-primary/10 border border-primary/30 rounded-lg text-xs flex items-center gap-2">
                <Zap className="w-3.5 h-3.5 text-primary flex-shrink-0" />
                <span className="text-muted-foreground">Activity Peak:</span>
                <span className="font-bold text-foreground">{formatDate(peakActivityPoint.date)}</span>
                <span className="text-primary font-semibold">({peakActivityPoint.detections} detections)</span>
              </div>
            )}
          </div>

          <div className="h-[200px] w-full">
            {activityTimeline.length === 0 ? (
              <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                No activity records available for this date range.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={activityTimeline} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(215 20% 15%)" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatDate}
                    tick={{ fill: "#6B7280", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#6B7280", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(val: any) => [`${val} detections`, "Plastic Detections"]}
                    labelFormatter={(label) => formatDate(String(label))}
                  />
                  <Line
                    type="monotone"
                    dataKey="detections"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: "hsl(var(--primary))", strokeWidth: 1.5, stroke: "#ffffff" }}
                    activeDot={{ r: 6, fill: "hsl(var(--primary))" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════
            6. HOTSPOT DETAILS & DATA-DRIVEN KEY INSIGHTS
        ═══════════════════════════════════════════════════════════ */}
        <div className="space-y-4">
          {/* Selected Hotspot Detailed View (when clicked) */}
          <AnimatePresence>
            {selectedHotspot && (
              <motion.div
                key={selectedHotspot.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="glass-card border border-primary/40 p-5 shadow-xl bg-primary/5"
              >
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-heading font-semibold text-base flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-primary" />
                      {selectedHotspot.label} — Hotspot Specific Metrics
                    </h3>
                    <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-2">
                      <span>Area:</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${selectedHotspot.area_badge_color}`}>
                        {selectedHotspot.area_name} ({selectedHotspot.area_type})
                      </span>
                    </div>
                  </div>
                  <button onClick={() => setSelectedHotspot(null)} className="text-muted-foreground hover:text-foreground">
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                  {[
                    { label: "Detections", value: selectedHotspot.detection_count },
                    { label: "Contributing Runs", value: selectedHotspot.run_count },
                    { label: "Avg Confidence", value: `${(selectedHotspot.avg_confidence * 100).toFixed(1)}%` },
                    { label: "Total Area", value: `${selectedHotspot.total_area_m2 > 1000 ? (selectedHotspot.total_area_m2 / 1000).toFixed(1) + "k" : Math.round(selectedHotspot.total_area_m2)} m²` },
                    { label: "Center Latitude", value: selectedHotspot.center[0].toFixed(4) },
                    { label: "Center Longitude", value: selectedHotspot.center[1].toFixed(4) },
                  ].map(({ label, value }) => (
                    <div key={label} className="bg-muted/10 border border-border/20 rounded-lg p-2.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">{label}</p>
                      <p className="font-bold text-base font-heading">{value}</p>
                    </div>
                  ))}
                </div>
                <div className="mt-3 text-xs text-muted-foreground flex items-center gap-4 flex-wrap">
                  <div>
                    <Calendar className="w-3.5 h-3.5 inline mr-1 text-primary" />
                    First detected: <span className="text-foreground">{formatDate(selectedHotspot.first_seen)}</span>
                  </div>
                  <div>
                    <Calendar className="w-3.5 h-3.5 inline mr-1 text-primary" />
                    Last detected: <span className="text-foreground">{formatDate(selectedHotspot.last_seen)}</span>
                  </div>
                </div>
              </motion.div>
            )}

            {selectedSource && (
              <motion.div
                key={selectedSource.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="glass-card border border-purple-500/40 p-5 shadow-xl bg-purple-950/10"
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-heading font-semibold text-base flex items-center gap-2 text-foreground">
                    <Anchor className="w-4 h-4 text-purple-400" />
                    Source Attributed Place: <span className="text-purple-300 font-bold">{selectedSource.location_name}</span>
                  </h3>
                  <button onClick={() => setSelectedSource(null)} className="text-muted-foreground hover:text-foreground">
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-3">
                  {[
                    { label: "Country", value: selectedSource.country },
                    { label: "Source Type", value: selectedSource.source_type },
                    { label: "Attribution Score", value: `${(selectedSource.attribution_score * 100).toFixed(1)}%` },
                    { label: "Reverse Drift", value: `${selectedSource.days_to_source.toFixed(1)} days` },
                    { label: "Latitude", value: selectedSource.lat.toFixed(4) },
                    { label: "Longitude", value: selectedSource.lng.toFixed(4) },
                  ].map(({ label, value }) => (
                    <div key={label} className="bg-muted/20 border border-border/20 rounded-lg p-2.5">
                      <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">{label}</p>
                      <p className="font-bold text-sm font-heading text-foreground capitalize">{value}</p>
                    </div>
                  ))}
                </div>
                {selectedSource.explanation && (
                  <div className="p-3 bg-black/20 rounded-lg border border-purple-500/20 text-xs text-muted-foreground">
                    <span className="font-semibold text-purple-300">Hydrodynamic Backtrack Explanation: </span>
                    {selectedSource.explanation}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Key Insights Section */}
          {keyInsights && (
            <div className="glass-card p-5 border border-border/40">
              <h2 className="text-base font-semibold font-heading flex items-center gap-2 mb-3 text-foreground">
                <Zap className="w-4 h-4 text-yellow-400" />
                Data-Driven Key Insights
                <span className="text-xs font-normal text-muted-foreground">· Derived from actual satellite detections in selected period</span>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {/* Most Active Area */}
                <div className="p-3 bg-muted/10 border border-border/20 rounded-xl">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground block mb-1">
                    Primary Coastal Contributor
                  </span>
                  <p className="text-sm font-bold text-foreground">
                    {keyInsights.topArea ? keyInsights.topArea.areaName : "N/A"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {keyInsights.topArea ? `Leading coastal accumulation zone with ${keyInsights.topArea.detectionCount} detections (${keyInsights.topArea.percentOfTotal.toFixed(0)}% of period across ${keyInsights.topArea.clusterCount} clusters).` : "No activity"}
                  </p>
                </div>

                {/* Top Source Origin */}
                <div className="p-3 bg-muted/10 border border-border/20 rounded-xl">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground block mb-1">
                    Dominant Attributed Source
                  </span>
                  <p className="text-sm font-bold text-foreground">
                    {keyInsights.topSource ? keyInsights.topSource.sourceName : "N/A"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {keyInsights.topSource ? `${keyInsights.topSource.sourceType.toUpperCase()} origin linked to ${keyInsights.topSource.clusterCount} clusters (~${keyInsights.topSource.daysToSource.toFixed(1)}d reverse transit).` : "No source"}
                  </p>
                </div>

                {/* Peak Activity */}
                <div className="p-3 bg-muted/10 border border-border/20 rounded-xl">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground block mb-1">
                    Peak Detection Day
                  </span>
                  <p className="text-sm font-bold text-foreground">
                    {keyInsights.peakActivity ? formatDate(keyInsights.peakActivity.date) : "N/A"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {keyInsights.peakActivity ? `Highest single-day debris accumulation with ${keyInsights.peakActivity.detections} detections registered.` : "No activity"}
                  </p>
                </div>

                {/* Largest Hotspot */}
                <div className="p-3 bg-muted/10 border border-border/20 rounded-xl">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground block mb-1">
                    Largest Hotspot by Area
                  </span>
                  <p className="text-sm font-bold text-foreground">
                    {keyInsights.largest ? `${keyInsights.largest.label} (${keyInsights.largest.area_name})` : "N/A"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {keyInsights.largest ? `Spanning ${keyInsights.largest.total_area_m2 > 1000 ? `${(keyInsights.largest.total_area_m2 / 1000).toFixed(1)}k` : Math.round(keyInsights.largest.total_area_m2)} m² of ocean surface.` : "No activity"}
                  </p>
                </div>

                {/* Highest Confidence */}
                <div className="p-3 bg-muted/10 border border-border/20 rounded-xl">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground block mb-1">
                    Highest-Confidence Hotspot
                  </span>
                  <p className="text-sm font-bold text-foreground">
                    {keyInsights.highestConf ? `${keyInsights.highestConf.label} (${keyInsights.highestConf.area_name})` : "N/A"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {keyInsights.highestConf ? `Mean model confidence of ${(keyInsights.highestConf.avg_confidence * 100).toFixed(1)}%.` : "No activity"}
                  </p>
                </div>

                {/* Recurring Hotspots */}
                <div className="p-3 bg-muted/10 border border-border/20 rounded-xl">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground block mb-1">
                    Recurring Accumulations
                  </span>
                  <p className="text-sm font-bold text-foreground">
                    {keyInsights.recurringCount} of {hotspots.length} Clusters
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {keyInsights.recurringCount > 0 ? "Hotspots detected repeatedly across multiple satellite acquisition dates." : "Detections confined to single-pass scenes."}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Collapsible Complete Data Breakdown with Coastal Area column */}
          <div className="glass-card border border-border/30 overflow-hidden">
            <div
              className="px-5 py-3.5 border-b border-border/20 flex items-center justify-between cursor-pointer hover:bg-muted/10 transition-colors"
              onClick={() => setExpandedTable(!expandedTable)}
            >
              <h3 className="font-heading font-semibold text-sm flex items-center gap-2">
                <Layers className="w-4 h-4 text-primary" />
                Complete Hotspot Clustering Table ({hotspots.length})
              </h3>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                {expandedTable ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </div>
            </div>

            {expandedTable && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border/20 text-muted-foreground text-xs">
                      {["#", "Cluster", "Coastal Area / Zone", "Center Coords", "Detections", "Runs", "Avg Confidence", "Total Area (m²)", "First Detected", "Last Detected"].map(h => (
                        <th key={h} className="text-left px-4 py-3 font-medium whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {hotspots.map(h => {
                      const norm = h.detection_count / maxDetections;
                      return (
                        <tr
                          key={h.id}
                          className={`border-b border-border/10 cursor-pointer transition-colors ${
                            selectedHotspot?.id === h.id ? "bg-primary/10" : "hover:bg-muted/10"
                          }`}
                          onClick={() => selectHotspot(h)}
                        >
                          <td className="px-4 py-2.5 font-mono text-muted-foreground text-xs">{String(h.id).padStart(2, "0")}</td>
                          <td className="px-4 py-2.5">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: intensityColor(norm) }} />
                              <span className="font-medium">{h.label}</span>
                            </div>
                          </td>
                          <td className="px-4 py-2.5">
                            <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold border ${h.area_badge_color}`}>
                              {h.area_name}
                            </span>
                          </td>
                          <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">{h.center[0].toFixed(4)}, {h.center[1].toFixed(4)}</td>
                          <td className="px-4 py-2.5 font-semibold">{h.detection_count}</td>
                          <td className="px-4 py-2.5">{h.run_count}</td>
                          <td className="px-4 py-2.5">{(h.avg_confidence * 100).toFixed(1)}%</td>
                          <td className="px-4 py-2.5">{h.total_area_m2.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                          <td className="px-4 py-2.5 text-xs text-muted-foreground">{formatDate(h.first_seen)}</td>
                          <td className="px-4 py-2.5 text-xs text-muted-foreground">{formatDate(h.last_seen)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default HotspotsPage;
