import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
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
  CalendarRange, ArrowRight, RotateCcw,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

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
}

interface ActivityTimelinePoint {
  date: string;
  detections: number;
  label: string;
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

  // Filters
  const [minConf, setMinConf] = useState(0);
  const [showFP, setShowFP] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [typeFilter, setTypeFilter] = useState("all");

  // Cloud Heatmap appearance controls
  const [showCloudHeatmap, setShowCloudHeatmap] = useState(true);
  const [cloudRadius, setCloudRadius] = useState(34);
  const [cloudOpacity, setCloudOpacity] = useState(0.85);

  // Source attribution controls
  const [showSources, setShowSources] = useState(true);

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
                
                const pointDate = (p.detection_date || cleanDate).substring(0, 10);

                detections.push({
                  lat: p.centroid_lat || coords[1],
                  lng: p.centroid_lon || coords[0],
                  confidence: p.mean_confidence || 0,
                  polymer_type: p.polymer_type || "Unknown",
                  is_false_positive: p.is_false_positive || false,
                  detection_date: pointDate,
                  area_m2: p.area_m2 || 0,
                  run_id: run.id,
                  run_name: run.run_name || run.id.substring(0, 8),
                  cluster_id: p.cluster_id || 0,
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

  // ── Most Active Locations (Ranked by detection count in selected range) ────
  const mostActiveLocations = useMemo(() => {
    return hotspots.slice(0, 6);
  }, [hotspots]);

  // ── Summary KPIs ───────────────────────────────────────────────────────────
  const totalDetections = filteredDetections.length;
  const mostActiveHotspot = hotspots.length > 0 ? hotspots[0] : null;
  const activeHotspotsCount = hotspots.length;
  const totalAffectedAreaM2 = useMemo(() => {
    return filteredDetections.reduce((sum, p) => sum + p.area_m2, 0);
  }, [filteredDetections]);
  const backtrackedSourcesCount = filteredSources.length;

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
      mostActive: mostActiveHotspot,
      peakActivity: peakActivityPoint,
      largest: largestHotspot,
      highestConf: highestConfHotspot,
      recurringCount: recurringHotspotsCount,
      topPolymer: topPolymer ? `${topPolymer[0]} (${topPolymer[1]} detections)` : "N/A",
    };
  }, [hotspots, filteredDetections, peakActivityPoint, mostActiveHotspot]);

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
          <p className="text-foreground font-medium">Aggregating Historical & Temporal Plastic Data</p>
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
            Complete at least one pipeline run to view hotspot and time-range analysis.
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
              Hotspot Locations & Time-Range Analysis
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Historical convergence, cloud density plumes, and period-specific debris concentrations across {runs.length} pipeline runs.
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
            2. KEY SUMMARY (Compact 5 KPIs with highlighted Most Active Location)
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

          {/* Most active location (CLEARLY HIGHLIGHTED) */}
          <div className="glass-card p-4 border-2 border-primary/60 bg-primary/10 shadow-lg shadow-primary/10 flex flex-col justify-between relative overflow-hidden">
            <div className="absolute top-0 right-0 px-2 py-0.5 bg-primary text-primary-foreground text-[9px] font-bold uppercase rounded-bl tracking-wider">
              Highest Activity
            </div>
            <div className="flex items-center justify-between text-primary mb-1">
              <span className="text-xs font-semibold flex items-center gap-1">
                <Flame className="w-3.5 h-3.5 text-primary" /> Most Active Location
              </span>
            </div>
            <div>
              <p className="text-xl font-bold font-heading text-foreground truncate">
                {mostActiveHotspot ? mostActiveHotspot.label : "None"}
              </p>
              <p className="text-xs text-primary font-medium mt-0.5">
                {mostActiveHotspot ? `${mostActiveHotspot.detection_count} detections (${((mostActiveHotspot.detection_count / (totalDetections || 1)) * 100).toFixed(0)}% of period)` : "No detections in range"}
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
              <p className="text-[11px] text-muted-foreground mt-0.5">spatial concentration clusters</p>
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
              <p className="text-[11px] text-muted-foreground mt-0.5">attributed origin places</p>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════
            3. HOTSPOT MAP (Interactive Leaflet Map with Cloud Heatmap)
        ═══════════════════════════════════════════════════════════ */}
        <div className="glass-card overflow-hidden relative rounded-xl border border-border/40" style={{ height: "600px" }}>
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

            {/* Hotspot Centroid Rings */}
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
                    <div className="text-xs space-y-1 min-w-[170px]" style={{ color: "#0f172a" }}>
                      <div className="font-bold text-sm text-slate-900">{h.label}</div>
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

                {/* Source Attributed Places Toggle */}
                <div className="pt-2 border-t border-border/30">
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

          {/* ── Map Legend ── */}
          <div className="absolute bottom-3 left-3 z-[1000] glass px-3.5 py-2.5 rounded-lg shadow-lg max-w-[240px]">
            <p className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground mb-1.5">Visualization Legend</p>
            <div className="mb-2">
              <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                <span>Cloud Density</span>
                <span className="text-primary font-medium">Plume</span>
              </div>
              <div className="h-2 w-full rounded-full" style={{ background: "linear-gradient(to right, #0284c7, #06b6d4, #10b981, #facc15, #f97316, #ef4444)" }} />
            </div>
            <div className="space-y-1.5 pt-1.5 border-t border-border/20 text-[11px]">
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-3 h-3 rounded-full bg-purple-400 border border-white shadow-sm flex-shrink-0" />
                <span className="text-foreground font-medium">Source Attributed Place</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-3 h-3 rounded-full border-2 border-white bg-orange-500 flex-shrink-0" />
                <span>Hotspot Centroid</span>
              </div>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════
            4. TOP ACTIVE LOCATIONS
            Rank locations by plastic activity for the selected date range
        ═══════════════════════════════════════════════════════════ */}
        <div className="glass-card p-5 border border-border/40">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-semibold font-heading flex items-center gap-2 text-foreground">
                <Award className="w-4 h-4 text-primary" />
                Most Active Locations
              </h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                Ranked by verified plastic detections in the selected period. Click any location to focus the map.
              </p>
            </div>
            <span className="text-xs text-muted-foreground font-mono bg-muted/30 px-2 py-1 rounded">
              {hotspots.length} clusters identified
            </span>
          </div>

          {mostActiveLocations.length === 0 ? (
            <div className="p-6 text-center text-sm text-muted-foreground border border-dashed border-border/40 rounded-lg">
              No plastic concentration locations recorded within the selected date range.
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {mostActiveLocations.map((h, index) => {
                const norm = h.detection_count / maxDetections;
                const isSelected = selectedHotspot?.id === h.id;
                const percentOfTotal = ((h.detection_count / (totalDetections || 1)) * 100).toFixed(1);

                return (
                  <motion.div
                    key={h.id}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    onClick={() => selectHotspot(h)}
                    className={`p-3.5 rounded-xl border cursor-pointer transition-all flex flex-col justify-between ${
                      isSelected
                        ? "border-primary bg-primary/10 shadow-md shadow-primary/5"
                        : index === 0
                        ? "border-primary/40 bg-gradient-to-br from-primary/10 to-transparent hover:border-primary/70"
                        : "border-border/30 bg-muted/10 hover:bg-muted/20 hover:border-border/60"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-2">
                          <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-bold ${
                            index === 0 ? "bg-primary text-primary-foreground" : "bg-muted/50 text-muted-foreground font-mono"
                          }`}>
                            {index + 1}
                          </span>
                          <span className="font-semibold text-sm text-foreground">{h.label}</span>
                        </div>
                        <span className="text-[11px] font-bold text-foreground">
                          {h.detection_count} <span className="font-normal text-muted-foreground">detections</span>
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-[11px] text-muted-foreground my-2">
                        <div>
                          <span>Coordinates:</span>
                          <span className="font-mono text-foreground block">{h.center[0].toFixed(3)}, {h.center[1].toFixed(3)}</span>
                        </div>
                        <div>
                          <span>Area / Share:</span>
                          <span className="text-foreground block">{h.total_area_m2 > 1000 ? `${(h.total_area_m2 / 1000).toFixed(1)}k m²` : `${Math.round(h.total_area_m2)} m²`} ({percentOfTotal}%)</span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-2 pt-2 border-t border-border/20 flex items-center justify-between text-[11px]">
                      <span className="text-muted-foreground">{h.run_count} contributing run{h.run_count !== 1 ? "s" : ""}</span>
                      <span className="text-primary font-medium hover:underline flex items-center gap-0.5">
                        Focus Map →
                      </span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
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
                  <h3 className="font-heading font-semibold text-base flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-primary" />
                    {selectedHotspot.label} — Hotspot Specific Metrics
                  </h3>
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
                {/* Most Active */}
                <div className="p-3 bg-muted/10 border border-border/20 rounded-xl">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground block mb-1">
                    Primary Plastic Convergence
                  </span>
                  <p className="text-sm font-bold text-foreground">
                    {keyInsights.mostActive ? keyInsights.mostActive.label : "N/A"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {keyInsights.mostActive ? `Concentrated ${keyInsights.mostActive.detection_count} detections across coordinates (${keyInsights.mostActive.center[0].toFixed(2)}°, ${keyInsights.mostActive.center[1].toFixed(2)}°).` : "No activity"}
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
                    {keyInsights.largest ? keyInsights.largest.label : "N/A"}
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
                    {keyInsights.highestConf ? keyInsights.highestConf.label : "N/A"}
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

                {/* Dominant Polymer */}
                <div className="p-3 bg-muted/10 border border-border/20 rounded-xl">
                  <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground block mb-1">
                    Dominant Plastic Signature
                  </span>
                  <p className="text-sm font-bold text-foreground truncate">
                    {keyInsights.topPolymer}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Most frequently classified spectral polymer profile in this timeframe.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Collapsible Complete Data Breakdown */}
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
                      {["#", "Cluster", "Center Coords", "Detections", "Runs", "Avg Confidence", "Total Area (m²)", "First Detected", "Last Detected"].map(h => (
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
