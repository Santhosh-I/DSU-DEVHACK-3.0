import React, { useEffect, useState, useMemo } from "react";
import { motion } from "framer-motion";
import { MapContainer, TileLayer, CircleMarker, Rectangle, Polyline, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { loadAttribution, loadBacktrackSummary, loadRunMetadata, loadAllBacktrackGeoJsons, loadDebrisSummaryCsv } from "@/services/dataService";
import { AttributionEntry, BacktrackEntry, RunMetadata, SOURCE_ICONS, DebrisSummaryRow } from "@/types";
import { backtrackCluster } from "@/lib/api";
import {
  GitBranch,
  ChevronDown,
  ChevronUp,
  Database,
  Cpu,
  Wind,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
} from "lucide-react";

const SCORE_COLORS = {
  fishing: "#3B82F6",
  industrial: "#F59E0B",
  shipping: "#8B5CF6",
  river: "#10B981",
};

const TILE_LAYERS = {
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Esri",
    label: "Satellite",
  },
  dark: {
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution: "&copy; CARTO &copy; OpenStreetMap",
    label: "Dark",
  },
  streets: {
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap",
    label: "Streets",
  },
};

// Fit bounds helper
function FitBacktrackBounds({ backtrack }: { backtrack: BacktrackEntry[] }) {
  const map = useMap();
  useEffect(() => {
    if (backtrack && backtrack.length > 0) {
      const bounds = L.latLngBounds([]);
      backtrack.forEach((bt) => {
        bounds.extend([bt.source_centroid[1], bt.source_centroid[0]]);
        bounds.extend([bt.source_bbox[1], bt.source_bbox[0]]);
        bounds.extend([bt.source_bbox[3], bt.source_bbox[2]]);
      });
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [50, 50] });
      }
    }
  }, [backtrack, map]);
  return null;
}

interface AttributionTabProps {
  runId: string;
}

const AttributionTab: React.FC<AttributionTabProps> = ({ runId }) => {
  const [attribution, setAttribution] = useState<AttributionEntry[]>([]);
  const [backtrack, setBacktrack] = useState<BacktrackEntry[]>([]);
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [csv, setCsv] = useState<DebrisSummaryRow[]>([]);
  const [backtrackGeoJsons, setBacktrackGeoJsons] = useState<Record<number, any>>({});
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [backtrackingIds, setBacktrackingIds] = useState<Set<number>>(new Set());

  // Map & Animation States
  const [tileKey, setTileKey] = useState<"satellite" | "dark" | "streets">("satellite");
  const [isPlaying, setIsPlaying] = useState(false);
  const [animProgress, setAnimProgress] = useState(1);
  const [animSpeed, setAnimSpeed] = useState<number>(1);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);

  // Map Layer Visibility Toggles
  const [showDebrisPoints, setShowDebrisPoints] = useState(true);
  const [showBacktrackPaths, setShowBacktrackPaths] = useState(true);
  const [showSourceRegions, setShowSourceRegions] = useState(true);
  const [showNonBacktracked, setShowNonBacktracked] = useState(true);

  useEffect(() => {
    Promise.all([
      loadAttribution(runId), 
      loadBacktrackSummary(runId), 
      loadRunMetadata(runId),
      loadDebrisSummaryCsv(runId)
    ]).then(
      async ([attr, bt, meta, csvData]) => {
        setAttribution(attr);
        setBacktrack(bt);
        setMetadata(meta);
        setCsv(csvData);

        const clusterIds = bt.map((b) => b.cluster_id);
        const geoJsons = await loadAllBacktrackGeoJsons(clusterIds, runId);
        setBacktrackGeoJsons(geoJsons);

        setLoading(false);
      }
    );
  }, [runId]);

  // Animation Loop
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isPlaying) {
      timer = setInterval(() => {
        setAnimProgress((prev) => {
          if (prev >= 1) return 0.05;
          return Math.min(1, prev + 0.02 * animSpeed);
        });
      }, 50);
    }
    return () => clearInterval(timer);
  }, [isPlaying, animSpeed]);

  // Compute sliced trajectories for animated backtrack paths
  const slicedTrajectories = useMemo(() => {
    const lines: { id: string; clusterId: number; positions: [number, number][]; endPoint: [number, number] | null }[] = [];

    Object.entries(backtrackGeoJsons).forEach(([clusterIdStr, geojson]) => {
      const clusterId = parseInt(clusterIdStr);
      if (selectedCluster !== null && selectedCluster !== clusterId) return;
      if (!geojson || !geojson.features) return;

      geojson.features.forEach((feature: any, idx: number) => {
        if (feature.geometry && feature.geometry.type === "LineString") {
          const coords = feature.geometry.coordinates;
          if (!coords || coords.length < 2) return;

          const totalPts = coords.length;
          const ptsToTake = Math.max(2, Math.floor(totalPts * animProgress));
          const sliced = coords.slice(0, ptsToTake);

          const positions: [number, number][] = sliced.map(([lon, lat]: [number, number]) => [lat, lon]);
          const lastPt = positions[positions.length - 1];

          lines.push({
            id: `${clusterId}-${idx}`,
            clusterId,
            positions,
            endPoint: lastPt,
          });
        }
      });
    });

    return lines;
  }, [backtrackGeoJsons, animProgress, selectedCluster]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const handleBacktrack = async (clusterId: number) => {
    setBacktrackingIds(prev => new Set(prev).add(clusterId));
    try {
      await backtrackCluster(runId, clusterId);
      // Wait a bit, then refresh the data
      setTimeout(() => {
        Promise.all([loadAttribution(runId), loadBacktrackSummary(runId)]).then(
          async ([attr, bt]) => {
            setAttribution(attr);
            setBacktrack(bt);
            const clusterIds = bt.map((b) => b.cluster_id);
            const geoJsons = await loadAllBacktrackGeoJsons(clusterIds, runId);
            setBacktrackGeoJsons(geoJsons);
            setBacktrackingIds(prev => {
              const next = new Set(prev);
              next.delete(clusterId);
              return next;
            });
          }
        );
      }, 2000);
    } catch (e) {
      console.error(e);
      setBacktrackingIds(prev => {
        const next = new Set(prev);
        next.delete(clusterId);
        return next;
      });
    }
  };

  const nonBacktrackedPlastics = csv.filter(
    (row) => row.polymer_type === "Marine Debris (Plastic)" && !backtrack.some((bt) => bt.cluster_id === row.cluster_id)
  );

  const scoreData = attribution.map((a) => ({
    name: `#${a.debris_cluster_id}`,
    fishing: parseFloat((a.fishing_score * 100).toFixed(1)),
    industrial: parseFloat((a.industrial_score * 100).toFixed(1)),
    shipping: parseFloat((a.shipping_score * 100).toFixed(1)),
    river: parseFloat((a.river_score * 100).toFixed(1)),
  }));

  const confColor = (c: string) =>
    c === "high" ? "text-emerald-400 bg-emerald-500/15" : c === "medium" ? "text-yellow-400 bg-yellow-500/15" : "text-red-400 bg-red-500/15";

  const currentTile = TILE_LAYERS[tileKey];

  return (
    <div className="space-y-6">
      {/* Hydrodynamic Backtrack Map */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card overflow-hidden flex flex-col"
      >
        <div className="px-5 py-3.5 border-b border-border/30 flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-primary" />
            <h2 className="font-heading font-semibold text-lg">Hydrodynamic Backtrack Map</h2>
          </div>

          <div className="flex items-center gap-3">
            {/* Cluster Filter */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Cluster Filter:</span>
              <select
                value={selectedCluster ?? "all"}
                onChange={(e) => setSelectedCluster(e.target.value === "all" ? null : parseInt(e.target.value))}
                className="px-2.5 py-1 bg-muted/50 border border-border/50 rounded-lg text-xs text-foreground focus:outline-none"
              >
                <option value="all">All Clusters ({backtrack.length})</option>
                {backtrack.map((b) => (
                  <option key={b.cluster_id} value={b.cluster_id}>
                    Cluster #{b.cluster_id}
                  </option>
                ))}
              </select>
            </div>

            {/* Layer switcher */}
            <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-lg border border-border/30">
              {(["satellite", "dark", "streets"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setTileKey(mode)}
                  className={`px-3 py-1 text-xs font-medium rounded-md capitalize transition-colors ${
                    tileKey === mode ? "bg-primary text-primary-foreground font-semibold shadow" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Map Canvas */}
        <div className="h-[480px] w-full relative">
          <MapContainer
            center={[16.28, -88.5]}
            zoom={9}
            className="w-full h-full"
            style={{ background: "#0A1628" }}
            zoomControl={false}
          >
            <TileLayer url={currentTile.url} attribution={currentTile.attribution} />
            <FitBacktrackBounds backtrack={backtrack} />

            {showBacktrackPaths && slicedTrajectories.map((traj) => (
              <Polyline
                key={traj.id}
                positions={traj.positions}
                pathOptions={{
                  color: "#38bdf8",
                  weight: 1.5,
                  opacity: 0.65,
                }}
              />
            ))}

            {showBacktrackPaths && isPlaying &&
              slicedTrajectories.map((traj) =>
                traj.endPoint ? (
                  <CircleMarker
                    key={`head-${traj.id}`}
                    center={traj.endPoint}
                    radius={2.5}
                    pathOptions={{
                      color: "#38bdf8",
                      fillColor: "#60a5fa",
                      fillOpacity: 0.9,
                      weight: 1,
                    }}
                  />
                ) : null
              )}

            {showSourceRegions && backtrack.map((bt) => (
              <Rectangle
                key={`bbox-${bt.cluster_id}`}
                bounds={[
                  [bt.source_bbox[1], bt.source_bbox[0]],
                  [bt.source_bbox[3], bt.source_bbox[2]],
                ]}
                pathOptions={{
                  color: "#F59E0B",
                  weight: 1.8,
                  fillColor: "#F59E0B",
                  fillOpacity: 0.12,
                  dashArray: "4 4",
                }}
              >
                <Popup>
                  <div className="text-xs">
                    <strong>Source Area for Cluster #{bt.cluster_id}</strong><br />
                    BBox: [{bt.source_bbox.join(", ")}]<br />
                    Probability: {(bt.source_probability * 100).toFixed(1)}%
                  </div>
                </Popup>
              </Rectangle>
            ))}

            {showDebrisPoints && backtrack.map((bt) => (
              <CircleMarker
                key={`cluster-${bt.cluster_id}`}
                center={[bt.release_lat || bt.source_centroid[1], bt.release_lon || bt.source_centroid[0]]}
                radius={8}
                pathOptions={{
                  color: "#FFFFFF",
                  fillColor: "#EF4444",
                  fillOpacity: 0.9,
                  weight: 2,
                }}
              >
                <Popup>
                  <div className="text-xs">
                    <strong>Debris Cluster #{bt.cluster_id}</strong><br />
                    Particles Tracked: {bt.n_particles}<br />
                    Backtrack Duration: {bt.days_to_source} Days
                  </div>
                </Popup>
              </CircleMarker>
            ))}

            {showNonBacktracked && nonBacktrackedPlastics.map((row) => (
              <CircleMarker
                key={`nb-cluster-${row.cluster_id}`}
                center={[row.lat, row.lon]}
                radius={7}
                pathOptions={{
                  color: "#FFFFFF",
                  fillColor: "#6B7280",
                  fillOpacity: 0.7,
                  weight: 1.5,
                  dashArray: "2 2",
                }}
              >
                <Popup>
                  <div className="text-xs flex flex-col gap-2 p-1">
                    <div>
                      <strong>Debris Cluster #{row.cluster_id}</strong><br />
                      Not Backtracked
                    </div>
                    <button 
                      onClick={() => handleBacktrack(row.cluster_id)}
                      disabled={backtrackingIds.has(row.cluster_id)}
                      className="px-3 py-1.5 bg-primary text-primary-foreground rounded text-xs font-semibold disabled:opacity-50"
                    >
                      {backtrackingIds.has(row.cluster_id) ? "Backtracking..." : "Backtrack this cluster"}
                    </button>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>

          {/* Map Legend & Layer Toggles */}
          <div className="absolute bottom-3 left-3 z-[1000] glass px-3.5 py-3 rounded-lg text-xs space-y-2.5 shadow-lg">
            <label className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
              <input type="checkbox" className="accent-primary w-3.5 h-3.5 rounded bg-background/50 border-border/50" checked={showDebrisPoints} onChange={(e) => setShowDebrisPoints(e.target.checked)} />
              <div className="w-3 h-3 rounded-full bg-red-500 border border-white" />
              <span className="text-foreground font-medium select-none">Debris Detection Point</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
              <input type="checkbox" className="accent-primary w-3.5 h-3.5 rounded bg-background/50 border-border/50" checked={showBacktrackPaths} onChange={(e) => setShowBacktrackPaths(e.target.checked)} />
              <div className="w-5 h-1 bg-sky-400 rounded" />
              <span className="text-foreground font-medium select-none">Backtrack Particle Path</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
              <input type="checkbox" className="accent-primary w-3.5 h-3.5 rounded bg-background/50 border-border/50" checked={showSourceRegions} onChange={(e) => setShowSourceRegions(e.target.checked)} />
              <div className="w-3 h-3 border border-dashed border-amber-500 bg-amber-500/20 rounded-sm" />
              <span className="text-foreground font-medium select-none">Inferred Source Region</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
              <input type="checkbox" className="accent-primary w-3.5 h-3.5 rounded bg-background/50 border-border/50" checked={showNonBacktracked} onChange={(e) => setShowNonBacktracked(e.target.checked)} />
              <div className="w-3 h-3 rounded-full bg-gray-500 border border-white border-dashed" />
              <span className="text-foreground font-medium select-none">Non-Backtracked Plastic</span>
            </label>
          </div>
        </div>

        {/* Animation Controls */}
        <div className="p-3.5 bg-muted/20 border-t border-border/30 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex items-center gap-1.5 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-colors"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? "Pause" : "Play Animation"}
            </button>
            <button
              onClick={() => { setIsPlaying(false); setAnimProgress(1); }}
              className="p-2 glass rounded-lg text-muted-foreground hover:text-foreground transition-colors"
              title="Reset Timeline"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 min-w-[240px] flex items-center gap-3">
            <span className="text-muted-foreground font-mono font-medium">Day -7.0</span>
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.01}
              value={animProgress}
              onChange={(e) => { setIsPlaying(false); setAnimProgress(parseFloat(e.target.value)); }}
              className="flex-1 accent-primary cursor-pointer h-2"
            />
            <span className="text-muted-foreground font-mono font-medium">Day 0.0</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-muted-foreground mr-1 font-medium">Speed:</span>
            {[1, 2, 4].map((spd) => (
              <button
                key={spd}
                onClick={() => setAnimSpeed(spd)}
                className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                  animSpeed === spd ? "bg-primary/20 text-primary border border-primary/30 font-semibold" : "text-muted-foreground hover:bg-muted/40"
                }`}
              >
                {spd}x
              </button>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Attribution Results + Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Attribution Table */}
        <div className="glass-card overflow-hidden">
          <div className="px-5 py-4 border-b border-border/30 flex items-center justify-between">
            <h2 className="font-heading font-semibold text-lg">Attribution Results</h2>
            <span className="text-xs text-muted-foreground">{attribution.length} records</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/20 text-muted-foreground">
                  <th className="text-left px-4 py-3 font-medium">Cluster</th>
                  <th className="text-left px-4 py-3 font-medium">Source</th>
                  <th className="text-left px-4 py-3 font-medium">Location</th>
                  <th className="text-left px-4 py-3 font-medium">Country</th>
                  <th className="text-left px-4 py-3 font-medium">Score</th>
                  <th className="text-left px-4 py-3 font-medium">Conf.</th>
                  <th className="text-left px-4 py-3 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {attribution.map((a) => (
                  <React.Fragment key={a.debris_cluster_id}>
                    <tr
                      onClick={() => setSelectedCluster(selectedCluster === a.debris_cluster_id ? null : a.debris_cluster_id)}
                      className={`border-b border-border/10 cursor-pointer transition-colors ${
                        selectedCluster === a.debris_cluster_id ? "bg-primary/15" : "hover:bg-muted/20"
                      }`}
                    >
                      <td className="px-4 py-3 font-mono text-primary font-semibold">#{a.debris_cluster_id}</td>
                      <td className="px-4 py-3">
                        <span className="flex items-center gap-1.5">
                          <span>{SOURCE_ICONS[a.source_type] || "❓"}</span>
                          <span className="capitalize">{a.source_type}</span>
                        </span>
                      </td>
                      <td className="px-4 py-3 text-muted-foreground text-xs">{a.location_name}</td>
                      <td className="px-4 py-3 text-muted-foreground text-xs">{a.country}</td>
                      <td className="px-4 py-3 font-semibold">{(a.attribution_score * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${confColor(a.confidence)}`}>
                          {a.confidence}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setExpandedId(expandedId === a.debris_cluster_id ? null : a.debris_cluster_id);
                          }}
                        >
                          {expandedId === a.debris_cluster_id ? (
                            <ChevronUp className="w-4 h-4 text-muted-foreground" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-muted-foreground" />
                          )}
                        </button>
                      </td>
                    </tr>
                    {expandedId === a.debris_cluster_id && (
                      <tr>
                        <td colSpan={7} className="px-4 py-3 bg-muted/10">
                          <p className="text-sm text-muted-foreground leading-relaxed">{a.explanation}</p>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right side: Score Breakdown Chart & Config */}
        <div className="space-y-6">
          {/* Score breakdown chart */}
          <div className="glass-card p-5">
            <h3 className="font-heading font-semibold mb-4">Source Score Breakdown</h3>
            <div className="h-[230px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={scoreData} layout="vertical" barGap={2}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 20% 16%)" />
                  <XAxis type="number" domain={[0, 50]} tick={{ fill: "#6B7280", fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: "#9CA3AF", fontSize: 12 }} width={50} />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(220 30% 8%)",
                      border: "1px solid hsl(215 20% 16%)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="fishing" stackId="a" fill={SCORE_COLORS.fishing} name="Fishing" />
                  <Bar dataKey="industrial" stackId="a" fill={SCORE_COLORS.industrial} name="Industrial" />
                  <Bar dataKey="shipping" stackId="a" fill={SCORE_COLORS.shipping} name="Shipping" />
                  <Bar dataKey="river" stackId="a" fill={SCORE_COLORS.river} name="River" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="flex gap-4 mt-3 justify-center">
              {Object.entries(SCORE_COLORS).map(([key, color]) => (
                <div key={key} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color }} />
                  <span className="capitalize">{key}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Backtrack config */}
          {metadata && (
            <div className="glass-card p-5">
              <h3 className="font-heading font-semibold mb-4">Backtrack Configuration</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                {[
                  { icon: Database, label: "Ocean Data", value: metadata.cmems_product.split("_").slice(-2).join("_") },
                  { icon: Wind, label: "Wind Data", value: "ERA5 (u10, v10)" },
                  { icon: Cpu, label: "Integrator", value: metadata.integrator },
                  { icon: Cpu, label: "Time Step", value: `${metadata.time_step_hours}h` },
                  { icon: GitBranch, label: "Particles", value: `${metadata.n_particles} per cluster` },
                  { icon: GitBranch, label: "Duration", value: `${metadata.bt_days} days` },
                  { icon: Cpu, label: "Diffusion (Kh)", value: metadata.horizontal_diffusion_kh.toString() },
                ].map((item) => {
                  const Icon = item.icon;
                  return (
                    <div key={item.label} className="flex items-start gap-3 p-3 bg-muted/20 rounded-lg">
                      <Icon className="w-4 h-4 text-primary mt-0.5" />
                      <div>
                        <div className="text-xs text-muted-foreground">{item.label}</div>
                        <div className="text-sm font-medium">{item.value}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="mt-4">
                <div className="text-xs text-muted-foreground mb-2">Kernels</div>
                <div className="flex flex-wrap gap-2">
                  {metadata.kernels.map((k) => (
                    <span key={k} className="px-2.5 py-1 bg-primary/10 text-primary rounded-md text-xs font-medium">
                      {k}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Diagnostics Table */}
      <div className="glass-card overflow-hidden">
        <div className="px-5 py-4 border-b border-border/30">
          <h2 className="font-heading font-semibold text-lg">Backtracking Diagnostics</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/20 text-muted-foreground">
                <th className="text-left px-4 py-3 font-medium">Cluster</th>
                <th className="text-left px-4 py-3 font-medium">Mean Trajectory (km)</th>
                <th className="text-left px-4 py-3 font-medium">Avg Drift Speed (m/s)</th>
                <th className="text-left px-4 py-3 font-medium">Velocity Std Dev (m/s)</th>
                <th className="text-left px-4 py-3 font-medium">Smoothness Index</th>
                <th className="text-left px-4 py-3 font-medium">Forcing Completeness</th>
              </tr>
            </thead>
            <tbody>
              {backtrack.map((bt) => (
                <tr key={bt.cluster_id} className="border-b border-border/10 hover:bg-muted/20 transition-colors">
                  <td className="px-4 py-3 font-mono text-primary font-semibold">#{bt.cluster_id}</td>
                  <td className="px-4 py-3">{(bt.diagnostics?.trajectory_mean_length_km || 0).toFixed(2)}</td>
                  <td className="px-4 py-3">{(bt.diagnostics?.average_drift_speed_ms || 0).toFixed(4)}</td>
                  <td className="px-4 py-3">{(bt.diagnostics?.ensemble_velocity_std_ms || 0).toFixed(4)}</td>
                  <td className="px-4 py-3">{(bt.diagnostics?.trajectory_smoothness_index || 0).toFixed(4)}</td>
                  <td className="px-4 py-3">{(bt.diagnostics?.forcing_data_completeness_pct || 0).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AttributionTab;
