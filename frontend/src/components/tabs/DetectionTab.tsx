import React, { useEffect, useState, useMemo } from "react";
import { motion } from "framer-motion";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import { loadFinalReport } from "@/services/dataService";
import { DetectionFeatureCollection, DetectionFeature, POLYMER_COLORS } from "@/types";
import {
  Layers,
  Filter,
  X,
  Eye,
  EyeOff,
  ChevronRight,
  MapPin,
} from "lucide-react";

// Fit map to GeoJSON bounds
function FitBounds({ data }: { data: DetectionFeatureCollection | null }) {
  const map = useMap();
  useEffect(() => {
    if (data && data.features.length > 0) {
      const layer = L.geoJSON(data as any);
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.flyToBounds(bounds, { padding: [40, 40], duration: 1.2 });
      }
    }
  }, [data, map]);
  return null;
}

interface DetectionTabProps {
  runId: string;
}

const DetectionTab: React.FC<DetectionTabProps> = ({ runId }) => {
  const [data, setData] = useState<DetectionFeatureCollection | null>(null);
  const [loading, setLoading] = useState(true);
  const [showFP, setShowFP] = useState(true);
  const [minConf, setMinConf] = useState(0);
  const [minArea, setMinArea] = useState(0);
  const [typeFilter, setTypeFilter] = useState("all");
  const [tileKey, setTileKey] = useState<"satellite" | "dark">("satellite");
  const [selected, setSelected] = useState<DetectionFeature | null>(null);
  const [showFilters, setShowFilters] = useState(true);

  useEffect(() => {
    loadFinalReport(runId).then((d) => { setData(d); setLoading(false); }).catch(() => setLoading(false));
  }, [runId]);

  const filtered = useMemo(() => {
    if (!data) return null;
    const features = data.features.filter((f) => {
      const p = f.properties;
      if (!showFP && p.is_false_positive) return false;
      if (p.mean_confidence < minConf) return false;
      if (p.area_m2 < minArea) return false;
      if (typeFilter !== "all" && p.polymer_type !== typeFilter) return false;
      return true;
    });
    return { ...data, features };
  }, [data, showFP, minConf, minArea, typeFilter]);

  const polymerTypes = useMemo(() => {
    if (!data) return [];
    const types = new Set(data.features.map((f) => f.properties.polymer_type));
    return Array.from(types).sort();
  }, [data]);

  const stats = useMemo(() => {
    if (!filtered) return { total: 0, plastic: 0, fp: 0 };
    const total = filtered.features.length;
    const plastic = filtered.features.filter((f) => !f.properties.is_false_positive).length;
    return { total, plastic, fp: total - plastic };
  }, [filtered]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const TILE_LAYERS = {
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "Esri",
    },
    dark: {
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      attribution: "&copy; CARTO",
    },
  };

  const tile = TILE_LAYERS[tileKey];

  return (
    <div className="relative">
      <div className="glass-card overflow-hidden" style={{ height: "600px" }}>
        <MapContainer
          center={[16.1, -88.4]}
          zoom={10}
          className="w-full h-full"
          style={{ background: "#0A1628" }}
          zoomControl={false}
        >
          <TileLayer url={tile.url} attribution={tile.attribution} />
          {filtered && (
            <GeoJSON
              key={JSON.stringify({ showFP, minConf, minArea, typeFilter })}
              data={filtered as any}
              style={(f: any) => ({
                color: POLYMER_COLORS[f.properties.polymer_type] || "#6B7280",
                weight: 1.5,
                fillColor: POLYMER_COLORS[f.properties.polymer_type] || "#6B7280",
                fillOpacity: 0.35,
              })}
              onEachFeature={(feature, layer) => {
                layer.on("click", () => setSelected(feature as any));
              }}
            />
          )}
          <FitBounds data={filtered} />
        </MapContainer>

        {/* Filter Panel */}
        <div className={`absolute top-3 right-3 z-[1000] transition-all ${showFilters ? "w-[260px]" : ""}`}>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-1.5 px-3 py-2 glass rounded-lg text-sm font-medium hover:bg-muted/60 transition-colors mb-2"
          >
            <Filter className="w-4 h-4" />
            Filters
          </button>

          {showFilters && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass p-4 rounded-xl space-y-4 shadow-xl"
            >
              {/* Stats */}
              <div className="flex gap-3 text-xs">
                <div className="glass px-2.5 py-1.5 rounded-lg">
                  <span className="text-muted-foreground">Total: </span>
                  <span className="font-semibold">{stats.total}</span>
                </div>
                <div className="glass px-2.5 py-1.5 rounded-lg">
                  <span className="text-muted-foreground">Plastic: </span>
                  <span className="font-semibold text-red-400">{stats.plastic}</span>
                </div>
              </div>

              {/* Show FP toggle */}
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-xs text-muted-foreground">Show False Positives</span>
                <button
                  onClick={() => setShowFP(!showFP)}
                  className={`p-1 rounded ${showFP ? "text-primary" : "text-muted-foreground"}`}
                >
                  {showFP ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                </button>
              </label>

              {/* Min confidence */}
              <div>
                <label className="text-xs text-muted-foreground block mb-1">
                  Min Confidence: <span className="text-foreground font-medium">{minConf.toFixed(2)}</span>
                </label>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={minConf}
                  onChange={(e) => setMinConf(parseFloat(e.target.value))}
                  className="w-full accent-primary h-1.5"
                />
              </div>

              {/* Min area */}
              <div>
                <label className="text-xs text-muted-foreground block mb-1">
                  Min Area (m²): <span className="text-foreground font-medium">{minArea}</span>
                </label>
                <input
                  type="range" min={0} max={10000} step={100}
                  value={minArea}
                  onChange={(e) => setMinArea(parseInt(e.target.value))}
                  className="w-full accent-primary h-1.5"
                />
              </div>

              {/* Type filter */}
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Polymer Type</label>
                <select
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value)}
                  className="w-full px-2.5 py-1.5 bg-muted/50 border border-border/50 rounded-lg text-xs text-foreground focus:outline-none"
                >
                  <option value="all">All Types</option>
                  {polymerTypes.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              {/* Tile layer toggle */}
              <div className="flex gap-1.5">
                {(["satellite", "dark"] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setTileKey(mode)}
                    className={`flex-1 px-2 py-1.5 text-xs rounded-lg capitalize transition-colors ${
                      tileKey === mode ? "bg-primary text-primary-foreground font-semibold" : "bg-muted/40 text-muted-foreground"
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </div>

        {/* Selected feature detail */}
        {selected && (
          <div className="absolute bottom-3 left-3 z-[1000] glass p-4 rounded-xl max-w-xs shadow-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold flex items-center gap-1.5">
                <MapPin className="w-3.5 h-3.5 text-primary" />
                Cluster #{selected.properties.cluster_id}
              </span>
              <button onClick={() => setSelected(null)} className="text-muted-foreground hover:text-foreground">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Type</span>
                <span className="font-medium">{selected.properties.polymer_type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Area</span>
                <span className="font-medium">{selected.properties.area_m2.toLocaleString()} m²</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-medium">{selected.properties.mean_confidence.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Coords</span>
                <span className="font-mono text-[11px]">
                  {selected.properties.centroid_lat.toFixed(4)}, {selected.properties.centroid_lon.toFixed(4)}
                </span>
              </div>
              {selected.properties.source_type && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Source</span>
                  <span className="font-medium capitalize">{selected.properties.source_type}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="absolute bottom-3 right-3 z-[1000] glass px-3 py-2.5 rounded-lg text-xs space-y-1.5 shadow-lg max-h-[200px] overflow-y-auto">
          {Object.entries(POLYMER_COLORS).slice(0, 6).map(([name, color]) => (
            <div key={name} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
              <span className="text-foreground text-[11px]">{name.replace("False Positive ", "FP ")}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DetectionTab;
