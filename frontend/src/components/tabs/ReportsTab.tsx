import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getReportFiles, getSceneId, loadDebrisSummaryCsv, loadRunSummary } from "@/services/dataService";
import { DebrisSummaryRow } from "@/types";
import {
  FileText,
  Download,
  Globe,
  Table2,
  Map,
  Image as ImageIcon,
  Eye,
} from "lucide-react";

interface ReportsTabProps {
  runId: string;
}

const ReportsTab: React.FC<ReportsTabProps> = ({ runId }) => {
  const [sceneId, setSceneId] = useState<string | null>(null);
  const REPORT_FILES = sceneId ? getReportFiles(runId, sceneId) : null;

  const reportCards = REPORT_FILES ? [
    {
      title: "PDF Report",
      desc: "Executive summary with detection maps, polymer charts, and attribution tables.",
      icon: FileText,
      file: REPORT_FILES.pdf,
      size: "1.2 MB",
      color: "bg-red-500/15 text-red-400",
    },
    {
      title: "GeoJSON Data",
      desc: "All detections with polygons & attribution data for GIS tools.",
      icon: Globe,
      file: REPORT_FILES.geojson,
      size: "498 KB",
      color: "bg-blue-500/15 text-blue-400",
    },
    {
      title: "CSV Summary",
      desc: "One row per cluster with all fields. Open in Excel or any spreadsheet.",
      icon: Table2,
      file: REPORT_FILES.csv,
      size: "98 KB",
      color: "bg-emerald-500/15 text-emerald-400",
    },
    {
      title: "Interactive Backtrack Map",
      desc: "Folium/Leaflet interactive HTML map with particle trajectories.",
      icon: Map,
      file: REPORT_FILES.backtrackMap,
      size: "1.8 MB",
      color: "bg-purple-500/15 text-purple-400",
    },
  ] : [];

  const imageCards = REPORT_FILES ? [
    { title: "RGB Scene with Detections", file: REPORT_FILES.rgbMap, size: "1.1 MB" },
    { title: "Detection Map", file: REPORT_FILES.detectionMap, size: "25 KB" },
    { title: "Polymer Distribution", file: REPORT_FILES.polymerDist, size: "88 KB" },
  ] : [];

  const [csvData, setCsvData] = useState<DebrisSummaryRow[]>([]);
  const [showCsv, setShowCsv] = useState(false);
  const [showBacktrackMap, setShowBacktrackMap] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  useEffect(() => {
    loadRunSummary(runId).then((summary) => setSceneId(getSceneId(summary)));
    loadDebrisSummaryCsv(runId).then(setCsvData).catch(() => {});
  }, [runId]);

  if (!REPORT_FILES) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div>
      {/* Download cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-8">
        {reportCards.map((card, i) => {
          const Icon = card.icon;
          return (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass-card p-5 flex flex-col"
            >
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3 ${card.color}`}>
                <Icon className="w-5 h-5" />
              </div>
              <h3 className="font-heading font-semibold mb-1">{card.title}</h3>
              <p className="text-xs text-muted-foreground mb-4 flex-1">{card.desc}</p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">{card.size}</span>
                <div className="flex gap-2">
                  {card.title === "Interactive Backtrack Map" && (
                    <button
                      onClick={() => setShowBacktrackMap(!showBacktrackMap)}
                      className="flex items-center gap-1 px-2.5 py-1.5 bg-muted/50 rounded-lg text-xs font-medium hover:bg-muted transition-colors"
                    >
                      <Eye className="w-3.5 h-3.5" />
                      Preview
                    </button>
                  )}
                  {card.title === "CSV Summary" && (
                    <button
                      onClick={() => setShowCsv(!showCsv)}
                      className="flex items-center gap-1 px-2.5 py-1.5 bg-muted/50 rounded-lg text-xs font-medium hover:bg-muted transition-colors"
                    >
                      <Eye className="w-3.5 h-3.5" />
                      Preview
                    </button>
                  )}
                  <a
                    href={card.file}
                    download
                    className="flex items-center gap-1 px-2.5 py-1.5 bg-primary/15 text-primary rounded-lg text-xs font-medium hover:bg-primary/25 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download
                  </a>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Backtrack map embed */}
      {showBacktrackMap && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="glass-card overflow-hidden mb-8"
        >
          <div className="px-5 py-3 border-b border-border/30 flex items-center justify-between">
            <h3 className="font-heading font-semibold text-sm">Interactive Backtrack Map</h3>
            <button onClick={() => setShowBacktrackMap(false)} className="text-xs text-muted-foreground hover:text-foreground">Close</button>
          </div>
          <iframe
            src={REPORT_FILES.backtrackMap}
            className="w-full h-[500px] border-0"
            title="Backtrack Map"
          />
        </motion.div>
      )}

      {/* CSV Preview */}
      {showCsv && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="glass-card overflow-hidden mb-8"
        >
          <div className="px-5 py-3 border-b border-border/30 flex items-center justify-between">
            <h3 className="font-heading font-semibold text-sm">CSV Preview (first 20 rows)</h3>
            <button onClick={() => setShowCsv(false)} className="text-xs text-muted-foreground hover:text-foreground">Close</button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/20 text-muted-foreground">
                  {["ID", "Lat", "Lon", "Area(m²)", "Polymer Type", "Conf.", "Source", "Score"].map((h) => (
                    <th key={h} className="text-left px-3 py-2 font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {csvData.slice(0, 20).map((row, i) => (
                  <tr key={i} className="border-b border-border/10 hover:bg-muted/10">
                    <td className="px-3 py-1.5 font-mono text-primary">{row.cluster_id}</td>
                    <td className="px-3 py-1.5 font-mono">{row.lat?.toFixed(4)}</td>
                    <td className="px-3 py-1.5 font-mono">{row.lon?.toFixed(4)}</td>
                    <td className="px-3 py-1.5">{row.area_sq_m}</td>
                    <td className="px-3 py-1.5 truncate max-w-[150px]">{row.polymer_type}</td>
                    <td className="px-3 py-1.5">{row.confidence?.toFixed(3)}</td>
                    <td className="px-3 py-1.5">{row.top_source_type || "—"}</td>
                    <td className="px-3 py-1.5">{row.attribution_score || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-5 py-2 text-xs text-muted-foreground border-t border-border/20">
            Showing 20 of {csvData.length} rows
          </div>
        </motion.div>
      )}

      {/* Generated Images */}
      <h2 className="font-heading font-semibold mb-4">Generated Images</h2>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        {imageCards.map((img, i) => (
          <motion.div
            key={img.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="glass-card overflow-hidden cursor-pointer group"
            onClick={() => setSelectedImage(selectedImage === img.file ? null : img.file)}
          >
            <div className="aspect-square bg-muted/20 flex items-center justify-center overflow-hidden">
              <img
                src={img.file}
                alt={img.title}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              />
            </div>
            <div className="p-3 flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{img.title}</p>
                <p className="text-xs text-muted-foreground">{img.size}</p>
              </div>
              <a href={img.file} download className="text-primary hover:text-primary/80">
                <Download className="w-4 h-4" />
              </a>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Expanded image modal */}
      {selectedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
          onClick={() => setSelectedImage(null)}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative max-w-6xl w-full h-full max-h-[90vh] glass-card flex items-center justify-center overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button 
              className="absolute top-4 right-4 p-2 bg-black/50 text-white hover:bg-black/80 rounded-full backdrop-blur-md transition-colors z-10"
              onClick={() => setSelectedImage(null)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
            <img src={selectedImage} alt="Preview" className="max-w-full max-h-full object-contain" />
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default ReportsTab;
