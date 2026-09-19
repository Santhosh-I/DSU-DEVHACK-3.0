import React from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import {
  RunSummary,
  DebrisSummaryRow,
  PIPELINE_STAGES,
  POLYMER_COLORS,
  PipelineRun,
} from "@/types";
import {
  CheckCircle2,
  XCircle,
  SkipForward,
  MapPin,
  Calendar,
  Clock,
  Cloud,
  Cpu,
  Layers,
} from "lucide-react";

const tooltipStyle = {
  background: "hsl(220 30% 8%)",
  border: "1px solid hsl(215 20% 16%)",
  borderRadius: "8px",
  fontSize: "12px",
};

interface OverviewTabProps {
  run: PipelineRun;
  summary: RunSummary;
  csv: DebrisSummaryRow[];
}

const OverviewTab: React.FC<OverviewTabProps> = ({ run, summary, csv }) => {
  const pc = summary.outputs.polymer_counts;
  const totalDetections = Object.values(pc).reduce((a, b) => a + b, 0);
  const plasticCount = pc["Marine Debris (Plastic)"] || 0;
  const fpRate = totalDetections > 0 ? (((totalDetections - plasticCount) / totalDetections) * 100).toFixed(1) : "0.0";
  const sceneId = summary.outputs.raw_scenes?.[0] || "N/A";

  const pieData = Object.entries(pc)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({
      name: name.replace("False Positive ", "FP "),
      value,
      color: POLYMER_COLORS[name] || "#6B7280",
    }));

  return (
    <div>
      {/* Run info cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        {[
          { icon: MapPin, label: "Bbox", value: run.bbox },
          { icon: Calendar, label: "Target Date", value: run.target_date },
          { icon: Clock, label: "Duration", value: `${summary.elapsed_seconds}s` },
          { icon: Cloud, label: "Cloud Cover", value: `${run.cloud_cover}%` },
          { icon: Cpu, label: "CRS", value: "EPSG:32616" },
          { icon: Layers, label: "Resolution", value: "10 m" },
        ].map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.label} className="glass-card p-3 flex items-center gap-3">
              <Icon className="w-4 h-4 text-primary flex-shrink-0" />
              <div className="min-w-0">
                <div className="text-xs text-muted-foreground">{item.label}</div>
                <div className="text-sm font-medium truncate">{item.value}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Pipeline stages */}
      <div className="glass-card p-4 mb-6">
        <h3 className="text-sm font-heading font-semibold mb-3">Pipeline Stages</h3>
        <div className="flex flex-wrap items-center gap-2">
          {PIPELINE_STAGES.map((stage, i) => {
            const completed = summary.stages_completed.includes(stage.id);
            const failed = summary.stages_failed.includes(stage.id);
            return (
              <React.Fragment key={stage.id}>
                <div
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium ${
                    completed
                      ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                      : failed
                      ? "bg-red-500/10 text-red-400 border border-red-500/20"
                      : "bg-muted/30 text-muted-foreground border border-border/20"
                  }`}
                >
                  {completed ? <CheckCircle2 className="w-3.5 h-3.5" /> : failed ? <XCircle className="w-3.5 h-3.5" /> : <SkipForward className="w-3.5 h-3.5" />}
                  {stage.short}
                </div>
                {i < PIPELINE_STAGES.length - 1 && (
                  <div className={`w-6 h-px ${completed ? "bg-emerald-500/40" : "bg-border/40"}`} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Processing Summary */}
        <div className="glass-card p-5">
          <h3 className="font-heading font-semibold mb-4">Processing Summary</h3>
          <div className="space-y-3">
            {[
              { label: "Total Detections", value: totalDetections.toString() },
              { label: "Confirmed Plastic Clusters", value: plasticCount.toString() },
              { label: "False Positive Rate", value: `${fpRate}%` },
              { label: "CRS", value: "EPSG:32616" },
              { label: "Avg Confidence", value: csv.length > 0 ? (csv.reduce((a, r) => a + r.confidence, 0) / csv.length).toFixed(3) : "N/A" },
              ...(csv.some((r) => r.nearest_ship_name || r.nearest_ship_mmsi)
                ? [{ label: "Clusters Near AIS Ships", value: `${csv.filter((r) => r.nearest_ship_name || r.nearest_ship_mmsi).length} clusters` }]
                : []),
              { label: "Pipeline Time", value: `${summary.elapsed_seconds}s` },
            ].map((item) => (
              <div key={item.label} className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">{item.label}</span>
                <span className="font-semibold">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Polymer chart */}
        <div className="glass-card p-5">
          <h3 className="font-heading font-semibold mb-4">Polymer Distribution</h3>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={80} dataKey="value" stroke="none">
                  {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-1 mt-2 text-xs">
            {pieData.slice(0, 6).map((e) => (
              <div key={e.name} className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
                <span className="text-muted-foreground truncate">{e.name}</span>
                <span className="ml-auto font-medium">{e.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Scene Info */}
        <div className="glass-card p-5">
          <h3 className="font-heading font-semibold mb-4">Scene Information</h3>
          <div className="space-y-3 text-sm">
            <div>
              <span className="text-muted-foreground text-xs block mb-0.5">Scene ID</span>
              <span className="font-mono text-xs break-all">{sceneId}</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-muted-foreground text-xs block mb-0.5">Date</span>
                <span className="font-medium">{summary.target_date}</span>
              </div>
              <div>
                <span className="text-muted-foreground text-xs block mb-0.5">Cloud Cover</span>
                <span className="font-medium">{run.cloud_cover}%</span>
              </div>
              <div>
                <span className="text-muted-foreground text-xs block mb-0.5">CRS</span>
                <span className="font-medium">EPSG:32616</span>
              </div>
              <div>
                <span className="text-muted-foreground text-xs block mb-0.5">Resolution</span>
                <span className="font-medium">10 m</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OverviewTab;
