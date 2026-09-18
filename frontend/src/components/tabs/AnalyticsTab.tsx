import React, { useEffect, useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";
import { loadRunSummary, loadDebrisSummaryCsv, loadAttribution } from "@/services/dataService";
import { RunSummary, DebrisSummaryRow, AttributionEntry, POLYMER_COLORS } from "@/types";
import { BarChart3, PieChart as PieIcon, Activity, ShieldAlert } from "lucide-react";

function ChartCard({ title, icon: Icon, children }: { title: string; icon: any; children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-5"
    >
      <h3 className="font-heading font-semibold flex items-center gap-2 mb-4">
        <Icon className="w-4 h-4 text-primary" />
        {title}
      </h3>
      {children}
    </motion.div>
  );
}

const tooltipStyle = {
  background: "hsl(220 30% 8%)",
  border: "1px solid hsl(215 20% 16%)",
  borderRadius: "8px",
  fontSize: "12px",
};

interface AnalyticsTabProps {
  runId: string;
}

const AnalyticsTab: React.FC<AnalyticsTabProps> = ({ runId }) => {
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [csv, setCsv] = useState<DebrisSummaryRow[]>([]);
  const [attribution, setAttribution] = useState<AttributionEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([loadRunSummary(runId), loadDebrisSummaryCsv(runId), loadAttribution(runId)]).then(
      ([sum, csvData, attr]) => {
        setSummary(sum);
        setCsv(csvData);
        setAttribution(attr);
        setLoading(false);
      }
    );
  }, [runId]);

  const polymerData = useMemo(() => {
    if (!summary) return [];
    return Object.entries(summary.outputs.polymer_counts)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({
        name: name.replace("False Positive ", "FP "),
        fullName: name,
        value,
        color: POLYMER_COLORS[name] || "#6B7280",
      }));
  }, [summary]);

  const confHistogram = useMemo(() => {
    const bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    return bins.slice(0, -1).map((min, i) => ({
      range: `${(min).toFixed(1)}-${bins[i + 1].toFixed(1)}`,
      count: csv.filter((r) => r.confidence >= min && r.confidence < bins[i + 1]).length,
    }));
  }, [csv]);

  const areaHistogram = useMemo(() => {
    const bins = [0, 200, 500, 1000, 2000, 5000, 10000, 50000];
    return bins.slice(0, -1).map((min, i) => ({
      range: min >= 1000 ? `${min / 1000}k-${bins[i + 1] / 1000}k` : `${min}-${bins[i + 1]}`,
      count: csv.filter((r) => r.area_sq_m >= min && r.area_sq_m < bins[i + 1]).length,
    }));
  }, [csv]);

  const fpBreakdown = useMemo(() => {
    if (!summary) return [];
    return Object.entries(summary.outputs.polymer_counts)
      .filter(([name]) => name.startsWith("False Positive"))
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({
        name: name.replace("False Positive (", "").replace(")", ""),
        value,
        color: POLYMER_COLORS[name] || "#6B7280",
      }));
  }, [summary]);

  const tpVsFp = useMemo(() => {
    if (!summary) return [];
    const pc = summary.outputs.polymer_counts;
    const plastic = pc["Marine Debris (Plastic)"] || 0;
    const organic = pc["Organic Matter (Foam)"] || 0;
    const total = Object.values(pc).reduce((a, b) => a + b, 0);
    return [
      { name: "Marine Debris", value: plastic, color: "#EF4444" },
      { name: "Organic", value: organic, color: "#10B981" },
      { name: "False Positive", value: total - plastic - organic, color: "#6B7280" },
    ];
  }, [summary]);

  const sourceBreakdown = useMemo(() => {
    const types: Record<string, number> = {};
    attribution.forEach((a) => {
      types[a.source_type] = (types[a.source_type] || 0) + 1;
    });
    return Object.entries(types).map(([name, value]) => ({
      name,
      value,
      color: name === "fishing" ? "#3B82F6" : name === "industrial" ? "#F59E0B" : name === "shipping" ? "#8B5CF6" : "#10B981",
    }));
  }, [attribution]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
      {/* Polymer Distribution */}
      <ChartCard title="Polymer Distribution" icon={PieIcon}>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={polymerData} cx="50%" cy="50%" innerRadius={50} outerRadius={85} dataKey="value" stroke="none">
                {polymerData.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-1 mt-2 max-h-[140px] overflow-y-auto text-xs">
          {polymerData.map((e) => (
            <div key={e.name} className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
                <span className="text-muted-foreground truncate max-w-[150px]">{e.name}</span>
              </div>
              <span className="font-medium">{e.value}</span>
            </div>
          ))}
        </div>
      </ChartCard>

      {/* Detection Confidence */}
      <ChartCard title="Detection Confidence" icon={Activity}>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={confHistogram}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 20% 16%)" />
              <XAxis dataKey="range" tick={{ fill: "#6B7280", fontSize: 10 }} angle={-30} textAnchor="end" height={50} />
              <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Area Distribution */}
      <ChartCard title="Cluster Area Distribution (m²)" icon={BarChart3}>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={areaHistogram}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 20% 16%)" />
              <XAxis dataKey="range" tick={{ fill: "#6B7280", fontSize: 10 }} angle={-30} textAnchor="end" height={50} />
              <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" fill="hsl(var(--secondary))" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* True vs False Positive */}
      <ChartCard title="True vs False Positive" icon={ShieldAlert}>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={tpVsFp} cx="50%" cy="50%" innerRadius={50} outerRadius={85} dataKey="value" stroke="none">
                {tpVsFp.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-1.5 mt-2 text-xs">
          {tpVsFp.map((e) => (
            <div key={e.name} className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: e.color }} />
                <span className="text-muted-foreground">{e.name}</span>
              </div>
              <span className="font-semibold">{e.value}</span>
            </div>
          ))}
        </div>
      </ChartCard>

      {/* Source Type Breakdown */}
      <ChartCard title="Source Type Breakdown" icon={PieIcon}>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={sourceBreakdown} cx="50%" cy="50%" innerRadius={50} outerRadius={85} dataKey="value" stroke="none">
                {sourceBreakdown.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-1.5 mt-2 text-xs">
          {sourceBreakdown.map((e) => (
            <div key={e.name} className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: e.color }} />
                <span className="text-muted-foreground capitalize">{e.name}</span>
              </div>
              <span className="font-semibold">{e.value}</span>
            </div>
          ))}
        </div>
      </ChartCard>

      {/* FP Sub-type */}
      <ChartCard title="False Positive Analysis" icon={ShieldAlert}>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={fpBreakdown} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 20% 16%)" />
              <XAxis type="number" tick={{ fill: "#6B7280", fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: "#9CA3AF", fontSize: 11 }} width={80} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {fpBreakdown.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </div>
  );
};

export default AnalyticsTab;
