import React, { useEffect, useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts";
import { loadRunSummary, loadDebrisSummaryCsv, loadAttribution } from "@/services/dataService";
import { RunSummary, DebrisSummaryRow, AttributionEntry, POLYMER_COLORS } from "@/types";
import { 
  Crosshair, FlaskConical, ShieldAlert, Activity, 
  Layers, Map, Info, AlertTriangle, Ship, Anchor, Target
} from "lucide-react";

function KpiCard({ label, value, subtext, icon: Icon, alert = false }: any) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-5 rounded-xl border ${alert ? 'bg-destructive/5 border-destructive/20' : 'glass-card'}`}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">{label}</span>
        <div className={`p-2 rounded-lg ${alert ? 'bg-destructive/10' : 'bg-primary/10'}`}>
          <Icon className={`w-4 h-4 ${alert ? 'text-destructive' : 'text-primary'}`} />
        </div>
      </div>
      <div>
        <div className={`text-3xl font-bold font-heading ${alert ? 'text-destructive' : 'text-foreground'}`}>
          {value}
        </div>
        {subtext && <div className="text-xs text-muted-foreground mt-1">{subtext}</div>}
      </div>
    </motion.div>
  );
}

function StatBar({ label, value, total, color }: { label: string; value: number; total: number; color: string }) {
  const percent = total > 0 ? (value / total) * 100 : 0;
  return (
    <div className="mb-3.5 last:mb-0">
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-muted-foreground font-medium">{label}</span>
        <span className="font-semibold">{value} <span className="text-muted-foreground font-normal ml-1">({percent.toFixed(1)}%)</span></span>
      </div>
      <div className="h-1.5 w-full bg-muted/20 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-1000 ease-out" style={{ width: `${percent}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

const tooltipStyle = {
  background: "hsl(220 30% 8%)",
  border: "1px solid hsl(215 20% 16%)",
  borderRadius: "8px",
  fontSize: "12px",
  boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.5)",
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

  // Derived Metrics
  const metrics = useMemo(() => {
    if (!summary) return null;
    const pc = summary.outputs.polymer_counts || {};
    
    let totalDetections = 0;
    let totalFP = 0;
    Object.entries(pc).forEach(([name, count]) => {
      totalDetections += count;
      if (name.startsWith("False Positive")) totalFP += count;
    });

    const totalPlastic = pc["Marine Debris (Plastic)"] || 0;
    const totalOrganic = pc["Organic Matter (Foam)"] || 0;
    
    const avgConf = csv.length > 0 
      ? (csv.reduce((acc, row) => acc + row.confidence, 0) / csv.length) 
      : 0;

    const totalArea = csv.reduce((acc, row) => acc + row.area_sq_m, 0);

    return {
      totalDetections,
      totalPlastic,
      totalOrganic,
      totalFP,
      fpRate: totalDetections > 0 ? (totalFP / totalDetections) * 100 : 0,
      avgConfidence: avgConf,
      totalClusters: csv.length,
      totalArea,
    };
  }, [summary, csv]);

  // Chart Data
  const confHistogram = useMemo(() => {
    const bins = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    return bins.slice(0, -1).map((min, i) => ({
      range: `${(min).toFixed(1)}-${bins[i + 1].toFixed(1)}`,
      count: csv.filter((r) => r.confidence >= min && r.confidence < bins[i + 1]).length,
    }));
  }, [csv]);

  const areaHistogram = useMemo(() => {
    const bins = [0, 100, 500, 1000, 5000, 10000, 50000];
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

  const sourceStats = useMemo(() => {
    const types: Record<string, number> = {};
    attribution.forEach((a) => {
      types[a.source_type] = (types[a.source_type] || 0) + 1;
    });
    return Object.entries(types).sort((a, b) => b[1] - a[1]).map(([name, value]) => ({
      name,
      value,
      color: name === "fishing" ? "#3B82F6" : name === "industrial" ? "#F59E0B" : name === "shipping" ? "#8B5CF6" : "#10B981",
    }));
  }, [attribution]);

  if (loading || !metrics) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-10">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-heading font-bold text-foreground">Run Analytics</h2>
          <p className="text-sm text-muted-foreground mt-1">High-level insights and detection quality metrics.</p>
        </div>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
        <KpiCard 
          label="Total Detections" 
          value={metrics.totalDetections} 
          icon={Crosshair} 
        />
        <KpiCard 
          label="Confirmed Plastic" 
          value={metrics.totalPlastic} 
          subtext={`${((metrics.totalPlastic / (metrics.totalDetections || 1)) * 100).toFixed(1)}% of total`}
          icon={FlaskConical} 
        />
        <KpiCard 
          label="Avg Confidence" 
          value={`${(metrics.avgConfidence * 100).toFixed(1)}%`} 
          icon={Activity} 
        />
        <KpiCard 
          label="Total Area (m²)" 
          value={metrics.totalArea > 1000 ? `${(metrics.totalArea / 1000).toFixed(1)}k` : Math.round(metrics.totalArea)} 
          subtext={`${metrics.totalClusters} clusters detected`}
          icon={Map} 
        />
        <KpiCard 
          label="False Positives" 
          value={metrics.totalFP} 
          subtext={`${metrics.fpRate.toFixed(1)}% FP rate`}
          icon={ShieldAlert} 
          alert={metrics.fpRate > 40}
        />
      </div>

      {/* Main Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Visualizations */}
        <div className="lg:col-span-8 space-y-6">
          <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card p-6 border border-border/30">
            <h3 className="font-heading font-semibold flex items-center gap-2 mb-6">
              <Target className="w-4 h-4 text-primary" />
              Detection Confidence Distribution
            </h3>
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confHistogram} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(215 20% 16%)" />
                  <XAxis dataKey="range" tick={{ fill: "#6B7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'hsl(var(--muted)/0.2)' }} />
                  <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} maxBarSize={50} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card p-6 border border-border/30">
            <h3 className="font-heading font-semibold flex items-center gap-2 mb-6">
              <Layers className="w-4 h-4 text-primary" />
              Cluster Size Distribution
            </h3>
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={areaHistogram} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(215 20% 16%)" />
                  <XAxis dataKey="range" tick={{ fill: "#6B7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'hsl(var(--muted)/0.2)' }} />
                  <Bar dataKey="count" fill="hsl(var(--secondary))" radius={[4, 4, 0, 0]} maxBarSize={50} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        </div>

        {/* Right Column: Breakdown Texts & FP Analysis */}
        <div className="lg:col-span-4 space-y-6">
          
          <motion.div initial={{ opacity: 0, x: 15 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="glass-card p-6 border border-border/30">
            <h3 className="font-heading font-semibold flex items-center gap-2 mb-5">
              <Info className="w-4 h-4 text-primary" />
              Detection Quality
            </h3>
            <div className="space-y-4">
              <StatBar 
                label="Marine Debris (Plastic)" 
                value={metrics.totalPlastic} 
                total={metrics.totalDetections} 
                color="#EF4444" 
              />
              <StatBar 
                label="Organic Matter (Foam)" 
                value={metrics.totalOrganic} 
                total={metrics.totalDetections} 
                color="#10B981" 
              />
              <StatBar 
                label="False Positives" 
                value={metrics.totalFP} 
                total={metrics.totalDetections} 
                color="#6B7280" 
              />
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, x: 15 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="glass-card p-6 border border-border/30">
            <h3 className="font-heading font-semibold flex items-center gap-2 mb-5">
              <AlertTriangle className="w-4 h-4 text-destructive" />
              False Positive Breakdown
            </h3>
            {fpBreakdown.length > 0 ? (
              <div className="h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={fpBreakdown} layout="vertical" margin={{ top: 0, right: 0, left: 10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(215 20% 16%)" />
                    <XAxis type="number" tick={{ fill: "#6B7280", fontSize: 11 }} axisLine={false} tickLine={false} hide />
                    <YAxis type="category" dataKey="name" tick={{ fill: "#9CA3AF", fontSize: 11 }} width={85} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'hsl(var(--muted)/0.1)' }} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                      {fpBreakdown.map((e, i) => <Cell key={i} fill={e.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-[220px] flex items-center justify-center text-muted-foreground text-sm border border-dashed border-border/50 rounded-lg">
                No false positives recorded.
              </div>
            )}
          </motion.div>

          {sourceStats.length > 0 && (
            <motion.div initial={{ opacity: 0, x: 15 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }} className="glass-card p-6 border border-border/30">
              <h3 className="font-heading font-semibold flex items-center gap-2 mb-5">
                <Ship className="w-4 h-4 text-primary" />
                Attribution Sources
              </h3>
              <div className="space-y-4">
                {sourceStats.map(stat => (
                  <StatBar 
                    key={stat.name}
                    label={<span className="capitalize">{stat.name}</span> as any}
                    value={stat.value} 
                    total={attribution.length} 
                    color={stat.color} 
                  />
                ))}
              </div>
            </motion.div>
          )}

        </div>
      </div>
    </div>
  );
};

export default AnalyticsTab;
