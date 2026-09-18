import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { getPipelineRuns } from "@/lib/api";
import { RunSummary, PipelineRun } from "@/types";
import {
  Crosshair,
  FlaskConical,
  ShieldAlert,
  Clock,
  Cloud,
  Undo2,
  CheckCircle2,
  XCircle,
  Map,
  GitBranch,
  BarChart3,
  FileText,
  ClipboardList,
  LayoutDashboard,
} from "lucide-react";

function KpiCard({ label, value, icon: Icon, color }: { label: string; value: string; icon: any; color: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-5 flex items-center gap-4"
    >
      <div className={`w-11 h-11 rounded-xl flex items-center justify-center ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-2xl font-heading font-bold text-foreground">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </motion.div>
  );
}

function ActionButton({ to, icon: Icon, label, color }: { to: string; icon: any; label: string; color: string }) {
  return (
    <Link
      to={to}
      className={`p-1.5 rounded-lg transition-all hover:scale-110 ${color}`}
      title={label}
    >
      <Icon className="w-4 h-4" />
    </Link>
  );
}

const DashboardPage: React.FC = () => {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchRuns = async () => {
    try {
      const data = await getPipelineRuns();
      setRuns(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
    const interval = setInterval(() => {
      setRuns((currentRuns) => {
        const hasActive = currentRuns.some(
          (r) => r.status === "PENDING" || r.status === "RUNNING"
        );
        if (hasActive) {
          fetchRuns();
        }
        return currentRuns;
      });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Aggregate KPIs across all completed runs
  const completedRuns = runs.filter((r) => r.status === "COMPLETED" && r.summary);
  let totalDetections = 0;
  let totalPlastic = 0;

  completedRuns.forEach((r) => {
    const pc = r.summary!.outputs?.polymer_counts || {};
    totalDetections += Object.values(pc).reduce((a, b) => a + b, 0);
    totalPlastic += pc["Marine Debris (Plastic)"] || 0;
  });

  const fpRate = totalDetections > 0
    ? (((totalDetections - totalPlastic) / totalDetections) * 100).toFixed(1)
    : "0.0";

  const latestCompleted = completedRuns[0];

  return (
    <div className="min-h-screen bg-background pt-14">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6 py-6">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
            <LayoutDashboard className="w-6 h-6 text-primary" />
            Dashboard
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Overview of all pipeline runs and aggregate statistics.
          </p>
        </motion.div>

        {/* KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          <KpiCard label="Total Detections" value={totalDetections.toString()} icon={Crosshair} color="bg-secondary/20 text-secondary" />
          <KpiCard label="Confirmed Plastic" value={totalPlastic.toString()} icon={FlaskConical} color="bg-destructive/20 text-destructive" />
          <KpiCard label="False Positive Rate" value={`${fpRate}%`} icon={ShieldAlert} color="bg-yellow-500/20 text-yellow-400" />
          <KpiCard label="Completed Runs" value={completedRuns.length.toString()} icon={CheckCircle2} color="bg-emerald-500/20 text-emerald-400" />
          <KpiCard label="Cloud Cover" value={latestCompleted ? `${latestCompleted.cloud_cover}%` : "—"} icon={Cloud} color="bg-blue-400/20 text-blue-400" />
          <KpiCard label="Backtrack Days" value={latestCompleted ? latestCompleted.backtrack_days.toString() : "—"} icon={Undo2} color="bg-purple-400/20 text-purple-400" />
        </div>

        {/* Runs Table */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card overflow-hidden"
        >
          <div className="px-5 py-4 border-b border-border/30 flex items-center justify-between">
            <h2 className="font-heading font-semibold text-lg">Pipeline Runs</h2>
            <div className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground">{runs.length} run(s)</span>
              <Link
                to="/tracking"
                className="px-3 py-1.5 bg-primary text-primary-foreground rounded-lg text-xs font-semibold hover:bg-primary/90 transition-colors"
              >
                + New Run
              </Link>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/20 text-muted-foreground">
                  <th className="text-left px-5 py-3 font-medium">Run Name</th>
                  <th className="text-left px-5 py-3 font-medium">Status</th>
                  <th className="text-left px-5 py-3 font-medium">Region</th>
                  <th className="text-left px-5 py-3 font-medium">Target Date</th>
                  <th className="text-left px-5 py-3 font-medium">Detections</th>
                  <th className="text-left px-5 py-3 font-medium">Plastic</th>
                  <th className="text-center px-5 py-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const rSummary = run.summary;
                  const rTotal = rSummary ? Object.values(rSummary.outputs?.polymer_counts || {}).reduce((a, b) => a + b, 0) : 0;
                  const rPlastic = rSummary ? (rSummary.outputs?.polymer_counts?.["Marine Debris (Plastic)"] || 0) : 0;
                  const isCompleted = run.status === "COMPLETED";
                  const displayName = run.run_name || run.id.substring(0, 8);

                  return (
                    <tr key={run.id} className="border-b border-border/10 hover:bg-muted/20 transition-colors">
                      <td className="px-5 py-3">
                        <Link to={`/runs/${run.id}`} className="text-primary hover:text-primary/80 font-medium transition-colors">
                          {displayName}
                        </Link>
                        <div className="text-[10px] text-muted-foreground font-mono mt-0.5">{run.id.substring(0, 8)}</div>
                      </td>
                      <td className="px-5 py-3">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                          run.status === 'COMPLETED' ? "bg-emerald-500/15 text-emerald-400" :
                          run.status === 'FAILED' ? "bg-destructive/15 text-destructive" :
                          run.status === 'RUNNING' ? "bg-blue-500/15 text-blue-400" :
                          "bg-yellow-500/15 text-yellow-400"
                        }`}>
                          {run.status === 'COMPLETED' ? <CheckCircle2 className="w-3 h-3" /> :
                           run.status === 'FAILED' ? <XCircle className="w-3 h-3" /> :
                           <Clock className="w-3 h-3 animate-pulse" />}
                          {run.status}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-muted-foreground text-xs max-w-[180px] truncate" title={run.bbox}>{run.bbox}</td>
                      <td className="px-5 py-3 text-muted-foreground">{run.target_date}</td>
                      <td className="px-5 py-3 font-semibold">{isCompleted ? rTotal : "—"}</td>
                      <td className="px-5 py-3 font-semibold text-destructive">{isCompleted ? rPlastic : "—"}</td>
                      <td className="px-5 py-3">
                        {isCompleted ? (
                          <div className="flex items-center justify-center gap-1.5">
                            <ActionButton
                              to={`/runs/${run.id}`}
                              icon={ClipboardList}
                              label="Run Details"
                              color="text-foreground hover:bg-muted/50"
                            />
                            <ActionButton
                              to={`/runs/${run.id}?tab=detection`}
                              icon={Map}
                              label="Detection Map"
                              color="text-blue-400 hover:bg-blue-500/15"
                            />
                            <ActionButton
                              to={`/runs/${run.id}?tab=attribution`}
                              icon={GitBranch}
                              label="Attribution"
                              color="text-emerald-400 hover:bg-emerald-500/15"
                            />
                            <ActionButton
                              to={`/runs/${run.id}?tab=analytics`}
                              icon={BarChart3}
                              label="Analytics"
                              color="text-purple-400 hover:bg-purple-500/15"
                            />
                            <ActionButton
                              to={`/runs/${run.id}?tab=reports`}
                              icon={FileText}
                              label="Reports"
                              color="text-yellow-400 hover:bg-yellow-500/15"
                            />
                          </div>
                        ) : (
                          <div className="flex items-center justify-center">
                            <Link
                              to={`/runs/${run.id}`}
                              className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-all"
                              title="View Run"
                            >
                              <ClipboardList className="w-4 h-4" />
                            </Link>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
                {runs.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-5 py-12 text-center text-muted-foreground">
                      <p className="mb-2">No pipeline runs yet.</p>
                      <Link to="/tracking" className="text-primary hover:text-primary/80 font-medium">
                        Create your first run →
                      </Link>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default DashboardPage;
