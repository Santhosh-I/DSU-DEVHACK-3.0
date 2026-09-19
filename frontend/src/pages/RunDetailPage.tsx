import React, { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  loadRunSummary,
  loadFinalReport,
  loadAttribution,
  loadRunMetadata,
  loadDebrisSummaryCsv,
} from "@/services/dataService";
import {
  RunSummary,
  DetectionFeatureCollection,
  AttributionEntry,
  RunMetadata,
  DebrisSummaryRow,
  PipelineRun,
} from "@/types";
import { getPipelineRun } from "@/lib/api";
import {
  ClipboardList,
  CheckCircle2,
  Clock,
  Map,
  GitBranch,
  BarChart3,
  FileText,
  Database,
  ArrowLeft,
  XCircle,
} from "lucide-react";

// Tab components
import OverviewTab from "@/components/tabs/OverviewTab";
import DetectionTab from "@/components/tabs/DetectionTab";
import AttributionTab from "@/components/tabs/AttributionTab";
import AnalyticsTab from "@/components/tabs/AnalyticsTab";
import ReportsTab from "@/components/tabs/ReportsTab";
import ErrorBoundary from "@/components/ErrorBoundary";

const TABS = [
  { key: "overview", label: "Overview", icon: ClipboardList },
  { key: "detection", label: "Detection Map", icon: Map },
  { key: "attribution", label: "Attribution", icon: GitBranch },
  { key: "analytics", label: "Analytics", icon: BarChart3 },
  { key: "reports", label: "Reports", icon: FileText },
  { key: "raw", label: "Raw Data", icon: Database },
];

const RunDetailPage: React.FC = () => {
  const { id: runId } = useParams<{ id: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = searchParams.get("tab") || "overview";

  const [run, setRun] = useState<PipelineRun | null>(null);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [csv, setCsv] = useState<DebrisSummaryRow[]>([]);
  const [loading, setLoading] = useState(true);

  const setTab = (tab: string) => {
    setSearchParams({ tab });
  };

  useEffect(() => {
    if (!runId) return;
    let interval: ReturnType<typeof setInterval>;

    const loadStatusAndData = async () => {
      try {
        const runData = await getPipelineRun(runId);
        setRun(runData);

        if (runData.status === "COMPLETED") {
          const [sum, meta, csvData] = await Promise.all([
            loadRunSummary(runId),
            loadRunMetadata(runId),
            loadDebrisSummaryCsv(runId),
          ]);
          setSummary(sum);
          setMetadata(meta);
          setCsv(csvData);
          setLoading(false);
          clearInterval(interval);
        } else if (runData.status === "FAILED") {
          setLoading(false);
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Error loading run data:", err);
      }
    };

    loadStatusAndData();
    interval = setInterval(loadStatusAndData, 5000);

    return () => clearInterval(interval);
  }, [runId]);

  if (!runId) return null;

  if (loading || !run) {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-muted-foreground">Loading Run Data...</p>
        </div>
      </div>
    );
  }

  if (run.status === "RUNNING" || run.status === "PENDING") {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="glass-card p-8 max-w-md w-full text-center">
          <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <h2 className="text-xl font-heading font-bold mb-2">Pipeline {run.status}</h2>
          <p className="text-muted-foreground text-sm mb-6">
            The pipeline is still processing this run. This page will auto-refresh.
          </p>
          <Link to="/dashboard" className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-semibold inline-flex items-center gap-2">
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (run.status === "FAILED") {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="glass-card p-8 max-w-md w-full text-center">
          <XCircle className="w-12 h-12 text-destructive mx-auto mb-4" />
          <h2 className="text-xl font-heading font-bold mb-2">Run Failed</h2>
          <p className="text-muted-foreground text-sm mb-4">{run.error_message || "An unknown error occurred."}</p>
          <Link to="/dashboard" className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-semibold inline-flex items-center gap-2">
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (!summary) return null;

  const displayName = run.run_name || runId.substring(0, 8);

  return (
    <div className="min-h-screen bg-background pt-14">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6 py-6">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div className="flex items-center gap-3">
              <Link
                to="/dashboard"
                className="p-2 glass rounded-lg text-muted-foreground hover:text-foreground transition-colors"
                title="Back to Dashboard"
              >
                <ArrowLeft className="w-4 h-4" />
              </Link>
              <div>
                <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
                  <ClipboardList className="w-6 h-6 text-primary" />
                  {displayName}
                </h1>
                <p className="text-xs text-muted-foreground mt-0.5 font-mono">{runId}</p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/15 text-emerald-400 rounded-full text-sm font-medium self-start">
              <CheckCircle2 className="w-4 h-4" />
              Completed
            </span>
          </div>
        </motion.div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6 overflow-x-auto scrollbar-hide border-b border-border/20 pb-px">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.key}
                onClick={() => setTab(tab.key)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium whitespace-nowrap transition-all border-b-2 -mb-px ${
                  activeTab === tab.key
                    ? "text-primary border-primary"
                    : "text-muted-foreground border-transparent hover:text-foreground hover:border-border/50"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <ErrorBoundary fallbackTitle={`Error in ${activeTab} tab`}>
          <motion.div key={activeTab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            {activeTab === "overview" && (
              <OverviewTab run={run} summary={summary} csv={csv} />
            )}

            {activeTab === "detection" && (
              <DetectionTab runId={runId} />
            )}

            {activeTab === "attribution" && (
              <AttributionTab runId={runId} />
            )}

            {activeTab === "analytics" && (
              <AnalyticsTab runId={runId} />
            )}

            {activeTab === "reports" && (
              <ReportsTab runId={runId} />
            )}

          {activeTab === "raw" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass-card p-5">
                <h3 className="font-heading font-semibold mb-3">run_summary.json</h3>
                <pre className="text-xs font-mono text-muted-foreground bg-muted/20 rounded-lg p-4 overflow-auto max-h-[400px]">
                  {JSON.stringify(summary, null, 2)}
                </pre>
              </div>
              {metadata && (
                <div className="glass-card p-5">
                  <h3 className="font-heading font-semibold mb-3">run_metadata.json (attribution)</h3>
                  <pre className="text-xs font-mono text-muted-foreground bg-muted/20 rounded-lg p-4 overflow-auto max-h-[400px]">
                    {JSON.stringify(metadata, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
          </motion.div>
        </ErrorBoundary>
      </div>
    </div>
  );
};

export default RunDetailPage;
