import React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Waves,
  Satellite,
  FlaskConical,
  GitBranch,
  BarChart3,
  ChevronRight,
  Sparkles,
  Shield,
  Zap,
  Globe,
} from "lucide-react";

const features = [
  {
    icon: Satellite,
    title: "AI Detection",
    desc: "SegFormer deep learning model detects marine plastic debris in Sentinel-2 satellite imagery at 10m resolution.",
  },
  {
    icon: FlaskConical,
    title: "Polymer Identification",
    desc: "XGBoost spectral fingerprinting classifies debris by polymer type using multispectral indices (PI, SR, NSI, FDI).",
  },
  {
    icon: GitBranch,
    title: "Source Attribution",
    desc: "Lagrangian particle tracking with CMEMS ocean currents & ERA5 wind data traces debris back to its origin.",
  },
  {
    icon: BarChart3,
    title: "Real-time Insights",
    desc: "Interactive GIS dashboard with detection maps, attribution charts, and downloadable PDF/GeoJSON/CSV reports.",
  },
];

const stats = [
  { value: "10m", label: "Resolution" },
  { value: "7-Day", label: "Backtracking" },
  { value: "44", label: "Clusters Detected" },
  { value: "61.5s", label: "Pipeline Speed" },
];

const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-background overflow-hidden">
      {/* Hero */}
      <section className="relative min-h-screen flex items-center justify-center pt-14">
        {/* Animated background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full bg-primary/5 blur-[120px] animate-pulse-glow" />
          <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />
          {/* Curved glow line */}
          <svg className="absolute top-1/3 left-0 right-0 w-full opacity-20" viewBox="0 0 1440 200" fill="none">
            <path d="M0 100 Q360 10 720 100 T1440 100" stroke="hsl(var(--primary))" strokeWidth="1.5" fill="none" />
          </svg>
        </div>

        <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass mb-8"
          >
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-muted-foreground">
              AI-Powered Satellite Analysis
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="font-heading text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight mb-6"
          >
            <span className="text-foreground">Detect </span>
            <span className="text-primary text-glow">Marine Plastic</span>
            <br />
            <span className="text-foreground">From </span>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">
              Space
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            AI-powered marine plastic pollution monitoring & source attribution from Sentinel-2 satellite imagery. 
            From space to source, empowering a cleaner future for our oceans.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-wrap items-center justify-center gap-4"
          >
            <Link
              to="/tracking"
              className="flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-xl font-semibold hover:bg-primary/90 transition-all hover:shadow-[0_0_30px_hsl(var(--primary)/0.3)]"
            >
              Start Tracking
              <ChevronRight className="w-4 h-4" />
            </Link>
            <Link
              to="/dashboard"
              className="flex items-center gap-2 px-6 py-3 glass rounded-xl font-semibold text-foreground hover:bg-muted/50 transition-all"
            >
              View Dashboard
            </Link>
          </motion.div>

          {/* Stats row */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="flex flex-wrap justify-center gap-8 mt-16"
          >
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-2xl font-heading font-bold text-primary">{stat.value}</div>
                <div className="text-xs text-muted-foreground mt-1">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="font-heading text-3xl sm:text-4xl font-bold mb-4">
              How It Works
            </h2>
            <p className="text-muted-foreground max-w-xl mx-auto">
              A 7-stage pipeline from satellite imagery to source attribution
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((f, i) => {
              const Icon = f.icon;
              return (
                <motion.div
                  key={f.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="glass-card p-6 hover:border-primary/30 transition-all group"
                >
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="font-heading font-semibold text-lg mb-2">{f.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="glass-card p-12 relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-secondary/5" />
            <div className="relative z-10">
              <Globe className="w-12 h-12 text-primary mx-auto mb-6" />
              <h2 className="font-heading text-3xl font-bold mb-4">
                Ready to monitor the oceans?
              </h2>
              <p className="text-muted-foreground mb-8 max-w-lg mx-auto">
                Select a region on the map, let the AI pipeline analyze satellite imagery,
                and get detailed reports on marine plastic pollution.
              </p>
              <Link
                to="/tracking"
                className="inline-flex items-center gap-2 px-8 py-3 bg-primary text-primary-foreground rounded-xl font-semibold hover:bg-primary/90 transition-all"
              >
                Get Started
                <ChevronRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/30 py-8 px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Waves className="w-4 h-4 text-primary" />
            <span>Plastic-Ledger</span>
          </div>
          <div className="flex items-center gap-1">
            <Shield className="w-3 h-3" />
            <span>Research & Educational Use</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
