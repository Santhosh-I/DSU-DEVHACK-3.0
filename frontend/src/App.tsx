import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "@/components/Navbar";
import LandingPage from "@/pages/LandingPage";
import TrackingPage from "@/pages/TrackingPage";
import DashboardPage from "@/pages/DashboardPage";
import RunDetailPage from "@/pages/RunDetailPage";
import ModelPage from "@/pages/modelpage";
import HotspotsPage from "@/pages/HotspotsPage";

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-foreground">
        <Navbar />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/tracking" element={<TrackingPage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/runs/:id" element={<RunDetailPage />} />
          <Route path="/hotspots" element={<HotspotsPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
};

export default App;
