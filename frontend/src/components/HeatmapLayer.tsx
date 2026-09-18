import React, { useEffect, useRef, useState } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";

interface HeatmapLayerProps {
  points: [number, number, number][]; // [lat, lng, intensity]
  options?: {
    radius?: number;
    blur?: number;
    maxZoom?: number;
    max?: number;
    gradient?: Record<string, string>;
  };
}

const HeatmapLayer: React.FC<HeatmapLayerProps> = ({ points, options = {} }) => {
  const map = useMap();
  const layerRef = useRef<any>(null);

  useEffect(() => {
    // Dynamically import leaflet.heat to avoid SSR issues
    import("leaflet.heat" as any).then(() => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
      }

      const defaultOpts = {
        radius: 35,
        blur: 25,
        maxZoom: 14,
        max: 1.0,
        gradient: {
          0.0: "#0d47a1",
          0.25: "#00897b",
          0.5: "#f9a825",
          0.75: "#e65100",
          1.0: "#b71c1c",
        },
        ...options,
      };

      // @ts-ignore - leaflet.heat extends L
      const heatLayer = (L as any).heatLayer(points, defaultOpts);
      heatLayer.addTo(map);
      layerRef.current = heatLayer;
    }).catch(() => {
      // Fallback: if leaflet.heat is not available, render circle markers
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
      }
      const group = L.layerGroup();
      points.forEach(([lat, lng, intensity]) => {
        L.circleMarker([lat, lng], {
          radius: 8,
          fillColor: `hsl(${180 - intensity * 180}, 80%, 50%)`,
          fillOpacity: 0.6,
          color: "transparent",
        }).addTo(group);
      });
      group.addTo(map);
      layerRef.current = group;
    });

    return () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
      }
    };
  }, [points, map, options]);

  return null;
};

export default HeatmapLayer;
