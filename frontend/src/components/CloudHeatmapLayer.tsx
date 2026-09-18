import React, { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";

export interface HeatmapPoint {
  lat: number;
  lng: number;
  intensity: number; // 0 to 1
}

interface CloudHeatmapLayerProps {
  points: HeatmapPoint[];
  radius?: number;
  blur?: number;
  opacity?: number;
  gradient?: Record<number, string>;
}

const DEFAULT_GRADIENT: Record<number, string> = {
  0.12: "#0284c7", // deep sky blue
  0.30: "#06b6d4", // vibrant cyan
  0.50: "#10b981", // emerald green
  0.70: "#facc15", // bright yellow
  0.86: "#f97316", // vivid orange
  1.0: "#ef4444",  // intense red core
};

/** Pre-generate a 256-color lookup table from gradient stops */
function createGradientPalette(stops: Record<number, string>): Uint8ClampedArray {
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new Uint8ClampedArray(1024);

  const grad = ctx.createLinearGradient(0, 0, 0, 256);
  Object.entries(stops).forEach(([offset, color]) => {
    grad.addColorStop(parseFloat(offset), color);
  });

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 1, 256);
  return ctx.getImageData(0, 0, 1, 256).data;
}

/** Pre-render a soft Gaussian radial gradient brush for seamless cloud blending */
function createCloudBrush(radius: number, blur: number): HTMLCanvasElement {
  const r = radius + blur;
  const canvas = document.createElement("canvas");
  canvas.width = r * 2;
  canvas.height = r * 2;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  // Multi-stop radial gradient for soft atmospheric / oceanic cloud dispersion
  const grad = ctx.createRadialGradient(r, r, 0, r, r, r);
  grad.addColorStop(0, "rgba(0, 0, 0, 1.0)");
  grad.addColorStop(0.2, "rgba(0, 0, 0, 0.85)");
  grad.addColorStop(0.45, "rgba(0, 0, 0, 0.45)");
  grad.addColorStop(0.75, "rgba(0, 0, 0, 0.15)");
  grad.addColorStop(1, "rgba(0, 0, 0, 0.0)");

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(r, r, r, 0, Math.PI * 2);
  ctx.fill();

  return canvas;
}

export const CloudHeatmapLayer: React.FC<CloudHeatmapLayerProps> = ({
  points,
  radius = 32,
  blur = 24,
  opacity = 0.85,
  gradient = DEFAULT_GRADIENT,
}) => {
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    // 1. Create and attach canvas to Leaflet's overlay pane
    const canvas = L.DomUtil.create("canvas", "leaflet-cloud-heatmap") as HTMLCanvasElement;
    canvas.style.position = "absolute";
    canvas.style.pointerEvents = "none";
    canvas.style.opacity = `${opacity}`;
    canvas.style.transition = "opacity 0.2s ease";
    canvas.style.zIndex = "400";
    canvasRef.current = canvas;

    const pane = map.getPanes().overlayPane;
    pane.appendChild(canvas);

    const palette = createGradientPalette(gradient);
    const brush = createCloudBrush(radius, blur);
    const brushR = radius + blur;

    const redraw = () => {
      if (!canvas || !map) return;

      const size = map.getSize();
      if (size.x <= 0 || size.y <= 0) return;

      // Position the canvas relative to map top-left
      const topLeft = map.containerPointToLayerPoint([0, 0]);
      L.DomUtil.setPosition(canvas, topLeft);

      // Match canvas pixel dimensions to viewport
      if (canvas.width !== size.x || canvas.height !== size.y) {
        canvas.width = size.x;
        canvas.height = size.y;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (points.length === 0) return;

      // Draw soft black radial alpha stamps for all points
      points.forEach((pt) => {
        const latLng = L.latLng(pt.lat, pt.lng);
        const pixelPoint = map.latLngToContainerPoint(latLng);

        // Skip if outside view bounds
        if (
          pixelPoint.x < -brushR * 2 ||
          pixelPoint.x > size.x + brushR * 2 ||
          pixelPoint.y < -brushR * 2 ||
          pixelPoint.y > size.y + brushR * 2
        ) {
          return;
        }

        // Intensity scales alpha stamp; higher concentration creates denser cloud core
        ctx.globalAlpha = Math.max(0.08, Math.min(1.0, pt.intensity * 0.9));
        ctx.drawImage(brush, pixelPoint.x - brushR, pixelPoint.y - brushR);
      });

      // Colorize the alpha channel through the gradient palette
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;
      const len = data.length;

      for (let i = 0; i < len; i += 4) {
        const alpha = data[i + 3];
        if (alpha > 0) {
          // Color lookup based on alpha density
          const offset = alpha * 4;
          data[i] = palette[offset];         // R
          data[i + 1] = palette[offset + 1]; // G
          data[i + 2] = palette[offset + 2]; // B
          // Soft non-linear alpha curve for cloud transparency
          data[i + 3] = Math.min(255, Math.floor(alpha * 1.1));
        }
      }

      ctx.putImageData(imgData, 0, 0);
    };

    const handleUpdate = () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = requestAnimationFrame(redraw);
    };

    // Initial render
    handleUpdate();

    // Map listeners for smooth redrawing during interaction
    map.on("viewreset move moveend zoom zoomend resize", handleUpdate);

    return () => {
      map.off("viewreset move moveend zoom zoomend resize", handleUpdate);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (canvas.parentElement) {
        canvas.parentElement.removeChild(canvas);
      }
    };
  }, [map, points, radius, blur, opacity, gradient]);

  return null;
};

export default CloudHeatmapLayer;
