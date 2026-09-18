"""
Plastic-Ledger — Stage 3: Marine Debris Detection
====================================================
Runs trained segmentation models (SegFormer and U-Net variants) on
preprocessed patches and produces georeferenced debris detections with
TTA and clustering.

Usage (standalone):
    python -m pipeline.03_detect \\
        --scene_id SCENE_ID \\
        --patches_dir data/processed/SCENE_ID/patches \\
        --model_path d:/Plastic-Ledger/models/segformer/runs/segformer_v1/best_model_SegTransformer.pth \\
        --output_dir data/detections

Dependencies: torch, transformers, rasterio, scipy, geopandas, numpy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import label as ndimage_label, find_objects as ndimage_find_objects
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.geo_utils import save_geotiff, array_to_polygons
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 11
NUM_BANDS   = 11
DEBRIS_CLASS_INDEX = 0
DEFAULT_THRESHOLD  = 0.10        # Lowered from 0.15 → improves recall for sparse debris
DEBRIS_LOGIT_BOOST = 0.5       # Added to the raw debris logit before softmax
MIN_CLUSTER_AREA_M2 = 100          # 4 pixels at 10m resolution
MAX_CLUSTER_AREA_M2 = 50_000_000   # 50 km² — anything larger is a false positive

CLASS_MAP = {
    0: "Marine Debris", 1: "Dense Sargassum", 2: "Sparse Sargassum",
    3: "Natural Organic Material", 4: "Ship", 5: "Clouds",
    6: "Marine Water", 7: "Sediment-Laden Water", 8: "Foam",
    9: "Turbid Water", 10: "Shallow Water", 11: "Waves",
    12: "Cloud Shadows", 13: "Wakes", 14: "Mixed Water",
}


def _double_conv_flat(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv-BN-ReLU x2 with index layout matching MARIDA official U-Net."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _OfficialMaridaUNet(nn.Module):
    """UNet matching the flat-key official MARIDA checkpoint structure."""

    def __init__(self, in_channels: int = NUM_BANDS, num_classes: int = NUM_CLASSES, hidden: int = 16):
        super().__init__()
        h = hidden
        self.inc = _double_conv_flat(in_channels, h)

        def _down(ic: int, oc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(ic, oc, 3, padding=1, bias=True),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=True),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        class Down(nn.Module):
            def __init__(self, ic: int, oc: int):
                super().__init__()
                self.maxpool_conv = _down(ic, oc)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.maxpool_conv(x)

        class Up(nn.Module):
            def __init__(self, ic: int, oc: int):
                super().__init__()
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv = _double_conv_flat(ic, oc)

            def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                x1 = self.up(x1)
                return self.conv(torch.cat([x2, x1], dim=1))

        self.down1 = Down(h, h * 2)
        self.down2 = Down(h * 2, h * 4)
        self.down3 = Down(h * 4, h * 8)
        self.down4 = Down(h * 8, h * 8)

        self.up1 = Up((h * 8) + (h * 8), h * 4)
        self.up2 = Up((h * 4) + (h * 4), h * 2)
        self.up3 = Up((h * 2) + (h * 2), h)
        self.up4 = Up(h + h, h)
        self.outc = nn.Conv2d(h, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def load_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    architecture: Optional[str] = None,
) -> torch.nn.Module:
    """Load a trained segmentation model checkpoint.

    Supported architectures:
      - ``segformer_b0`` / ``segformer``
      - ``unet_smp`` (segmentation_models_pytorch)
      - ``unet_official`` (flat-key official MARIDA U-Net)

    If *architecture* is omitted, it is auto-detected from checkpoint keys.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Torch device to load the model onto.

    Returns:
        Model in eval mode.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(ckpt_obj, dict):
            if "model_state" in ckpt_obj:
                return ckpt_obj["model_state"]
            if "state_dict" in ckpt_obj:
                return ckpt_obj["state_dict"]
            if "model_state_dict" in ckpt_obj:
                return ckpt_obj["model_state_dict"]
            return ckpt_obj
        return ckpt_obj

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    state_keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []

    arch = architecture
    if not arch and isinstance(checkpoint, dict):
        arch = checkpoint.get("architecture")
    if not arch and isinstance(checkpoint, dict):
        encoder_name = str(checkpoint.get("encoder", "")).strip()
        if encoder_name:
            arch = "unet_smp"
    if not arch:
        if any(k.startswith("segformer.stages.") for k in state_keys):
            # B2 pretrained structure — uses .stages not .encoder
            arch = "segformer_b2"
        elif any(k.startswith("segformer.encoder.") for k in state_keys):
            # B0 scratch structure — uses .encoder
            arch = "segformer_b0"
        elif any(k.startswith("inc.") for k in state_keys) and "outc.weight" in state_dict:
            arch = "unet_official"
        elif any(k.startswith("decoder.") for k in state_keys) or any(k.startswith("segmentation_head.") for k in state_keys):
            arch = "unet_smp"
        else:
            # Last resort: detect from classifier head hidden size
            cls_w = state_dict.get("decode_head.classifier.weight") if isinstance(state_dict, dict) else None
            if cls_w is not None and hasattr(cls_w, "shape"):
                hidden = cls_w.shape[1]
                arch = "segformer_b2" if hidden >= 512 else "segformer_b0"

    num_cls = NUM_CLASSES
    cls_w = state_dict.get("decode_head.classifier.weight") if isinstance(state_dict, dict) else None
    if cls_w is not None and hasattr(cls_w, "shape"):
        num_cls = cls_w.shape[0]
        logger.info("Detected checkpoint shape: %d output classes → %s", num_cls, arch)

    arch = str(arch).lower()
    if arch in {"segformer_b2"}:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_cls,
            ignore_mismatched_sizes=True
        )
        old_conv = model.segformer.stages[0].patch_embeddings.proj
        new_conv = nn.Conv2d(
            in_channels=NUM_BANDS, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=(old_conv.bias is not None)
        )
        model.segformer.stages[0].patch_embeddings.proj = new_conv
        model.config.num_channels = NUM_BANDS
        model.load_state_dict(state_dict, strict=False)

    elif arch in {"segformer_b0"}:
        cfg = SegformerConfig(
            num_labels=NUM_CLASSES,
            num_channels=NUM_BANDS,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            decoder_hidden_size=256,
        )
        model = SegformerForSemanticSegmentation(cfg)
        old = model.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(
            NUM_BANDS,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
        )
        model.segformer.encoder.patch_embeddings[0].proj = new_conv
        model.load_state_dict(state_dict, strict=False)

    elif arch in {"unet_smp", "unet", "unet_resnet34"}:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required for SMP U-Net checkpoints."
            ) from exc

        encoder_name = "resnet34"
        if isinstance(checkpoint, dict):
            encoder_name = checkpoint.get("encoder", encoder_name)
        num_bands = int(checkpoint.get("num_bands", NUM_BANDS)) if isinstance(checkpoint, dict) else NUM_BANDS
        num_classes = int(checkpoint.get("num_classes", NUM_CLASSES)) if isinstance(checkpoint, dict) else NUM_CLASSES

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=num_bands,
            classes=num_classes,
            activation=None,
        )
        model.load_state_dict(state_dict, strict=False)

    elif arch in {"unet_official", "official_unet", "marida_unet"}:
        if "outc.weight" not in state_dict:
            raise ValueError("Official U-Net checkpoint missing outc.weight")
        num_classes = int(state_dict["outc.weight"].shape[0])
        hidden = int(state_dict["outc.weight"].shape[1])
        model = _OfficialMaridaUNet(
            in_channels=NUM_BANDS,
            num_classes=num_classes,
            hidden=hidden,
        )
        model.load_state_dict(state_dict, strict=True)

    else:
        raise ValueError(f"Unsupported model architecture: {arch}")

    model = model.to(device).float()
    model.eval()

    logger.info(
        "Loaded model checkpoint: %s  (arch=%s, device=%s)",
        checkpoint_path.name,
        arch,
        device,
    )
    return model


def _extract_logits(model_output: Any) -> torch.Tensor:
    """Normalize model outputs to a logits tensor ``(B, C, H, W)``."""
    if hasattr(model_output, "logits"):
        return model_output.logits
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)) and model_output and torch.is_tensor(model_output[0]):
        return model_output[0]
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


# ─────────────────────────────────────────────
# TTA INFERENCE
# ─────────────────────────────────────────────
def _apply_augmentation(patch: np.ndarray, aug_type: str) -> np.ndarray:
    """Apply a single augmentation transform to a patch.

    Args:
        patch: ``(C, H, W)`` array.
        aug_type: One of ``original``, ``hflip``, ``vflip``,
            ``rot90``, ``rot180``, ``rot270``.

    Returns:
        Augmented ``(C, H, W)`` array.
    """
    if aug_type == "original":
        return patch
    elif aug_type == "hflip":
        return patch[:, :, ::-1].copy()
    elif aug_type == "vflip":
        return patch[:, ::-1, :].copy()
    elif aug_type == "rot90":
        return np.rot90(patch, k=1, axes=(1, 2)).copy()
    elif aug_type == "rot180":
        return np.rot90(patch, k=2, axes=(1, 2)).copy()
    elif aug_type == "rot270":
        return np.rot90(patch, k=3, axes=(1, 2)).copy()
    return patch


def _reverse_augmentation(probs: np.ndarray, aug_type: str) -> np.ndarray:
    """Reverse an augmentation on a probability map.

    Args:
        probs: ``(C, H, W)`` probability array.
        aug_type: The augmentation that was applied.

    Returns:
        De-augmented ``(C, H, W)`` array.
    """
    if aug_type == "original":
        return probs
    elif aug_type == "hflip":
        return probs[:, :, ::-1].copy()
    elif aug_type == "vflip":
        return probs[:, ::-1, :].copy()
    elif aug_type == "rot90":
        return np.rot90(probs, k=-1, axes=(1, 2)).copy()
    elif aug_type == "rot180":
        return np.rot90(probs, k=-2, axes=(1, 2)).copy()
    elif aug_type == "rot270":
        return np.rot90(probs, k=-3, axes=(1, 2)).copy()
    return probs


@torch.no_grad()
def run_tta_inference(
    model: torch.nn.Module,
    patch: np.ndarray,
    device: torch.device,
    use_tta: bool = True,
    debris_logit_boost: float = DEBRIS_LOGIT_BOOST,
) -> np.ndarray:
    """Run inference with Test-Time Augmentation (6 variants).

    Applies an optional logit boost to the Marine Debris channel before
    softmax to structurally increase recall for the primary target class.

    Args:
        model: Loaded SegFormer model in eval mode.
        patch: ``(C, H, W)`` normalised float32 array.
        device: Torch device.
        use_tta: If ``False``, skip TTA and run a single forward pass.
        debris_logit_boost: Scalar added to the debris logit channel before
            softmax.  Higher values increase recall at the cost of precision.

    Returns:
        ``(num_classes, H, W)`` averaged probability map, upsampled to match
        the input patch spatial size.
    """
    H, W = patch.shape[1], patch.shape[2]
    aug_types = (
        ["original", "hflip", "vflip", "rot90", "rot180", "rot270"]
        if use_tta
        else ["original"]
    )

    accumulated = None

    for aug in aug_types:
        augmented = _apply_augmentation(patch, aug)
        tensor = torch.from_numpy(augmented).unsqueeze(0).float().to(device)

        out = model(tensor)
        logits = _extract_logits(out)  # (1, num_classes, H', W')

        # ── Debris logit boost — raise the debris channel before softmax ──
        if debris_logit_boost > 0:
            logits = logits.clone()
            logits[:, DEBRIS_CLASS_INDEX, :, :] += debris_logit_boost

        # ── Upsample back to patch resolution ─────────────────────────────
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        logits = torch.clamp(logits, -30.0, 30.0)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()  # (C, H, W)

        probs = _reverse_augmentation(probs, aug)

        if accumulated is None:
            accumulated = probs
        else:
            accumulated += probs

    return accumulated / len(aug_types)


def _load_patch_array(patch_path: Path) -> np.ndarray:
    """Load a patch array from .npy or .npz file."""
    if patch_path.suffix == ".npz":
        with np.load(patch_path) as data:
            if "patch" in data:
                return data["patch"]
            return data[list(data.files)[0]]
    return np.load(patch_path)


# ─────────────────────────────────────────────
# PATCH STITCHING
# ─────────────────────────────────────────────
def stitch_patches(
    predictions: List[np.ndarray],
    patch_ids: List[str],
    patch_index: Dict[str, Dict],
    scene_shape: Tuple[int, int],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Stitch patch predictions back into a full-scene probability map.

    Overlap regions are averaged.

    Args:
        predictions: List of ``(num_classes, H, W)`` probability arrays.
        patch_ids: List of patch IDs corresponding to predictions.
        patch_index: Patch index dict mapping patch_id → info.
        scene_shape: ``(height, width)`` of the original scene.
        num_classes: Number of output classes.

    Returns:
        ``(num_classes, H, W)`` full-scene probability map.
    """
    if num_classes is None and predictions:
        num_classes = predictions[0].shape[0]
    elif num_classes is None:
        num_classes = NUM_CLASSES

    h, w = scene_shape
    prob_sum = np.zeros((num_classes, h, w), dtype=np.float16)
    count = np.zeros((h, w), dtype=np.float16)

    for patch_id, pred in zip(patch_ids, predictions):
        info = patch_index[patch_id]
        rs = info["row_start"]
        cs = info["col_start"]
        ah = info["actual_h"]
        aw = info["actual_w"]

        prob_sum[:, rs:rs + ah, cs:cs + aw] += pred[:, :ah, :aw]
        count[rs:rs + ah, cs:cs + aw] += 1.0

    # Avoid division by zero
    count = np.maximum(count, 1.0)
    prob_map = prob_sum / count[np.newaxis, :, :]

    return prob_map.astype(np.float32)


# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────
def extract_clusters(
    debris_mask: np.ndarray,
    debris_prob: np.ndarray,
    transform: Affine,
    crs: Any,
    min_area_m2: float = MIN_CLUSTER_AREA_M2,
    max_area_m2: float = MAX_CLUSTER_AREA_M2,
    detection_date: str = "",
) -> gpd.GeoDataFrame:
    """Cluster connected debris pixels into detection objects.

    Args:
        debris_mask: ``(H, W)`` boolean mask of debris pixels.
        debris_prob: ``(H, W)`` debris probability map.
        transform: Affine geotransform.
        crs: Coordinate reference system.
        min_area_m2: Minimum cluster area in m².
        max_area_m2: Maximum cluster area in m² — clusters exceeding this are
            rejected as false positives (cloud shadow, sunglint, etc.).
        detection_date: ISO date string for the detection.

    Returns:
        :class:`GeoDataFrame` with columns: geometry, area_m2,
        mean_confidence, centroid_lon, centroid_lat, detection_date, cluster_id.
    """
    if not debris_mask.any():
        logger.info("No debris pixels detected — returning empty GeoDataFrame")
        return gpd.GeoDataFrame(
            columns=["geometry", "area_m2", "mean_confidence",
                     "centroid_lon", "centroid_lat", "detection_date",
                     "cluster_id"],
            crs=crs if crs else "EPSG:4326",
        )

    labeled, num_features = ndimage_label(debris_mask.astype(np.int32))
    component_slices = ndimage_find_objects(labeled)
    logger.info("Found %d raw connected components", num_features)

    # Pixel resolution in meters (approximate from transform)
    pixel_area_m2 = abs(transform.a * transform.e)
    min_pixels = max(1, int(min_area_m2 / pixel_area_m2))

    clusters = []
    wgs84_projector = None
    if crs and str(crs) != "EPSG:4326":
        import pyproj
        wgs84_projector = pyproj.Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True,
        ).transform

    for cluster_id, cluster_slice in enumerate(component_slices, start=1):
        if cluster_slice is None:
            continue

        local_labels = labeled[cluster_slice]
        local_mask = (local_labels == cluster_id)
        n_pixels = int(local_mask.sum())

        if n_pixels < min_pixels:
            continue

        area_m2 = float(n_pixels * pixel_area_m2)

        if area_m2 > max_area_m2:
            logger.debug(
                "Cluster %d rejected: area %.0f m² exceeds max %.0f m²",
                cluster_id, area_m2, max_area_m2,
            )
            continue
        mean_conf = float(debris_prob[cluster_slice][local_mask].mean())

        # Convert only the local component window to polygons.
        local_transform = transform * Affine.translation(
            cluster_slice[1].start, cluster_slice[0].start,
        )
        gdf = array_to_polygons(
            local_mask.astype(np.uint8), local_transform, crs,
        )
        if gdf.empty:
            continue

        # Merge all polygons for this cluster
        merged_geom = gdf.geometry.union_all()
        centroid = merged_geom.centroid

        # Re-project centroid to lon/lat if needed
        if wgs84_projector is not None:
            from shapely.ops import transform as shp_transform
            centroid_lonlat = shp_transform(wgs84_projector, centroid)
        else:
            centroid_lonlat = centroid

        clusters.append({
            "geometry": merged_geom,
            "area_m2": area_m2,
            "mean_confidence": mean_conf,
            "centroid_lon": centroid_lonlat.x,
            "centroid_lat": centroid_lonlat.y,
            "detection_date": detection_date,
            "cluster_id": len(clusters),
        })

        if cluster_id % 200 == 0 or cluster_id == num_features:
            logger.info(
                "  Processed %d/%d components (%d retained)",
                cluster_id,
                num_features,
                len(clusters),
            )

    if not clusters:
        logger.info("No clusters above minimum area (%d m²)", min_area_m2)
        return gpd.GeoDataFrame(
            columns=["geometry", "area_m2", "mean_confidence",
                     "centroid_lon", "centroid_lat", "detection_date",
                     "cluster_id"],
            crs=crs if crs else "EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(clusters, crs=crs if crs else "EPSG:4326")
    logger.info(
        "Extracted %d debris clusters (filtered from %d components, min %d m²)",
        len(gdf), num_features, min_area_m2,
    )
    return gdf


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_id: str,
    patches_dir: Union[str, Path],
    model_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/detections",
    config: Optional[Dict] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Path:
    """Run the full detection stage for one scene.

    Args:
        scene_id: Identifier of the scene being processed.
        patches_dir: Directory with ``.npy`` patch files.
        model_path: Path to the trained model checkpoint.
        output_dir: Root output directory for detections.
        config: Optional config dict.

    Returns:
        Path to the output ``detections.geojson``.

    Raises:
        FileNotFoundError: If patches or model cannot be found.
    """
    patches_dir = Path(patches_dir)
    out_dir = Path(output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight environment check: ensure PROJ is functional before spending 15 mins on inference
    try:
        import pyproj
        _test_transformer = pyproj.Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
    except Exception as e:
        logger.error("PROJ environment check failed! This means your spatial libraries are misconfigured.")
        logger.error(f"Error details: {e}")
        raise RuntimeError("PROJ environment check failed. See logs for details.") from e

    # Settings
    threshold = DEFAULT_THRESHOLD
    use_tta = True
    debris_logit_boost = 0.0
    min_area = MIN_CLUSTER_AREA_M2
    max_area = MAX_CLUSTER_AREA_M2
    model_arch = None
    if config:
        model_cfg = config.get("model", {})
        threshold = model_cfg.get("debris_threshold", DEFAULT_THRESHOLD)
        use_tta = model_cfg.get("tta", True)
        debris_logit_boost = model_cfg.get("debris_logit_boost", 0.0)
        model_arch = model_cfg.get("architecture")
        det_cfg = config.get("detection", {})
        min_area = det_cfg.get("min_cluster_area_m2", MIN_CLUSTER_AREA_M2)
        max_area = det_cfg.get("max_cluster_area_m2", MAX_CLUSTER_AREA_M2)

    # Check cache
    if stage_output_exists(out_dir, ["detections.geojson"]):
        return out_dir / "detections.geojson"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, architecture=model_arch)

    # Load patch index
    processed_dir = patches_dir.parent
    index_path = processed_dir / "patch_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Patch index not found: {index_path}")
    with open(index_path) as fh:
        patch_index = json.load(fh)

    # Load scene metadata for shape/CRS
    meta_path = processed_dir / "scene_meta.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            scene_meta = json.load(fh)
        scene_shape = (scene_meta["original_shape"][1],
                       scene_meta["original_shape"][2])
        scene_crs = scene_meta.get("crs")
        scene_transform_list = scene_meta.get("transform")
        if scene_transform_list:
            scene_transform = Affine(*scene_transform_list[:6])
        else:
            scene_transform = Affine(10, 0, 0, 0, -10, 0)  # fallback 10m
    else:
        # Infer from patches
        max_row_start = max(p["row_start"] + p["actual_h"]
                           for p in patch_index.values())
        max_col_start = max(p["col_start"] + p["actual_w"]
                           for p in patch_index.values())
        scene_shape = (max_row_start, max_col_start)
        scene_crs = None
        scene_transform = Affine(10, 0, 0, 0, -10, 0)

    # Run inference on each patch
    logger.info("Running inference on %d patches (TTA=%s)", len(patch_index), use_tta)

    predictions = []
    patch_ids = sorted(patch_index.keys())
    
    # Filter patches by bbox if provided
    if bbox:
        try:
            from pyproj import Transformer
            from shapely.geometry import box
            logger.info("Filtering patches with bbox=%s", bbox)
            
            filtered_ids = []
            for pid in patch_ids:
                info = patch_index[pid]
                geo_transform = info.get("geo_transform")
                crs = info.get("crs")
                if not geo_transform or not crs:
                    filtered_ids.append(pid)
                    continue
                    
                col_start = info.get("col_start", 0)
                row_start = info.get("row_start", 0)
                
                minx = geo_transform[2]
                maxy = geo_transform[5]
                maxx = minx + info.get("actual_w", 256) * geo_transform[0]
                miny = maxy + info.get("actual_h", 256) * geo_transform[4]
                minx, maxx = min(minx, maxx), max(minx, maxx)
                miny, maxy = min(miny, maxy), max(miny, maxy)
                patch_box = box(minx, miny, maxx, maxy)
                
                transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                bl_x, bl_y = transformer.transform(bbox[0], bbox[1])
                tr_x, tr_y = transformer.transform(bbox[2], bbox[3])
                tl_x, tl_y = transformer.transform(bbox[0], bbox[3])
                br_x, br_y = transformer.transform(bbox[2], bbox[1])
                
                user_minx = min(bl_x, tr_x, tl_x, br_x)
                user_maxx = max(bl_x, tr_x, tl_x, br_x)
                user_miny = min(bl_y, tr_y, tl_y, br_y)
                user_maxy = max(bl_y, tr_y, tl_y, br_y)
                user_box = box(user_minx, user_miny, user_maxx, user_maxy)
                
                if patch_box.intersects(user_box):
                    filtered_ids.append(pid)
                    
            logger.info("Retained %d out of %d patches within bbox", len(filtered_ids), len(patch_ids))
            patch_ids = filtered_ids
        except ImportError:
            logger.warning("pyproj or shapely not found, cannot filter patches by bbox")
    else:
        logger.info("No bbox provided for detection, processing all %d patches", len(patch_ids))

    full_probs = None
    counts = None

    for i, patch_id in enumerate(patch_ids):
        info = patch_index.get(patch_id, {})
        patch_file = info.get("patch_file", f"{patch_id}.npy")
        patch_path = patches_dir / patch_file
        if not patch_path.exists():
            npz_fallback = patches_dir / f"{patch_id}.npz"
            npy_fallback = patches_dir / f"{patch_id}.npy"
            if npz_fallback.exists():
                patch_path = npz_fallback
            elif npy_fallback.exists():
                patch_path = npy_fallback
        if not patch_path.exists():
            logger.warning("Patch file missing: %s", patch_path)
            continue

        patch = _load_patch_array(patch_path)
        prob_map = run_tta_inference(model, patch, device, use_tta=use_tta, debris_logit_boost=debris_logit_boost)
        
        if full_probs is None:
            num_classes = prob_map.shape[0]
            logger.info("Allocating float16 prediction buffers (approx 3.6 GB)")
            full_probs = np.zeros((num_classes, scene_shape[0], scene_shape[1]), dtype=np.float16)
            counts = np.zeros((scene_shape[0], scene_shape[1]), dtype=np.float16)

        rs = info.get("row_start", 0)
        cs = info.get("col_start", 0)
        ah = info.get("actual_h", 256)
        aw = info.get("actual_w", 256)

        full_probs[:, rs:rs + ah, cs:cs + aw] += prob_map[:, :ah, :aw].astype(np.float16)
        counts[rs:rs + ah, cs:cs + aw] += 1.0

        if (i + 1) % 50 == 0 or i == len(patch_ids) - 1:
            logger.info("  Processed %d/%d patches", i + 1, len(patch_ids))

    logger.info("Averaging overlapped patch predictions")
    counts = np.maximum(counts, 1.0)
    full_probs /= counts[np.newaxis, :, :]

    # Generate masks
    debris_prob = full_probs[DEBRIS_CLASS_INDEX].astype(np.float32)
    class_mask = full_probs.argmax(axis=0).astype(np.uint8)

    # Apply land mask filter if available
    land_mask_path = processed_dir / "land_mask.npy"
    if land_mask_path.exists():
        land_mask = np.load(land_mask_path)
        if land_mask.shape == debris_prob.shape:
            land_pixels_count = int(land_mask.sum())
            logger.info("Applying land mask filter (%d land pixels masked out)", land_pixels_count)
            debris_prob[land_mask] = 0.0

    debris_mask = (debris_prob > threshold) & (class_mask == DEBRIS_CLASS_INDEX)

    # Save outputs
    profile_base = {
        "driver": "GTiff",
        "height": scene_shape[0],
        "width": scene_shape[1],
        "transform": scene_transform,
        "crs": scene_crs,
    }

    # Binary debris mask
    save_geotiff(
        out_dir / "debris_mask.tif",
        debris_mask.astype(np.uint8),
        profile_base,
        dtype="uint8",
    )

    # Debris probability map
    save_geotiff(
        out_dir / "debris_prob.tif",
        debris_prob,
        profile_base,
        dtype="float32",
    )

    # Full class mask
    save_geotiff(
        out_dir / "class_mask.tif",
        class_mask,
        profile_base,
        dtype="uint8",
    )

    # Get detection date from scene metadata if available
    detection_date = ""
    ingest_meta = processed_dir.parent.parent / "raw" / scene_id / "metadata.json"
    if ingest_meta.exists():
        with open(ingest_meta) as fh:
            img_meta = json.load(fh)
        detection_date = img_meta.get("datetime", "")

    # Extract clusters as GeoJSON
    gdf = extract_clusters(
        debris_mask, debris_prob, scene_transform, scene_crs,
        min_area_m2=min_area, max_area_m2=max_area, detection_date=detection_date,
    )

    geojson_path = out_dir / "detections.geojson"
    geojson_gdf = gdf
    if not gdf.empty and gdf.crs and str(gdf.crs) != "EPSG:4326":
        geojson_gdf = gdf.to_crs("EPSG:4326")
    geojson_gdf.to_file(geojson_path, driver="GeoJSON")

    logger.info(
        "[bold green]Stage 3 complete[/] — %d clusters, debris pixels=%d (%.3f%%)",
        len(gdf),
        debris_mask.sum(),
        100 * debris_mask.sum() / debris_mask.size,
    )
    return geojson_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Run marine debris detection on preprocessed patches",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--patches_dir", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"d:/Plastic-Ledger/models/unet/runs/marida_v1/best_model.pth",
    )
    parser.add_argument("--output_dir", type=str, default="data/detections")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    geojson_path = run(
        scene_id=args.scene_id,
        patches_dir=args.patches_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"\nDetections saved to {geojson_path}")


if __name__ == "__main__":
    main()
