"""
Visualise SegFormer v2 predictions on MARIDA test patches.

Produces a 4-column figure per sample:
  Col 1 – RGB composite (Sentinel-2 bands B04/B03/B02)
  Col 2 – Ground-truth label map (Marine Debris highlighted in red)
  Col 3 – Raw SegFormer v2 prediction
  Col 4 – SegFormer prediction AFTER polymer XGBoost false-positive filtering

Usage
-----
python visualize_predictions.py
    --data_dir   "d:/DSU-DEVHACK-3.0/dataset/MARIDA"
    --models_dir "d:/DSU-DEVHACK-3.0/models/production"
    --output_dir "d:/DSU-DEVHACK-3.0/ml_training/segformer/vis_output"
    --n_samples  10          # how many test patches to visualise
    --split      test_X      # which split file to use
    --seed       42
    --debris_only            # only pick patches that contain Marine Debris GT
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import xgboost as xgb
from transformers import SegformerForSemanticSegmentation

matplotlib.use("Agg")

# ── Class / colour constants ───────────────────────────────────────────────────
# 11 aggregated classes (agg_to_water=True, same as training)
AGG_CLASS_NAMES = [
    "Marine Debris",        # 0
    "Dense Sargassum",      # 1
    "Sparse Sargassum",     # 2
    "Natural Organic",      # 3
    "Ship",                 # 4
    "Clouds",               # 5
    "Marine Water",         # 6
    "Sediment-Laden Water", # 7
    "Foam",                 # 8
    "Turbid Water",         # 9
    "Shallow Water",        # 10
]

CLASS_COLORS = np.array([
    [220,  20,  60],   # Marine Debris      – crimson
    [  0, 128,   0],   # Dense Sargassum    – green
    [144, 238, 144],   # Sparse Sargassum   – light green
    [210, 105,  30],   # Natural Organic    – chocolate
    [255, 165,   0],   # Ship               – orange
    [169, 169, 169],   # Clouds             – dark grey
    [  0,   0, 205],   # Marine Water       – medium blue
    [210, 180, 140],   # Sediment-Laden     – tan
    [255, 255, 255],   # Foam               – white
    [ 64, 224, 208],   # Turbid Water       – turquoise
    [ 32, 178, 170],   # Shallow Water      – light sea green
], dtype=np.uint8)

# Band indices inside the MARIDA 11-band .tif (0-indexed)
BAND_IDX = {
    "nm440": 0, "nm490": 1, "nm560": 2, "nm665": 3, "nm705": 4,
    "nm740": 5, "nm783": 6, "nm842": 7, "nm865": 8,
    "nm1600": 9, "nm2200": 10,
}
R_IDX, G_IDX, B_IDX = 3, 2, 1   # B04, B03, B02


# ── SegFormer model factory (mirrors train.py) ─────────────────────────────────
def create_segformer_model(num_classes: int = 11) -> torch.nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    old_conv = model.segformer.stages[0].patch_embeddings.proj
    new_conv = nn.Conv2d(
        in_channels=11,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.data = mean_weight.repeat(1, 11, 1, 1)
        new_conv.weight.data[:, 3, :, :] = old_conv.weight[:, 0, :, :].clone()  # B04 = Red
        new_conv.weight.data[:, 2, :, :] = old_conv.weight[:, 1, :, :].clone()  # B03 = Green
        new_conv.weight.data[:, 1, :, :] = old_conv.weight[:, 2, :, :].clone()  # B02 = Blue
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.clone()
    model.segformer.stages[0].patch_embeddings.proj = new_conv
    model.config.num_channels = 11
    return model


# ── Spectral index helpers (mirrors polymer evaluate.py) ──────────────────────
def compute_spectral_indices(X: np.ndarray) -> np.ndarray:
    """X: (N, 11). Returns (N, 15) with PI, SR, NSI, FDI appended."""
    eps = 1e-8
    b04 = X[:, BAND_IDX["nm665"]]
    b06 = X[:, BAND_IDX["nm740"]]
    b08 = X[:, BAND_IDX["nm842"]]
    b8a = X[:, BAND_IDX["nm865"]]
    b11 = X[:, BAND_IDX["nm1600"]]

    pi  = (b08 - b04) / (b08 + b04 + eps)
    sr  = b11 / (b08 + eps)
    nsi = (b8a - b11) / (b8a + b11 + eps)
    interp = (832 - 665) / (1610 - 665 + eps)
    fdi = b08 - (b06 + (b11 - b06) * interp)

    return np.hstack([X, pi[:, None], sr[:, None], nsi[:, None], fdi[:, None]])


# ── Data helpers ───────────────────────────────────────────────────────────────
def load_patch(img_path: Path) -> np.ndarray:
    """Returns (11, H, W) float32 image array."""
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    return np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)


def load_label(lbl_path: Path, agg_to_water: bool = True) -> np.ndarray:
    """(H, W) int64; 0-based class index; 255 = unannotated."""
    with rasterio.open(lbl_path) as src:
        label = src.read(1).astype(np.int64)
    if agg_to_water:
        label[np.isin(label, [12, 13, 14, 15])] = 7   # agg secondary water to Marine Water
    mask_zero = label == 0
    label = label - 1          # 1..11 → 0..10
    label[mask_zero] = 255     # unannotated → ignore
    return label


def rgb_composite(image: np.ndarray) -> np.ndarray:
    """(11, H, W) → (H, W, 3) uint8 percentile-stretched RGB."""
    rgb = image[[R_IDX, G_IDX, B_IDX]].transpose(1, 2, 0).copy()
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        ch = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
        out[:, :, c] = (ch * 255).astype(np.uint8)
    return out


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    """(H, W) int labels → (H, W, 3) uint8. Ignore (255) → black."""
    H, W = label.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_idx, colour in enumerate(CLASS_COLORS):
        rgb[label == cls_idx] = colour
    return rgb


def gt_highlight_rgb(label: np.ndarray) -> np.ndarray:
    """White bg, Marine Debris (0) → crimson, other annotated → blue."""
    H, W = label.shape
    out = np.ones((H, W, 3), dtype=np.uint8) * 255
    out[label == 0] = [220, 20, 60]
    out[(label != 255) & (label != 0)] = [0, 0, 205]
    return out


# ── Polymer XGB false-positive filter ─────────────────────────────────────────
def apply_polymer_filter(
    image: np.ndarray,
    seg_pred: np.ndarray,
    xgb_model,
    feature_names: list,
    debris_seg_idx: int = 0,
    polymer_debris_label: int = 4,
) -> np.ndarray:
    """
    For every pixel predicted as Marine Debris by SegFormer, query the XGBoost
    polymer model. If XGBoost does NOT confirm debris, reassign to Marine Water.
    """
    filtered = seg_pred.copy()
    debris_mask = seg_pred == debris_seg_idx
    if not debris_mask.any():
        return filtered

    H, W = image.shape[1], image.shape[2]
    pixels_flat = image.reshape(11, H * W).T.astype(np.float32)
    debris_indices = np.where(debris_mask.ravel())[0]
    debris_pixels = pixels_flat[debris_indices]

    debris_features = compute_spectral_indices(debris_pixels)   # (N, 15)

    dmat = xgb.DMatrix(debris_features, feature_names=feature_names)
    xgb_preds = xgb_model.predict(dmat).astype(int)

    keep = xgb_preds == polymer_debris_label
    remove_mask_flat = np.zeros(H * W, dtype=bool)
    remove_mask_flat[debris_indices[~keep]] = True
    filtered[remove_mask_flat.reshape(H, W)] = 6   # → Marine Water
    return filtered


# ── Legend helper ─────────────────────────────────────────────────────────────
def make_legend_patches(present_classes: set) -> list:
    return [
        mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=AGG_CLASS_NAMES[i])
        for i in range(len(AGG_CLASS_NAMES)) if i in present_classes
    ]


# ── Per-patch figure ───────────────────────────────────────────────────────────
def visualize_patch(pname, img_path, segformer_model, xgb_model, feature_names, polymer_debris_label, device, out_dir):
    image = load_patch(img_path)
    lbl_path = img_path.parent / f"S2_{pname}_cl.tif"
    label = load_label(lbl_path, agg_to_water=True)

    # SegFormer inference
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = segformer_model(pixel_values=tensor).logits
        upsampled = nn.functional.interpolate(logits, size=image.shape[1:], mode="bilinear", align_corners=False)
        seg_pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

    # Polymer filter
    filtered_pred = apply_polymer_filter(image, seg_pred, xgb_model, feature_names, polymer_debris_label=polymer_debris_label)

    # Panels
    panel_rgb      = rgb_composite(image)
    panel_gt       = gt_highlight_rgb(label)
    panel_seg      = label_to_rgb(seg_pred)
    panel_filtered = label_to_rgb(filtered_pred)

    debris_gt  = int((label == 0).sum())
    debris_seg = int((seg_pred == 0).sum())
    debris_flt = int((filtered_pred == 0).sum())

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor("#1a1a2e")

    titles = [
        f"RGB (B04/B03/B02)\n{pname}",
        f"Ground Truth\n(Red=Debris, n={debris_gt:,})",
        f"SegFormer v2 Prediction\n(Debris pixels: {debris_seg:,})",
        f"+ Polymer Filter (XGBoost)\n(Debris pixels: {debris_flt:,}  ↓ {debris_seg - debris_flt:,} removed)",
    ]
    panels = [panel_rgb, panel_gt, panel_seg, panel_filtered]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color="white", pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Class legend on last panel
    all_pred_classes = set(seg_pred.ravel()) | set(filtered_pred.ravel())
    legend_patches = make_legend_patches(all_pred_classes)
    axes[3].legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=6.5,
        framealpha=0.9,
        labelcolor="black",
        facecolor="white",
        edgecolor="#aaa",
        ncol=1,
    )

    plt.tight_layout(pad=1.5)
    save_path = out_dir / f"{pname}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="4-column SegFormer + Polymer filter visualisation")
    parser.add_argument("--data_dir",   default=r"d:\DSU-DEVHACK-3.0\dataset\MARIDA")
    parser.add_argument("--models_dir", default=r"d:\DSU-DEVHACK-3.0\models\production")
    parser.add_argument("--output_dir", default=r"d:\DSU-DEVHACK-3.0\ml_training\segformer\vis_output")
    parser.add_argument("--split",      default="test_X")
    parser.add_argument("--n_samples",  type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--debris_only", action="store_true",
                        help="Only visualise patches that contain Marine Debris in ground truth")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir   = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SegFormer
    print("Loading SegFormer v2 …")
    segformer_model = create_segformer_model(num_classes=11)
    ckpt = models_dir / "best_model_SegFormer_v2.pth"
    segformer_model.load_state_dict(torch.load(ckpt, map_location=device))
    segformer_model.to(device).eval()
    print(f"  Loaded: {ckpt}")

    # Load XGBoost polymer model
    print("Loading Polymer XGBoost model …")
    xgb_model = joblib.load(models_dir / "polymer_xgb_model.pkl")
    with open(models_dir / "polymer_label_map.json") as f:
        polymer_label_map = json.load(f)
    polymer_debris_label = polymer_label_map.get("Marine Debris", 4)
    with open(models_dir / "polymer_feature_names.json") as f:
        feature_names = json.load(f)
    print(f"  Loaded: {models_dir / 'polymer_xgb_model.pkl'}")
    print(f"  Polymer debris label index: {polymer_debris_label}")
    print(f"  Feature names ({len(feature_names)}): {feature_names}")

    # Collect patches
    split_path = data_dir / "splits" / f"{args.split}.txt"
    patch_names = [l.strip() for l in split_path.read_text().strip().splitlines() if l.strip()]

    valid_patches = []
    for pname in patch_names:
        parts = pname.rsplit("_", 1)
        scene = "S2_" + parts[0] if len(parts) == 2 else "S2_" + pname
        img_path = data_dir / "patches" / scene / f"S2_{pname}.tif"
        if img_path.exists():
            valid_patches.append((pname, img_path))

    print(f"Found {len(valid_patches)} valid patches in split '{args.split}'")

    if args.debris_only:
        debris_patches = []
        print("  Filtering to patches with Marine Debris ground truth …")
        for pname, img_path in valid_patches:
            lbl_path = img_path.parent / f"S2_{pname}_cl.tif"
            with rasterio.open(lbl_path) as src:
                lbl = src.read(1)
            if (lbl == 1).any():   # MARIDA raw class 1 = Marine Debris
                debris_patches.append((pname, img_path))
        valid_patches = debris_patches
        print(f"  {len(valid_patches)} patches contain Marine Debris")

    n = min(args.n_samples, len(valid_patches))
    selected = random.sample(valid_patches, n)

    print(f"\nGenerating {n} figure(s) → {out_dir}")
    for i, (pname, img_path) in enumerate(selected, 1):
        print(f"[{i}/{n}] {pname}")
        try:
            visualize_patch(pname, img_path, segformer_model, xgb_model,
                            feature_names, polymer_debris_label, device, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
