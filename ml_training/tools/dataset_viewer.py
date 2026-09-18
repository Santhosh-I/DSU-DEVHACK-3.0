"""
MARIDA Dataset Viewer
=====================
Run this script to explore and visualize the MARIDA dataset.
Share the printed outputs and saved images with Claude for analysis.

Usage:
    python marida_viewer.py --data_dir /path/to/MARIDA
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict, Counter

try:
    import rasterio
    from rasterio.plot import show
except ImportError:
    print("Installing rasterio...")
    os.system("pip install rasterio")
    import rasterio

try:
    import geopandas as gpd
except ImportError:
    print("Installing geopandas...")
    os.system("pip install geopandas")
    import geopandas as gpd

# ─────────────────────────────────────────────
# CLASS & CONFIDENCE MAPPINGS
# ─────────────────────────────────────────────
CLASS_MAP = {
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Sargassum",
    4: "Natural Organic Material",
    5: "Ship",
    6: "Clouds",
    7: "Marine Water",
    8: "Sediment-Laden Water",
    9: "Foam",
    10: "Turbid Water",
    11: "Shallow Water",
    12: "Waves",
    13: "Cloud Shadows",
    14: "Wakes",
    15: "Mixed Water",
}

CONF_MAP = {1: "High", 2: "Moderate", 3: "Low"}

CLASS_COLORS = [
    "#E63946", "#2A9D8F", "#57CC99", "#F4A261",
    "#264653", "#A8DADC", "#1D3557", "#E9C46A",
    "#F0EFEB", "#8338EC", "#3A86FF", "#FFFFFF",
    "#6B6570", "#B5D5C5", "#80B3FF",
]


# ─────────────────────────────────────────────
# 1. DATASET STRUCTURE OVERVIEW
# ─────────────────────────────────────────────
def overview(data_dir: Path):
    print("\n" + "=" * 60)
    print("  MARIDA DATASET OVERVIEW")
    print("=" * 60)

    splits_dir = data_dir / "splits"
    patches_dir = data_dir / "patches" if (data_dir / "patches").exists() else data_dir

    # Count patches
    all_tifs = list(patches_dir.rglob("*.tif"))
    image_tifs = [f for f in all_tifs if "_cl" not in f.stem and "_conf" not in f.stem]
    mask_tifs  = [f for f in all_tifs if "_cl" in f.stem]
    conf_tifs  = [f for f in all_tifs if "_conf" in f.stem]

    print(f"\n📁 Data directory   : {data_dir}")
    print(f"🖼  Image patches    : {len(image_tifs)}")
    print(f"🎭 Mask patches     : {len(mask_tifs)}")
    print(f"📊 Conf patches     : {len(conf_tifs)}")

    # Unique tiles & dates
    tiles = set()
    dates = set()
    for f in image_tifs:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            dates.add(parts[1])
            tiles.add(parts[2])
    print(f"📅 Unique dates     : {len(dates)}")
    print(f"🗺  Unique S2 tiles  : {len(tiles)}")
    print(f"   Tiles            : {sorted(tiles)}")

    # Split info
    if splits_dir.exists():
        print(f"\n{'─'*40}")
        print("  TRAIN / VAL / TEST SPLITS")
        print(f"{'─'*40}")
        for split_file in sorted(splits_dir.glob("*.txt")):
            lines = split_file.read_text().strip().splitlines()
            print(f"  {split_file.stem:<20}: {len(lines):>5} patches")

    return image_tifs, mask_tifs


# ─────────────────────────────────────────────
# 2. CLASS DISTRIBUTION
# ─────────────────────────────────────────────
def class_distribution(mask_tifs: list, max_masks: int = 200, save_path: str = "class_distribution.png"):
    print(f"\n{'='*60}")
    print("  CLASS DISTRIBUTION (sampling up to {max_masks} masks)")
    print(f"{'='*60}")

    pixel_counts = Counter()
    sampled = mask_tifs[:max_masks]

    for mf in sampled:
        with rasterio.open(mf) as src:
            arr = src.read(1).flatten().astype(np.int32)
            for cls, cnt in Counter(arr[arr > 0]).items():
                pixel_counts[int(cls)] += cnt

    total = sum(pixel_counts.values())
    print(f"\n{'Class':<30} {'Pixels':>12} {'%':>8}")
    print("─" * 52)
    for cls_id in sorted(pixel_counts):
        name = CLASS_MAP.get(cls_id, f"Unknown({cls_id})")
        cnt  = pixel_counts[cls_id]
        pct  = 100 * cnt / total
        print(f"  {name:<28} {cnt:>12,} {pct:>7.2f}%")

    # Bar chart
    labels = [CLASS_MAP.get(k, str(k)) for k in sorted(pixel_counts)]
    values = [pixel_counts[k] for k in sorted(pixel_counts)]
    colors = [CLASS_COLORS[(k-1) % len(CLASS_COLORS)] for k in sorted(pixel_counts)]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Total Pixel Count")
    ax.set_title("MARIDA — Class Distribution (pixel-level)", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{val:,}", ha="center", va="bottom", fontsize=7, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Saved → {save_path}")
    plt.close()

    return pixel_counts


# ─────────────────────────────────────────────
# 3. BAND STATISTICS
# ─────────────────────────────────────────────
def band_statistics(image_tifs: list, num_samples: int = 50, save_path: str = "band_stats.png"):
    print(f"\n{'='*60}")
    print(f"  BAND STATISTICS (sampling {num_samples} images)")
    print(f"{'='*60}")

    sampled = image_tifs[:num_samples]
    band_means = defaultdict(list)
    band_stds  = defaultdict(list)
    n_bands    = None

    for f in sampled:
        with rasterio.open(f) as src:
            data = src.read().astype(np.float32)
            n_bands = data.shape[0]
            for b in range(n_bands):
                band_means[b+1].append(np.nanmean(data[b]))
                band_stds[b+1].append(np.nanstd(data[b]))

    print(f"\n  Bands detected: {n_bands}")
    print(f"\n{'Band':<8} {'Mean':>12} {'Std':>12}")
    print("─" * 34)
    all_means, all_stds = [], []
    for b in range(1, n_bands+1):
        m = np.mean(band_means[b])
        s = np.mean(band_stds[b])
        all_means.append(m)
        all_stds.append(s)
        print(f"  B{b:<6} {m:>12.2f} {s:>12.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, n_bands+1)
    ax.bar(x, all_means, yerr=all_stds, capsize=4,
           color="#3A86FF", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Band Index")
    ax.set_ylabel("Mean Reflectance")
    ax.set_title("MARIDA — Per-band Mean ± Std (sampled patches)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{b}" for b in x])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Saved → {save_path}")
    plt.close()

    return all_means, all_stds, n_bands


# ─────────────────────────────────────────────
# 4. SAMPLE PATCH VISUALIZER
# ─────────────────────────────────────────────
def visualize_samples(image_tifs: list, mask_tifs: list, n_samples: int = 6,
                      save_path: str = "sample_patches.png"):
    print(f"\n{'='*60}")
    print(f"  SAMPLE PATCH VISUALIZER ({n_samples} patches)")
    print(f"{'='*60}")

    # Match images to masks by stem prefix
    mask_lookup = {f.stem.replace("_cl", ""): f for f in mask_tifs}
    pairs = []
    for img_f in image_tifs:
        key = img_f.stem
        if key in mask_lookup:
            pairs.append((img_f, mask_lookup[key]))
        if len(pairs) == n_samples:
            break

    print(f"  Matched pairs found: {len(pairs)}")

    fig, axes = plt.subplots(n_samples, 3, figsize=(14, n_samples * 3.5))
    if n_samples == 1:
        axes = [axes]

    for i, (img_f, msk_f) in enumerate(pairs):
        with rasterio.open(img_f) as src:
            data = src.read().astype(np.float32)
            n_bands = data.shape[0]

        with rasterio.open(msk_f) as src:
            mask = src.read(1).astype(np.int32)

        # RGB composite — use bands 3,2,1 if ≥3 bands (S2 standard)
        if n_bands >= 3:
            r = data[2]; g = data[1]; b = data[0]
        else:
            r = g = b = data[0]

        def norm(x):
            lo, hi = np.percentile(x, 2), np.percentile(x, 98)
            return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

        rgb = np.stack([norm(r), norm(g), norm(b)], axis=-1)

        # NIR band (B8 = index 7 in S2)
        if n_bands >= 8:
            nir = norm(data[7])
        else:
            nir = norm(data[-1])

        # Class mask colored
        colored_mask = np.zeros((*mask.shape, 3))
        present_classes = np.unique(mask[mask > 0])
        for cls_id in present_classes:
            color_hex = CLASS_COLORS[(cls_id - 1) % len(CLASS_COLORS)]
            r_c = int(color_hex[1:3], 16) / 255
            g_c = int(color_hex[3:5], 16) / 255
            b_c = int(color_hex[5:7], 16) / 255
            colored_mask[mask == cls_id] = [r_c, g_c, b_c]

        axes[i][0].imshow(rgb)
        axes[i][0].set_title(f"RGB — {img_f.stem}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(nir, cmap="inferno")
        axes[i][1].set_title("NIR Band", fontsize=8)
        axes[i][1].axis("off")

        axes[i][2].imshow(colored_mask)
        legend_patches = [
            mpatches.Patch(color=CLASS_COLORS[(c-1) % len(CLASS_COLORS)],
                           label=CLASS_MAP.get(c, str(c)))
            for c in present_classes
        ]
        axes[i][2].legend(handles=legend_patches, loc="lower right",
                          fontsize=6, framealpha=0.7)
        axes[i][2].set_title("Class Mask", fontsize=8)
        axes[i][2].axis("off")

    plt.suptitle("MARIDA — Sample Patches (RGB | NIR | Class Mask)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 5. CONFIDENCE LEVEL DISTRIBUTION
# ─────────────────────────────────────────────
def confidence_distribution(conf_tifs: list, max_conf: int = 200, save_path: str = "confidence_dist.png"):
    print(f"\n{'='*60}")
    print("  CONFIDENCE LEVEL DISTRIBUTION")
    print(f"{'='*60}")

    conf_counts = Counter()
    for cf in conf_tifs[:max_conf]:
        with rasterio.open(cf) as src:
            arr = src.read(1).flatten().astype(np.int32)
            for lvl, cnt in Counter(arr[arr > 0]).items():
                conf_counts[int(lvl)] += cnt

    total = sum(conf_counts.values())
    print(f"\n{'Level':<12} {'Name':<12} {'Pixels':>12} {'%':>8}")
    print("─" * 46)
    for lvl in sorted(conf_counts):
        name = CONF_MAP.get(lvl, "Unknown")
        cnt  = conf_counts[lvl]
        pct  = 100 * cnt / total
        print(f"  {lvl:<10} {name:<12} {cnt:>12,} {pct:>7.2f}%")

    labels = [CONF_MAP.get(k, str(k)) for k in sorted(conf_counts)]
    values = [conf_counts[k] for k in sorted(conf_counts)]
    colors = ["#2A9D8F", "#F4A261", "#E63946"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(values, labels=labels, colors=colors[:len(labels)],
           autopct="%1.1f%%", startangle=140,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("MARIDA — Confidence Level Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 6. PATCH-LEVEL MULTI-LABEL SUMMARY
# ─────────────────────────────────────────────
def multilabel_summary(data_dir: Path, save_path: str = "multilabel_summary.png"):
    label_file = data_dir / "labels_mapping.txt"
    if not label_file.exists():
        print(f"\n⚠️  labels_mapping.txt not found at {label_file}")
        return

    print(f"\n{'='*60}")
    print("  PATCH-LEVEL MULTI-LABEL SUMMARY")
    print(f"{'='*60}")

    label_counts = Counter()
    combo_counter = Counter()
    n_patches = 0

    import re
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle format: "patch_name [1, 7, 10]" or "patch_name 1 7 10"
            bracket_match = re.search(r'\[([^\]]+)\]', line)
            if bracket_match:
                labels = list(map(int, re.findall(r'\d+', bracket_match.group(1))))
            else:
                parts = line.split()
                try:
                    labels = list(map(int, parts[1:]))
                except ValueError:
                    continue
            if not labels:
                continue
            n_patches += 1
            for l in labels:
                label_counts[l] += 1
            combo_counter[tuple(sorted(labels))] += 1

    print(f"\n  Total patches parsed: {n_patches}")
    print(f"\n{'Class':<30} {'Patches':>10} {'%':>8}")
    print("─" * 50)
    for cls_id in sorted(label_counts):
        name = CLASS_MAP.get(cls_id, f"Unknown({cls_id})")
        cnt  = label_counts[cls_id]
        pct  = 100 * cnt / n_patches
        print(f"  {name:<28} {cnt:>10} {pct:>7.1f}%")

    print(f"\n  Top 10 class combinations:")
    for combo, cnt in combo_counter.most_common(10):
        names = " + ".join(CLASS_MAP.get(c, str(c)) for c in combo)
        print(f"    {cnt:>5}x  {names}")

    # Plot
    sorted_labels = sorted(label_counts, key=lambda x: label_counts[x], reverse=True)
    labels_names  = [CLASS_MAP.get(k, str(k)) for k in sorted_labels]
    values        = [label_counts[k] for k in sorted_labels]
    colors        = [CLASS_COLORS[(k-1) % len(CLASS_COLORS)] for k in sorted_labels]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(labels_names)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels_names)))
    ax.set_xticklabels(labels_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Patches")
    ax.set_title("MARIDA — Patch-level Multi-label Frequency", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MARIDA Dataset Viewer")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the root MARIDA dataset directory")
    parser.add_argument("--output_dir", type=str, default="marida_analysis",
                        help="Directory to save output images (default: marida_analysis)")
    parser.add_argument("--n_samples", type=int, default=6,
                        help="Number of sample patches to visualize (default: 6)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        sys.exit(1)

    # Run all analyses
    image_tifs, mask_tifs = overview(data_dir)

    conf_tifs = [f for f in data_dir.rglob("*.tif") if "_conf" in f.stem]

    class_distribution(mask_tifs,
        save_path=str(output_dir / "class_distribution.png"))

    band_statistics(image_tifs,
        save_path=str(output_dir / "band_stats.png"))

    visualize_samples(image_tifs, mask_tifs,
        n_samples=args.n_samples,
        save_path=str(output_dir / "sample_patches.png"))

    confidence_distribution(conf_tifs,
        save_path=str(output_dir / "confidence_dist.png"))

    multilabel_summary(data_dir,
        save_path=str(output_dir / "multilabel_summary.png"))

    print(f"\n{'='*60}")
    print(f"  ✅ ALL DONE — outputs saved to: {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()