"""
MARIDA Dataset Class Imbalance Analysis
========================================
Analyzes pixel-level class distribution across train/val/test splits
and identifies Marine Debris representation imbalance.

Usage:
    python analyze_class_imbalance.py \
        --data_dir D:/Plastic-Ledger/ml_training/dataset/MARIDA \
        --output imbalance_report.json
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import rasterio
except ImportError:
    os.system("pip install rasterio")
    import rasterio

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 15
CLASS_MAP = {
    1:  "Marine Debris",        2:  "Dense Sargassum",
    3:  "Sparse Sargassum",     4:  "Natural Organic Material",
    5:  "Ship",                 6:  "Clouds",
    7:  "Marine Water",         8:  "Sediment-Laden Water",
    9:  "Foam",                 10: "Turbid Water",
    11: "Shallow Water",        12: "Waves",
    13: "Cloud Shadows",        14: "Wakes",
    15: "Mixed Water",
}

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
def analyze_split(split_file, patches_dir):
    """Count pixel-level class distribution in a split."""
    if not split_file.exists():
        return None
    
    patch_names = split_file.read_text().strip().splitlines()
    patch_names = [n.strip() for n in patch_names if n.strip()]
    
    # Build lookup
    all_tifs = list(patches_dir.rglob("*.tif"))
    mask_lookup = {f.stem.replace("_cl", ""): f for f in all_tifs if "_cl" in f.stem}
    
    class_pixels = defaultdict(int)
    total_labeled = 0
    patches_with_debris = 0
    patch_count = 0
    
    for name in patch_names:
        name = name.strip()
        disk_name = f"S2_{name}" if not name.upper().startswith("S2_") else name
        
        if disk_name not in mask_lookup:
            continue
        
        msk_path = mask_lookup[disk_name]
        try:
            with rasterio.open(msk_path) as src:
                gt_raw = src.read(1).astype(np.int32)
            
            patch_count += 1
            # gt_raw has values 1-15 for classes, 0 for unconfirmed/missing
            unique, counts = np.unique(gt_raw, return_counts=True)
            
            for cls, cnt in zip(unique, counts):
                if cls > 0 and cls <= NUM_CLASSES:  # Valid class
                    class_pixels[cls] += int(cnt)
                    total_labeled += int(cnt)
            
            # Check if patch has Marine Debris (class 1)
            if (gt_raw > 0).sum() > 0:  # Has any labeled pixels
                if (gt_raw == 1).sum() > 0:
                    patches_with_debris += 1
        
        except Exception as e:
            print(f"  ⚠️  Skipped {disk_name}: {e}")
            continue
    
    return {
        "patches_total": patch_count,
        "patches_with_debris": patches_with_debris,
        "patches_debris_ratio": patches_with_debris / patch_count if patch_count > 0 else 0,
        "total_labeled_pixels": total_labeled,
        "class_pixels": dict(class_pixels),
        "class_percentages": {
            cls: (cnt / total_labeled * 100) if total_labeled > 0 else 0
            for cls, cnt in class_pixels.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="imbalance_report.json")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    patches_dir = data_dir / "patches"
    
    if not patches_dir.exists():
        print(f"❌ Patches directory not found: {patches_dir}")
        return
    
    # Analyze each split
    splits = ["train", "val", "test"]
    results = {}
    
    print(f"\n📊 MARIDA CLASS IMBALANCE ANALYSIS")
    print(f"{'='*80}")
    
    for split in splits:
        split_file = data_dir / "splits" / f"{split}_X.txt"
        print(f"\n📂 {split.upper()} split:")
        
        result = analyze_split(split_file, patches_dir)
        if result is None:
            print(f"  ⚠️  Split file not found: {split_file}")
            continue
        
        results[split] = result
        
        print(f"  Patches:                {result['patches_total']}")
        print(f"  Patches with debris:    {result['patches_with_debris']} ({result['patches_debris_ratio']*100:.1f}%)")
        print(f"  Total labeled pixels:   {result['total_labeled_pixels']:,}")
        print(f"\n  {'Class':<28} {'Pixels':>12}  {'%':>8}  {'Ratio vs Debris':>15}")
        print(f"  {'-'*70}")
        
        debris_pixels = result['class_pixels'].get(1, 1e-9)  # Marine Debris
        for cls in sorted(result['class_pixels'].keys()):
            pix = result['class_pixels'][cls]
            pct = result['class_percentages'][cls]
            ratio = pix / debris_pixels if cls != 1 else 1.0
            marker = " ← MARINE DEBRIS" if cls == 1 else ""
            print(f"  {CLASS_MAP[cls]:<28} {pix:>12,}  {pct:>7.2f}%  {ratio:>14.1f}x{marker}")
    
    # ── Cross-split analysis ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"📈 CROSS-SPLIT IMBALANCE SUMMARY")
    print(f"{'='*80}\n")
    
    # Calculate imbalance ratio per split
    for split, result in results.items():
        if result is None:
            continue
        debris_pix = result['class_pixels'].get(1, 1e-9)
        other_pix = sum(v for k, v in result['class_pixels'].items() if k != 1)
        ratio = other_pix / debris_pix if debris_pix > 0 else float('inf')
        
        print(f"{split.upper():8} | Debris: {debris_pix:>8,} px | Other: {other_pix:>10,} px | Imbalance ratio: {ratio:>6.1f}:1")
    
    # ── Recommendation ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"🎯 RECOMMENDATIONS FOR OVERSAMPLING")
    print(f"{'='*80}\n")
    
    if "train" in results:
        train_result = results["train"]
        debris_pix = train_result['class_pixels'].get(1, 1)
        other_pix = sum(v for k, v in train_result['class_pixels'].items() if k != 1)
        imbalance = other_pix / debris_pix
        
        print(f"Current training imbalance: {imbalance:.1f}:1 (other pixels : debris pixels)")
        print(f"Marine Debris patches: {train_result['patches_with_debris']} / {train_result['patches_total']} ({train_result['patches_debris_ratio']*100:.1f}%)")
        print(f"\nStrategy:")
        print(f"  1. Oversample patches with Marine Debris up to cover ~25-40% of training samples")
        print(f"  2. Use class weighting in loss: weight[Marine Debris] = 1.0 + (imbalance / 2)")
        print(f"  3. Consider Focal Loss or Dice Loss for better balance")
        
        target_debris_ratio = 0.35
        target_debris_patches = int(train_result['patches_total'] * target_debris_ratio)
        current_debris = train_result['patches_with_debris']
        augment_count = max(0, target_debris_patches - current_debris)
        
        print(f"\n  Proposed: Oversample {augment_count} additional debris patches")
        print(f"  (From {current_debris} → {target_debris_patches} debris patches, ~{target_debris_ratio*100:.0f}% of training set)")
    
    # ── Save report ──────────────────────────────────────────────
    output_path = Path(args.output)
    # Convert numpy int32 keys to strings for JSON serialization
    json_safe_results = {}
    for split, result in results.items():
        if result:
            json_safe_results[split] = {
                k: {str(cls): v for cls, v in result[k].items()} if isinstance(result[k], dict) else result[k]
                for k, v in result.items()
            }
    with open(output_path, "w") as f:
        json.dump(json_safe_results, f, indent=2, default=str)
    
    print(f"\n💾 Report saved → {output_path}\n")


if __name__ == "__main__":
    main()
