"""
MARIDA Dataset Oversampling for Marine Debris
==============================================
Creates a balanced training dataset by oversampling patches with Marine Debris.

Usage:
    python oversample_marine_debris.py \
        --data_dir D:/Plastic-Ledger/ml_training/dataset/MARIDA \
        --output_dir D:/Plastic-Ledger/ml_training/dataset/MARIDA_BALANCED \
        --target_debris_ratio 0.35
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import rasterio
except ImportError:
    os.system("pip install rasterio")
    import rasterio

import numpy as np

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 15


def load_split_file(split_file):
    """Load patch names from split file."""
    if not split_file.exists():
        return []
    return [n.strip() for n in split_file.read_text().strip().splitlines() if n.strip()]


def has_marine_debris(patch_name, patches_dir):
    """Check if patch contains Marine Debris class (class 1)."""
    disk_name = f"S2_{patch_name}" if not patch_name.upper().startswith("S2_") else patch_name
    
    # Find mask file
    mask_files = list(patches_dir.rglob(f"{disk_name}_cl.tif"))
    if not mask_files:
        return False
    
    mask_path = mask_files[0]
    try:
        with rasterio.open(mask_path) as src:
            gt_raw = src.read(1).astype(np.int32)
            return (gt_raw == 1).sum() > 0  # Marine Debris = class 1
    except:
        return False


def copy_patch_pair(patch_name, src_patches_dir, dst_patches_dir):
    """Copy image and mask for a patch to destination."""
    disk_name = f"S2_{patch_name}" if not patch_name.upper().startswith("S2_") else patch_name
    
    # Find source files
    img_files = list(src_patches_dir.rglob(f"{disk_name}.tif"))
    mask_files = list(src_patches_dir.rglob(f"{disk_name}_cl.tif"))
    
    if not img_files or not mask_files:
        return False
    
    img_src = img_files[0]
    mask_src = mask_files[0]
    
    # Get subdirectory structure
    rel_img = img_src.relative_to(src_patches_dir)
    rel_mask = mask_src.relative_to(src_patches_dir)
    
    img_dst = dst_patches_dir / rel_img
    mask_dst = dst_patches_dir / rel_mask
    
    # Create destination directories
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    try:
        shutil.copy2(img_src, img_dst)
        shutil.copy2(mask_src, mask_dst)
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to copy {disk_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_debris_ratio", type=float, default=0.35,
                        help="Target ratio of patches with Marine Debris in training set")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    src_patches_dir = data_dir / "patches"
    dst_patches_dir = Path(args.output_dir) / "patches"
    splits_dir = Path(args.output_dir) / "splits"
    
    if not src_patches_dir.exists():
        print(f"❌ Source patches directory not found: {src_patches_dir}")
        return
    
    print(f"\n🔄 MARIDA OVERSAMPLING FOR MARINE DEBRIS")
    print(f"{'='*80}")
    print(f"Source:        {src_patches_dir}")
    print(f"Destination:   {dst_patches_dir}")
    print(f"Target ratio:  {args.target_debris_ratio*100:.0f}% patches with debris")
    print(f"{'='*80}\n")
    
    # Load training split
    train_split_file = data_dir / "splits" / "train_X.txt"
    train_patches = load_split_file(train_split_file)
    
    if not train_patches:
        print(f"❌ Training split file not found: {train_split_file}")
        return
    
    # Categorize patches
    debris_patches = []
    non_debris_patches = []
    
    print(f"Scanning {len(train_patches)} training patches...")
    for patch_name in train_patches:
        if has_marine_debris(patch_name, src_patches_dir):
            debris_patches.append(patch_name)
        else:
            non_debris_patches.append(patch_name)
    
    print(f"  ✓ Found {len(debris_patches)} patches WITH Marine Debris")
    print(f"  ✓ Found {len(non_debris_patches)} patches WITHOUT Marine Debris")
    
    # Calculate oversampling strategy
    total_patches = len(train_patches)
    target_debris_count = int(total_patches * args.target_debris_ratio)
    current_debris_count = len(debris_patches)
    augment_count = target_debris_count - current_debris_count
    
    print(f"\n  Target: {target_debris_count} debris patches ({args.target_debris_ratio*100:.0f}% of {total_patches})")
    print(f"  Current: {current_debris_count} debris patches")
    print(f"  Oversample: {augment_count} additional debris patches\n")
    
    if augment_count <= 0:
        print(f"  ℹ️  Dataset already balanced or target ratio too low. No oversampling needed.")
        augment_count = 0
    
    # Copy all non-debris patches
    print(f"Copying {len(non_debris_patches)} non-debris patches...")
    for i, patch_name in enumerate(non_debris_patches):
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(non_debris_patches)}")
        copy_patch_pair(patch_name, src_patches_dir, dst_patches_dir)
    
    # Copy all original debris patches
    print(f"Copying {len(debris_patches)} original debris patches...")
    for i, patch_name in enumerate(debris_patches):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(debris_patches)}")
        copy_patch_pair(patch_name, src_patches_dir, dst_patches_dir)
    
    # Oversample debris patches (with rotation/flipping could be added here)
    augmented_debris = []
    if augment_count > 0:
        print(f"Oversampling {augment_count} debris patches (repeating with suffix)...")
        # Simple strategy: repeat patches cyclically
        for i in range(augment_count):
            patch_idx = i % len(debris_patches)
            src_patch = debris_patches[patch_idx]
            # Create augmented name (could add rotation/flip later)
            aug_patch = f"{src_patch}_aug{i}"
            if copy_patch_pair(src_patch, src_patches_dir, dst_patches_dir):
                augmented_debris.append(aug_patch)
                # Rename copied files to have the augmented name
                disk_src = f"S2_{src_patch}" if not src_patch.upper().startswith("S2_") else src_patch
                disk_aug = f"S2_{aug_patch}" if not aug_patch.upper().startswith("S2_") else aug_patch
                
                # Find and rename
                for file in dst_patches_dir.rglob(f"{disk_src}*"):
                    new_file = file.parent / file.name.replace(disk_src, disk_aug)
                    try:
                        file.rename(new_file)
                    except:
                        pass
            
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{augment_count}")
    
    # Create new train split file with balanced patches
    balanced_train_patches = non_debris_patches + debris_patches + augmented_debris
    
    splits_dir.mkdir(parents=True, exist_ok=True)
    balanced_train_file = splits_dir / "train_X.txt"
    with open(balanced_train_file, "w") as f:
        for patch in balanced_train_patches:
            f.write(f"{patch}\n")
    
    # Copy val/test splits unchanged
    for split in ["val", "test"]:
        src_file = data_dir / "splits" / f"{split}_X.txt"
        dst_file = splits_dir / f"{split}_X.txt"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
    
    # ── Summary report ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"✅ OVERSAMPLING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Original training patches:  {len(train_patches)}")
    print(f"Balanced training patches:  {len(balanced_train_patches)}")
    print(f"  Non-debris:               {len(non_debris_patches)}")
    print(f"  Debris (original):        {len(debris_patches)}")
    print(f"  Debris (augmented):       {len(augmented_debris)}")
    print(f"\nBalanced debris ratio:      {len(debris_patches + augmented_debris) / len(balanced_train_patches) * 100:.1f}%")
    print(f"\nOutput directory:           {Path(args.output_dir)}")
    print(f"  patches/                  {dst_patches_dir.relative_to(Path(args.output_dir))}")
    print(f"  splits/train_X.txt        (balanced)")
    print(f"  splits/val_X.txt          (unchanged)")
    print(f"  splits/test_X.txt         (unchanged)")
    print()


if __name__ == "__main__":
    main()
