"""
Find patches with the most labeled Marine Debris pixels in GT masks.

Usage:
    python find_debris_patches.py --data_dir D:\Plastic-Ledger\models\dataset\MARIDA
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import rasterio
except ImportError:
    import os; os.system("pip install rasterio")
    import rasterio

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--top", type=int, default=20)
args = parser.parse_args()

data_dir = Path(args.data_dir)
cl_files = sorted(data_dir.rglob("*_cl.tif"))
print(f"\nScanning {len(cl_files)} mask files...\n")

results = []
for cl_path in cl_files:
    with rasterio.open(cl_path) as src:
        mask = src.read(1).astype(int)
    debris_px = (mask == 1).sum()   # class 1 = Marine Debris in raw file (1-indexed)
    if debris_px > 0:
        img_path = cl_path.parent / cl_path.name.replace("_cl.tif", ".tif")
        results.append((debris_px, cl_path.stem.replace("_cl",""), img_path))

results.sort(reverse=True)

print(f"Found {len(results)} patches with labeled debris\n")
print(f"{'Rank':<6} {'Debris px':>10} {'Patch name'}")
print("─" * 65)
for i, (px, name, path) in enumerate(results[:args.top], 1):
    exists = "✅" if path.exists() else "❌"
    print(f"  {i:<4} {px:>10,}   {exists} {name}")

print(f"\n── Top patch full path:")
if results:
    print(f"   {results[0][2]}")
    print(f"\n── Run inference on top {min(5,len(results))} debris patches:")
    for _, name, path in results[:5]:
        print(f"   python marida_inference.py --model runs\\marida_v1\\best_model.pth "
              f"--input \"{path}\" --output inference_output")