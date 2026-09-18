"""
Plastic-Ledger — SegFormer v2 Evaluation Script
==================================================
Evaluates trained SegFormer v2 models on the MARIDA test set (`test_X.txt`).
Computes per-class IoU, overall mIoU, Debris IoU/Precision/Recall/F1,
exports side-by-side visual prediction overlays, and writes a rich Markdown
evaluation report with per-patch breakdown and overall model summary.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from transformers import SegformerForSemanticSegmentation

# Class maps
RAW_CLASS_MAP = {
    0: "Marine Debris", 1: "Dense Sargassum", 2: "Sparse Sargassum",
    3: "Natural Organic", 4: "Ship", 5: "Clouds", 6: "Marine Water",
    7: "Sediment-Laden Water", 8: "Foam", 9: "Turbid Water",
    10: "Shallow Water", 11: "Waves", 12: "Cloud Shadows",
    13: "Wakes", 14: "Mixed Water"
}

AGG_CLASS_MAP = {
    0: "Marine Debris", 1: "Dense Sargassum", 2: "Sparse Sargassum",
    3: "Natural Organic", 4: "Ship", 5: "Clouds", 6: "Marine Water",
    7: "Sediment-Laden Water", 8: "Foam", 9: "Turbid Water",
    10: "Shallow Water"
}

COLORS_15 = [
    '#FF0000', '#00FF00', '#007F00', '#8B4513', '#808080', '#FFFFFF', '#0000FF',
    '#BDB76B', '#E0FFFF', '#4682B4', '#00CED1', '#4169E1', '#708090', '#F5FFFA', '#1E90FF'
]
COLORS_11 = COLORS_15[:11]

CMAP_15 = ListedColormap(COLORS_15)
CMAP_11 = ListedColormap(COLORS_11)


class MARIDATestDataset(Dataset):
    """Dataset for evaluating SegFormer v2 on MARIDA test split."""

    def __init__(self, data_dir: str, agg_to_water: bool = True):
        self.data_dir = Path(data_dir)
        self.patches_dir = self.data_dir / "patches"
        self.agg_to_water = agg_to_water

        split_path = self.data_dir / "splits" / "test_X.txt"
        if not split_path.exists():
            split_path = self.data_dir.parent / "MARIDA" / "splits" / "test_X.txt"

        self.patch_names = []
        if split_path.exists():
            names = [l.strip() for l in split_path.read_text().strip().splitlines() if l.strip()]
            for pname in names:
                parts = pname.rsplit("_", 1)
                scene = "S2_" + parts[0] if len(parts) == 2 else "S2_" + pname
                img_path = self.patches_dir / scene / f"S2_{pname}.tif"
                if not img_path.exists():
                    img_path = self.data_dir.parent / "MARIDA" / "patches" / scene / f"S2_{pname}.tif"
                if img_path.exists():
                    self.patch_names.append((pname, img_path))

    def __len__(self) -> int:
        return len(self.patch_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        pname, img_path = self.patch_names[idx]
        lbl_path = img_path.parent / f"S2_{pname}_cl.tif"

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        with rasterio.open(lbl_path) as src:
            label = src.read(1).astype(np.int64)

        if self.agg_to_water:
            mask_agg = np.isin(label, [12, 13, 14, 15])
            label[mask_agg] = 7

        mask_zero = (label == 0)
        label = label - 1
        label[mask_zero] = 255

        return torch.from_numpy(image), torch.from_numpy(label), pname, str(img_path)


def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def plot_patch_prediction(
    img: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: Path,
    pname: str,
    num_classes: int = 11
):
    """Plot side-by-side RGB, GT, and SegFormer prediction."""
    # RGB mapping: Band 4 (Red, idx 3), Band 3 (Green, idx 2), Band 2 (Blue, idx 1)
    rgb = img[[3, 2, 1], :, :]

    # Percentile normalize for visual contrast
    for i in range(3):
        p2, p98 = np.percentile(rgb[i], (2, 98))
        rgb[i] = np.clip((rgb[i] - p2) / (p98 - p2 + 1e-5), 0, 1)

    rgb = np.transpose(rgb, (1, 2, 0))

    gt_vis = np.ma.masked_where(gt == 255, gt)
    cmap = CMAP_11 if num_classes == 11 else CMAP_15

    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    titles = [f"RGB ({pname})", "Ground Truth (Red=Debris)", "SegFormer v2 Prediction"]
    images = [rgb, gt_vis, pred]

    for ax, title, img_data in zip(axs, titles, images):
        if title.startswith("RGB"):
            ax.imshow(img_data)
        else:
            ax.imshow(img_data, cmap=cmap, vmin=0, vmax=num_classes - 1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_markdown_report(
    out_path: Path,
    checkpoint: str,
    device: str,
    agg_to_water: bool,
    num_classes: int,
    class_map: Dict[int, str],
    results: Dict[str, Any],
    patch_rows: List[Dict[str, Any]],
    total_patches: int,
) -> None:
    """Write a rich Markdown evaluation report with per-patch table and model summary."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    miou = results["mIoU"]
    d_iou = results["debris_iou"]
    d_prec = results["debris_precision"]
    d_rec = results["debris_recall"]
    d_f1 = results["debris_f1"]
    per_class = results["per_class_iou"]

    # Sort patches: debris-containing first (actual_debris_px desc), then rest
    sorted_rows = sorted(patch_rows, key=lambda r: r["actual_debris_px"], reverse=True)

    lines: List[str] = []

    # ── Header ─────────────────────────────────────────────────────────────────
    lines += [
        "# SegFormer v2 — Evaluation Report",
        "",
        f"**Generated:** {ts}  ",
        f"**Checkpoint:** `{checkpoint}`  ",
        f"**Device:** `{device}`  ",
        f"**Mode:** {'11-class (agg_to_water)' if agg_to_water else '15-class (raw MARIDA)'}  ",
        f"**Total patches evaluated:** {total_patches}  ",
        "",
        "---",
        "",
    ]

    # ── Overall Model Summary ──────────────────────────────────────────────────
    lines += [
        "## Overall Model Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Overall mIoU** | `{miou:.4f}` |",
        f"| **Marine Debris IoU** | `{d_iou:.4f}` |",
        f"| **Marine Debris Precision** | `{d_prec:.4f}` |",
        f"| **Marine Debris Recall** | `{d_rec:.4f}` |",
        f"| **Marine Debris F1** | `{d_f1:.4f}` |",
        "",
        "---",
        "",
    ]

    # ── Per-Class IoU Table ────────────────────────────────────────────────────
    lines += [
        "## Per-Class IoU Breakdown",
        "",
        "| # | Class | IoU |",
        "|---|-------|-----|",
    ]
    for c_idx in range(num_classes):
        c_name = class_map.get(c_idx, str(c_idx))
        c_iou = per_class.get(c_name, 0.0)
        marker = " 🎯" if c_name == "Marine Debris" else ""
        lines.append(f"| {c_idx} | {c_name}{marker} | `{c_iou:.4f}` |")
    lines += ["", "---", ""]

    # ── Per-Patch Breakdown Table ──────────────────────────────────────────────
    patches_with_debris = sum(1 for r in patch_rows if r["has_gt_debris"])
    patches_pred_debris = sum(1 for r in patch_rows if r["pred_debris_px"] > 0)

    lines += [
        "## Per-Patch Breakdown",
        "",
        f"- Patches with **ground-truth debris**: {patches_with_debris} / {total_patches}",
        f"- Patches where **model predicted debris**: {patches_pred_debris} / {total_patches}",
        "",
        "| Patch Name | GT Debris px | Pred Debris px | TP | FP | FN "
        "| Debris IoU | Precision | Recall | F1 | Has GT Debris |",
        "|------------|-------------|----------------|----|----|----"
        "|------------|-----------|--------|----|--------------|",
    ]

    for r in sorted_rows:
        has_debris_str = "✅" if r["has_gt_debris"] else "—"
        lines.append(
            f"| `{r['patch_name']}` "
            f"| {r['actual_debris_px']:,} "
            f"| {r['pred_debris_px']:,} "
            f"| {r['tp']:,} "
            f"| {r['fp']:,} "
            f"| {r['fn']:,} "
            f"| `{r['patch_iou']:.4f}` "
            f"| `{r['patch_precision']:.4f}` "
            f"| `{r['patch_recall']:.4f}` "
            f"| `{r['patch_f1']:.4f}` "
            f"| {has_debris_str} |"
        )

    lines += ["", "---", ""]

    # ── Footer ─────────────────────────────────────────────────────────────────
    lines += [
        "*Report generated by Plastic-Ledger SegFormer v2 evaluation script.*",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Markdown report saved -> {out_path}", flush=True)


def load_segformer_checkpoint(
    checkpoint_path: str, num_classes: int = 11, device: torch.device = None, state_dict: dict = None
) -> torch.nn.Module:
    """Load model architecture and weights."""
    from transformers import SegformerConfig

    config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    config.num_labels = num_classes
    config.id2label = {i: str(i) for i in range(num_classes)}
    config.label2id = {str(i): i for i in range(num_classes)}

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        config=config,
        ignore_mismatched_sizes=True
    )

    old_conv = model.segformer.stages[0].patch_embeddings.proj
    new_conv = nn.Conv2d(
        in_channels=11,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    model.segformer.stages[0].patch_embeddings.proj = new_conv
    model.config.num_channels = 11

    if state_dict is None:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate SegFormer v2 on MARIDA Test Set")
    parser.add_argument("--data_dir", required=True, help="Path to MARIDA dataset directory")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model_SegFormer_v2.pth")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--agg_to_water", type=str, default="auto", help="Whether model was trained with agg_to_water ('auto', 'True', or 'False')")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inspect checkpoint to validate / auto-detect class count
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    ckpt_classes = state_dict["decode_head.classifier.weight"].shape[0]

    if args.agg_to_water.lower() == "auto":
        agg_to_water = (ckpt_classes == 11)
        print(f"[INFO] Auto-detected {ckpt_classes} classes from checkpoint -> agg_to_water={agg_to_water}", flush=True)
    else:
        user_agg = (args.agg_to_water.lower() in ["true", "1", "yes"])
        expected_classes = 11 if user_agg else 15
        if ckpt_classes != expected_classes:
            print(
                f"[WARN] Checkpoint has {ckpt_classes} classes but --agg_to_water={args.agg_to_water} implies {expected_classes} classes. "
                f"Auto-adjusting to match checkpoint ({ckpt_classes} classes).",
                flush=True
            )
            agg_to_water = (ckpt_classes == 11)
        else:
            agg_to_water = user_agg

    num_classes = ckpt_classes
    class_map = AGG_CLASS_MAP if (num_classes == 11) else RAW_CLASS_MAP

    print(f"Evaluating model {args.checkpoint} on {device} (num_classes={num_classes}, agg_to_water={agg_to_water})...", flush=True)

    dataset = MARIDATestDataset(args.data_dir, agg_to_water=agg_to_water)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    model = load_segformer_checkpoint(args.checkpoint, num_classes=num_classes, device=device, state_dict=state_dict)

    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    hist = np.zeros((num_classes, num_classes))
    total_patches = len(dataset)
    plotted_count = 0
    patch_rows: List[Dict[str, Any]] = []  # per-patch stats for the Markdown report

    with torch.no_grad():
        for images, labels, pnames, _ in loader:
            images = images.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

            preds = upsampled_logits.argmax(dim=1).cpu().numpy()
            targs = labels.numpy()

            for i in range(len(pnames)):
                pred = preds[i]
                targ = targs[i]
                pname = pnames[i]

                valid = (targ != 255)
                hist += fast_hist(targ[valid], pred[valid], num_classes)

                # ── Per-patch debris statistics ──────────────────────────────
                targ_valid = targ[valid]
                pred_valid = pred[valid]

                actual_debris_px = int((targ_valid == 0).sum())
                pred_debris_px   = int((pred_valid  == 0).sum())

                tp_px = int(((targ_valid == 0) & (pred_valid == 0)).sum())
                fp_px = int(((targ_valid != 0) & (pred_valid == 0)).sum())
                fn_px = int(((targ_valid == 0) & (pred_valid != 0)).sum())

                p_iou  = tp_px / (tp_px + fp_px + fn_px + 1e-6)
                p_prec = tp_px / (tp_px + fp_px + 1e-6)
                p_rec  = tp_px / (tp_px + fn_px + 1e-6)
                p_f1   = 2 * p_prec * p_rec / (p_prec + p_rec + 1e-6)

                patch_rows.append({
                    "patch_name":       pname,
                    "actual_debris_px": actual_debris_px,
                    "pred_debris_px":   pred_debris_px,
                    "tp":               tp_px,
                    "fp":               fp_px,
                    "fn":               fn_px,
                    "patch_iou":        float(p_iou),
                    "patch_precision":  float(p_prec),
                    "patch_recall":     float(p_rec),
                    "patch_f1":         float(p_f1),
                    "has_gt_debris":    actual_debris_px > 0,
                })

                # Save plots for patches that contain debris or sample patches
                if (targ == 0).any() or (pred == 0).any() or plotted_count < 10:
                    plot_patch_prediction(
                        images[i].cpu().numpy(), pred, targ,
                        plots_dir / f"eval_{pname}.png", pname,
                        num_classes=num_classes
                    )
                    plotted_count += 1

    # Compute metrics
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-6)
    miou = np.nanmean(iu)
    debris_iou = iu[0]

    # Debris confusion matrix stats
    tp = hist[0, 0]
    fp = hist[:, 0].sum() - tp
    fn = hist[0, :].sum() - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    results = {
        "mIoU": float(miou),
        "debris_iou": float(debris_iou),
        "debris_precision": float(precision),
        "debris_recall": float(recall),
        "debris_f1": float(f1),
        "per_class_iou": {class_map[c]: float(iu[c]) for c in range(num_classes)}
    }

    print("\n" + "="*50, flush=True)
    print("      SegFormer v2 Benchmark Test Results", flush=True)
    print("="*50, flush=True)
    print(f"Overall mIoU:         {miou:.4f}", flush=True)
    print(f"Marine Debris IoU:    {debris_iou:.4f}", flush=True)
    print(f"Marine Debris Prec:   {precision:.4f}", flush=True)
    print(f"Marine Debris Rec:    {recall:.4f}", flush=True)
    print(f"Marine Debris F1:     {f1:.4f}", flush=True)
    print("-" * 50, flush=True)
    print("Per-Class IoU Breakdown:", flush=True)
    for c_name, c_iou in results["per_class_iou"].items():
        print(f"  {c_name:<25}: {c_iou:.4f}", flush=True)
    print("="*50 + "\n", flush=True)

    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Write Markdown report ──────────────────────────────────────────────────
    write_markdown_report(
        out_path=out_dir / "evaluation_report.md",
        checkpoint=args.checkpoint,
        device=str(device),
        agg_to_water=agg_to_water,
        num_classes=num_classes,
        class_map=class_map,
        results=results,
        patch_rows=patch_rows,
        total_patches=total_patches,
    )

    print(f"Evaluation complete. Results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
