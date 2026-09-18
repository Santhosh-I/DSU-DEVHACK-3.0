"""
Plastic-Ledger — SegFormer v2 Training Script
================================================
Fixes critical flaws in SegFormer training:
1. Pretrained Weight Channel Realignment: Maps pretrained RGB weights
   accurately to MARIDA B04 (Red, ch 3), B03 (Green, ch 2), B02 (Blue, ch 1).
2. Water Class Aggregation (--agg_to_water): Aggregates secondary background
   water classes (Waves, Cloud Shadows, Wakes, Mixed Water) into Marine Water.
3. Supervision options for unannotated shoreline pixels.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Union

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from transformers import SegformerForSemanticSegmentation, SegformerConfig

# Class definitions
RAW_CLASS_NAMES = [
    "Marine Debris", "Dense Sargassum", "Sparse Sargassum",
    "Natural Organic", "Ship", "Clouds", "Marine Water",
    "Sediment-Laden Water", "Foam", "Turbid Water",
    "Shallow Water", "Waves", "Cloud Shadows",
    "Wakes", "Mixed Water"
]

AGG_CLASS_NAMES = [
    "Marine Debris", "Dense Sargassum", "Sparse Sargassum",
    "Natural Organic", "Ship", "Clouds", "Marine Water",
    "Sediment-Laden Water", "Foam", "Turbid Water",
    "Shallow Water"
]


class MARIDADataset(Dataset):
    """MARIDA Sentinel-2 Patch Dataset for SegFormer v2."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train_X",
        augment: bool = False,
        agg_to_water: bool = True,
        include_unannotated_as_water: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.patches_dir = self.data_dir / "patches"
        self.augment = augment
        self.agg_to_water = agg_to_water
        self.include_unannotated_as_water = include_unannotated_as_water

        split_path = self.data_dir / "splits" / f"{split}.txt"
        if not split_path.exists():
            split_path = self.data_dir.parent / "MARIDA" / "splits" / f"{split}.txt"

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pname, img_path = self.patch_names[idx]
        lbl_path = img_path.parent / f"S2_{pname}_cl.tif"

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        with rasterio.open(lbl_path) as src:
            label = src.read(1).astype(np.int64)

        if self.agg_to_water:
            # Map raw classes 12 (Waves), 13 (Cloud Shadows), 14 (Wakes), 15 (Mixed Water) to 7 (Marine Water)
            mask_agg = np.isin(label, [12, 13, 14, 15])
            label[mask_agg] = 7

        mask_zero = (label == 0)
        label = label - 1  # Map 1..15 (or 1..11) to 0-based indices

        if self.include_unannotated_as_water:
            # Set unannotated pixels to Marine Water (index 6, raw class 7)
            label[mask_zero] = 6
        else:
            # Set unannotated pixels to ignore_index (255)
            label[mask_zero] = 255

        if self.augment:
            if np.random.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                label = np.rot90(label, k=k, axes=(0, 1)).copy()

        return torch.from_numpy(image), torch.from_numpy(label)


class FocalLoss(nn.Module):
    """Focal Loss for handling severe class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        valid_pixels = (targets != self.ignore_index).sum()
        if valid_pixels == 0:
            return focal_loss.sum() * 0.0
        return focal_loss.sum() / valid_pixels


class ComboLoss(nn.Module):
    """Combined Focal Loss + Dice Loss for multi-class semantic segmentation."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, dice_weight: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return (1.0 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 15) -> Dict[str, float]:
    """Compute per-class IoU, mIoU, Debris IoU, Precision, Recall, and F1."""
    valid = (targets != 255)
    pred_valid = predictions[valid]
    targ_valid = targets[valid]

    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_valid == cls)
        targ_cls = (targ_valid == cls)

        intersection = (pred_cls & targ_cls).sum()
        union = (pred_cls | targ_cls).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(float(intersection / union))

    miou = float(np.nanmean(ious))
    debris_iou = ious[0] if not np.isnan(ious[0]) else 0.0

    # Debris Precision, Recall, F1 (Class 0: Marine Debris)
    debris_pred = (pred_valid == 0)
    debris_targ = (targ_valid == 0)
    tp = (debris_pred & debris_targ).sum()
    fp = (debris_pred & ~debris_targ).sum()
    fn = (~debris_pred & debris_targ).sum()

    precision = float(tp / (tp + fp + 1e-6))
    recall = float(tp / (tp + fn + 1e-6))
    f1 = float(2 * precision * recall / (precision + recall + 1e-6))

    return {
        "miou": miou,
        "debris_iou": float(debris_iou),
        "debris_precision": precision,
        "debris_recall": recall,
        "debris_f1": f1
    }


def create_segformer_model(num_classes: int = 15) -> torch.nn.Module:
    """Create SegFormer-B2 with aligned pretrained ImageNet/ADE20K weights."""
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
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

    with torch.no_grad():
        # Initialize non-RGB channels with average of pretrained RGB weights
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.data = mean_weight.repeat(1, 11, 1, 1)

        # Precise Weight Alignment for Sentinel-2 / MARIDA 11-band ordering:
        # Pretrained index 0 (Red)   -> MARIDA index 3 (B04 - Red)
        # Pretrained index 1 (Green) -> MARIDA index 2 (B03 - Green)
        # Pretrained index 2 (Blue)  -> MARIDA index 1 (B02 - Blue)
        new_conv.weight.data[:, 3, :, :] = old_conv.weight[:, 0, :, :].clone()  # B04 (Red)
        new_conv.weight.data[:, 2, :, :] = old_conv.weight[:, 1, :, :].clone()  # B03 (Green)
        new_conv.weight.data[:, 1, :, :] = old_conv.weight[:, 2, :, :].clone()  # B02 (Blue)

        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.clone()

    model.segformer.stages[0].patch_embeddings.proj = new_conv
    model.config.num_channels = 11
    return model


def main():
    parser = argparse.ArgumentParser(description="SegFormer v2 Training Script for Marine Debris Detection")
    parser.add_argument("--data_dir", required=True, help="Path to MARIDA dataset directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--agg_to_water", type=str, default="True", help="Aggregate secondary water classes to Marine Water")
    parser.add_argument("--include_unannotated_as_water", type=str, default="False", help="Treat unannotated pixels as Marine Water")
    args = parser.parse_args()

    agg_to_water = (args.agg_to_water.lower() in ["true", "1", "yes"])
    include_unannotated_as_water = (args.include_unannotated_as_water.lower() in ["true", "1", "yes"])

    num_classes = 11 if agg_to_water else 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Training SegFormer v2 with num_classes={num_classes}, agg_to_water={agg_to_water}, include_unannotated_as_water={include_unannotated_as_water}", flush=True)

    train_dataset = MARIDADataset(
        args.data_dir, split="train_X", augment=True,
        agg_to_water=agg_to_water, include_unannotated_as_water=include_unannotated_as_water
    )
    val_dataset = MARIDADataset(
        args.data_dir, split="val_X", augment=False,
        agg_to_water=agg_to_water, include_unannotated_as_water=include_unannotated_as_water
    )
    print(f"Loaded {len(train_dataset)} training patches, {len(val_dataset)} validation patches", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = create_segformer_model(num_classes=num_classes)
    model.to(device)

    criterion = ComboLoss(gamma=2.0, alpha=0.25, dice_weight=0.5, ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_debris_iou = -1.0
    metrics_log = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targs = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(pixel_values=images)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(upsampled_logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = upsampled_logits.argmax(dim=1).cpu().numpy()
            train_preds.append(preds)
            train_targs.append(labels.cpu().numpy())

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targs = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(pixel_values=images)
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

                loss = criterion(upsampled_logits, labels)
                val_loss += loss.item()

                preds = upsampled_logits.argmax(dim=1).cpu().numpy()
                val_preds.append(preds)
                val_targs.append(labels.cpu().numpy())

        train_preds = np.concatenate(train_preds)
        train_targs = np.concatenate(train_targs)
        val_preds = np.concatenate(val_preds)
        val_targs = np.concatenate(val_targs)

        train_metrics = compute_metrics(train_preds, train_targs, num_classes=num_classes)
        val_metrics = compute_metrics(val_preds, val_targs, num_classes=num_classes)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_miou": train_metrics["miou"],
            "train_debris_iou": train_metrics["debris_iou"],
            "val_miou": val_metrics["miou"],
            "val_debris_iou": val_metrics["debris_iou"],
            "val_debris_precision": val_metrics["debris_precision"],
            "val_debris_recall": val_metrics["debris_recall"],
            "val_debris_f1": val_metrics["debris_f1"],
        }
        metrics_log.append(epoch_log)

        print(f"Epoch {epoch+1}/{args.epochs}", flush=True)
        print(f"  Train Loss: {epoch_log['train_loss']:.4f}, Val Loss: {epoch_log['val_loss']:.4f}", flush=True)
        print(f"  Train mIoU: {epoch_log['train_miou']:.4f}, Val mIoU: {epoch_log['val_miou']:.4f}", flush=True)
        print(f"  Val Debris IoU: {epoch_log['val_debris_iou']:.4f}, Precision: {epoch_log['val_debris_precision']:.4f}, Recall: {epoch_log['val_debris_recall']:.4f}, F1: {epoch_log['val_debris_f1']:.4f}", flush=True)

        if epoch_log["val_debris_iou"] > best_val_debris_iou:
            best_val_debris_iou = epoch_log["val_debris_iou"]
            torch.save(model.state_dict(), out_dir / "best_model_SegFormer_v2.pth")

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

    # Plot metrics
    epochs_range = range(1, args.epochs + 1)
    train_losses = [m["train_loss"] for m in metrics_log]
    val_losses = [m["val_loss"] for m in metrics_log]
    val_debris_ious = [m["val_debris_iou"] for m in metrics_log]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_debris_ious, label="Val Debris IoU", color="red")
    plt.title("Val Debris IoU Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Debris IoU")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_plot.png")
    plt.close()

    print(f"Training complete. Artifacts saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
