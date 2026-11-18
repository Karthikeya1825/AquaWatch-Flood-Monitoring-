"""
AquaWatch - U-Net Model Training Script (PyTorch + SMP)

Features:
- U-Net with ResNet34 encoder adapted for 4 Sentinel-2 bands (RGB + NIR)
- Masked Binary Cross-Entropy + Dice Loss (ignoring -1 cloud/invalid pixels)
- AdamW optimizer with Cosine Annealing learning rate schedule
- Early stopping tracking validation water IoU
- Saves best checkpoint, curves.json, and training_curves.png
- Includes --quick_test mode for rapid CPU verification
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Internal module imports
from dataset import Sen1Floods11Dataset, get_train_transforms, get_valid_transforms, DEFAULT_BANDS
from model import build_model
from loss import MaskedBCEDiceLoss, calculate_batch_metrics


def set_seed(seed: int = 42):
    """Ensure reproducibility across PyTorch, NumPy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss, _, _ = criterion(logits, targets)

        loss.backward()
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics = calculate_batch_metrics(logits, targets)
        total_loss += loss.item()
        total_iou += metrics["iou"]
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_iou = total_iou / max(num_batches, 1)
    return avg_loss, avg_iou


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Validate on the validation split."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            logits = model(images)
            loss, _, _ = criterion(logits, targets)

            metrics = calculate_batch_metrics(logits, targets)
            total_loss += loss.item()
            total_iou += metrics["iou"]
            total_f1 += metrics["f1"]
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_iou = total_iou / max(num_batches, 1)
    avg_f1 = total_f1 / max(num_batches, 1)
    return avg_loss, avg_iou, avg_f1


def plot_curves(history: Dict[str, List[float]], output_path: Path):
    """Generate and save loss and IoU curve figures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor('#0B1F33')

    # 1. Loss curves
    ax1.set_facecolor('#0F2438')
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color='#1D8CF8', linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color='#FF8A3D', linewidth=2, linestyle='--')
    ax1.set_title("Training & Validation Loss (BCE + Dice)", color='#F6FAFD', fontsize=12)
    ax1.set_xlabel("Epoch", color='#F6FAFD')
    ax1.set_ylabel("Loss", color='#F6FAFD')
    ax1.tick_params(colors='#F6FAFD')
    ax1.grid(True, linestyle=':', alpha=0.3, color='#5B7083')
    ax1.legend(facecolor='#0B1F33', edgecolor='#5B7083', labelcolor='#F6FAFD')

    # 2. IoU curves
    ax2.set_facecolor('#0F2438')
    ax2.plot(epochs, history["train_iou"], label="Train IoU", color='#19C3B1', linewidth=2)
    ax2.plot(epochs, history["val_iou"], label="Val IoU", color='#1D8CF8', linewidth=2, linestyle='--')
    ax2.set_title("Water Class IoU over Epochs", color='#F6FAFD', fontsize=12)
    ax2.set_xlabel("Epoch", color='#F6FAFD')
    ax2.set_ylabel("IoU (Water)", color='#F6FAFD')
    ax2.tick_params(colors='#F6FAFD')
    ax2.grid(True, linestyle=':', alpha=0.3, color='#5B7083')
    ax2.legend(facecolor='#0B1F33', edgecolor='#5B7083', labelcolor='#F6FAFD')

    plt.tight_layout()
    plt.savefig(str(output_path), facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Training curves figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AquaWatch U-Net Training")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="AdamW weight decay")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet weights")
    parser.add_argument("--quick_test", action="store_true", help="Quick test mode (few epochs/samples)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
    parser.add_argument("--results_dir", type=str, default="../results", help="Path to results directory")
    args = parser.parse_args()

    set_seed(42)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("AquaWatch: Deep Learning Water Segmentation Training")
    print(f"Device: {device} | Pretrained: {args.pretrained} | Quick Test: {args.quick_test}")
    print("=" * 70)

    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Setup datasets & dataloaders
    if args.quick_test:
        print("\n[QUICK TEST MODE ACTIVATED]")
        print("Using small subset of local chips for rapid verification...")
        # Use available test chips as quick dataset
        full_ds = Sen1Floods11Dataset(
            data_dir=data_dir,
            split="test",
            bands=DEFAULT_BANDS,
            transform=get_train_transforms()
        )
        indices = list(range(min(12, len(full_ds))))
        train_indices = indices[:8]
        val_indices = indices[8:12]

        train_dataset = Subset(full_ds, train_indices)
        val_dataset = Subset(full_ds, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        epochs = 2
    else:
        # Full training on official train/valid splits
        train_dataset = Sen1Floods11Dataset(
            data_dir=data_dir,
            split="train",
            bands=DEFAULT_BANDS,
            transform=get_train_transforms()
        )
        val_dataset = Sen1Floods11Dataset(
            data_dir=data_dir,
            split="valid",
            bands=DEFAULT_BANDS,
            transform=get_valid_transforms()
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        epochs = args.epochs

    # Build model
    weights = "imagenet" if args.pretrained else None
    model = build_model(
        model_type="unet_resnet34",
        in_channels=len(DEFAULT_BANDS),
        classes=1,
        encoder_weights=weights
    ).to(device)

    # Loss & Optimizer
    criterion = MaskedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training state
    best_val_iou = -1.0
    history = {
        "train_loss": [],
        "train_iou": [],
        "val_loss": [],
        "val_iou": [],
        "val_f1": [],
        "learning_rates": []
    }

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_iou"].append(round(train_iou, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_iou"].append(round(val_iou, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["learning_rates"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}"
        )

        # Checkpoint if improved
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = checkpoints_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_f1": val_f1,
                "in_channels": len(DEFAULT_BANDS),
                "encoder": args.encoder
            }, best_model_path)
            print(f"  >>> Best model checkpoint saved to: {best_model_path} (Val IoU: {val_iou:.4f})")

    # Save curves history JSON
    curves_path = results_dir / "curves.json"
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training curve logs to: {curves_path}")

    # Plot curves
    plot_curves(history, figures_dir / "training_curves.png")

    print("\n" + "=" * 70)
    print(f"Training session complete. Best Validation IoU: {best_val_iou:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
