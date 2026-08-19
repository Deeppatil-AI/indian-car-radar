"""
Training engine for Indian Car Classification.
Optimized for NVIDIA RTX 3050 (4GB VRAM) with Automatic Mixed Precision (AMP).
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import create_dataloaders
from src.model import build_model
from src.indian_cars_metadata import INDIAN_CAR_CLASSES


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device
) -> Dict[str, float]:
    """
    Executes one training epoch with FP16 Automatic Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaled backward pass
        if use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates model performance on validation split.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def run_training(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    backbone: str = "efficientnet_b0",
    train_dir: str = "data/train",
    val_dir: str = "data/val",
    save_dir: str = "models",
    progress_callback = None
) -> Dict[str, Any]:
    """
    Main training routine.
    Tracks history and saves best checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAINER] Active hardware device: {device}")
    if device.type == "cuda":
        print(f"[TRAINER] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[TRAINER] Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")

    num_classes = len(INDIAN_CAR_CLASSES)
    train_loader, val_loader, class_to_idx = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=0
    )

    if train_loader is None or len(train_loader) == 0:
        print("[TRAINER] No training data found. Please run data collector first.")
        return {"error": "No training data found"}

    model = build_model(backbone_name=backbone, num_classes=num_classes, pretrained=True, device=str(device))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "gpu_vram_mb": []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device) if val_loader else train_metrics
        scheduler.step()

        vram_usage = torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_metrics["loss"], 4))
        history["train_acc"].append(round(train_metrics["accuracy"], 4))
        history["val_loss"].append(round(val_metrics["loss"], 4))
        history["val_acc"].append(round(val_metrics["accuracy"], 4))
        history["gpu_vram_mb"].append(round(vram_usage, 1))

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.1f}% | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.1f}% | "
              f"VRAM: {vram_usage:.1f}MB")

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            checkpoint_path = Path(save_dir) / "best_indian_car_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "backbone": backbone,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes
            }, checkpoint_path)

        # Save metrics to JSON for live frontend charting
        with open(Path(save_dir) / "training_metrics.json", "w") as f:
            json.dump(history, f, indent=2)

        if progress_callback:
            progress_callback(epoch, epochs, train_metrics, val_metrics)

    total_time = time.time() - start_time
    print(f"[TRAINER] Training completed in {total_time:.1f}s. Best Val Acc: {best_val_acc*100:.1f}%")
    return history


if __name__ == "__main__":
    run_training(epochs=5, batch_size=8)
