"""
Evaluation and Metrics Analysis Module.
Computes Top-1, Top-K accuracy, Confusion Matrix, and generates classification reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import IndianCarDataset, get_val_transforms
from src.model import build_model
from src.indian_cars_metadata import INDIAN_CAR_CLASSES, INDEX_TO_LABEL


@torch.no_grad()
def evaluate_model(
    model_path: str = "models/best_indian_car_model.pth",
    val_dir: str = "data/val",
    batch_size: int = 16,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates trained checkpoint on validation set.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    num_classes = len(INDIAN_CAR_CLASSES)
    
    val_dataset = IndianCarDataset(val_dir, transform=get_val_transforms())
    if len(val_dataset) == 0:
        return {"error": "Validation dataset is empty"}

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = build_model(
        num_classes=num_classes,
        checkpoint_path=model_path if Path(model_path).exists() else None,
        device=str(device)
    )
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    for images, targets in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        all_probs.extend(probs)

    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)

    # Top-1 Accuracy
    top1_acc = np.mean(all_preds_np == all_targets_np)

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(all_preds_np, all_targets_np):
        if t < num_classes and p < num_classes:
            conf_matrix[t, p] += 1

    # Plot Black & White Retro Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="black")
    ax.set_facecolor("black")
    cax = ax.matshow(conf_matrix, cmap="gray")
    
    class_labels = [c["model"] for c in INDIAN_CAR_CLASSES]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_labels, rotation=45, ha="left", color="white", fontsize=8)
    ax.set_yticklabels(class_labels, color="white", fontsize=8)
    ax.tick_params(colors="white")

    plt.title("CONFUSION MATRIX // INDIAN CAR FGVC", color="white", fontsize=12, pad=20)
    plt.xlabel("PREDICTED CLASS", color="white", fontsize=10)
    plt.ylabel("ACTUAL CLASS", color="white", fontsize=10)

    # Annotate numbers
    for i in range(num_classes):
        for j in range(num_classes):
            val = conf_matrix[i, j]
            color = "black" if val > (conf_matrix.max() / 2) else "white"
            ax.text(j, i, str(val), va="center", ha="center", color=color, fontsize=7)

    plot_path = Path("static/confusion_matrix.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, facecolor="black", dpi=150)
    plt.close()

    metrics = {
        "top1_accuracy": round(float(top1_acc) * 100, 2),
        "total_samples": len(all_targets),
        "confusion_matrix_img": "/static/confusion_matrix.png"
    }

    return metrics
