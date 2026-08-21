"""
Comprehensive Modular Training, Fine-Tuning & Evaluation Harness.
Supports Baseline Zero-Shot, Linear Probing, End-to-End ViT Fine-Tuning, and ArcFace Metric Learning.
"""

import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from src.augmentation_pipeline import get_train_transforms, get_val_test_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IndianCarsDataset(Dataset):
    def __init__(self, split_file: str, class_to_idx: Dict[str, int], transform=None):
        with open(split_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_p = item["image_path"]
        try:
            img = Image.open(img_p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            
        label = self.class_to_idx.get(item["standard_class"], 0)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, item["standard_class"], img_p


# ==========================================
# MODEL ARCHITECTURES
# ==========================================

class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True, unfreeze_blocks: int = 0):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            skip_validation=True, trust_repo=True
        )
        embed_dim = self.backbone.embed_dim # 384 for vits14
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        if unfreeze_blocks > 0:
            # Unfreeze the last N transformer blocks for domain adaptation
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits, features


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class DINOv2ArcFaceModel(nn.Module):
    def __init__(self, num_classes: int, unfreeze_blocks: int = 1):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            skip_validation=True, trust_repo=True
        )
        embed_dim = self.backbone.embed_dim
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze_blocks > 0:
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
                
        self.bn = nn.BatchNorm1d(embed_dim)
        self.arcface = ArcFaceHead(embed_dim, num_classes, s=30.0, m=0.30)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = self.bn(features)
        if labels is not None:
            logits = self.arcface(features, labels)
        else:
            logits = F.linear(F.normalize(features), F.normalize(self.arcface.weight)) * self.arcface.s
        return logits, features


# ==========================================
# EVALUATION ROUTINE
# ==========================================

def evaluate_model(model, dataloader, criterion, is_arcface=False):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            if is_arcface:
                logits, _ = model(images, labels=None)
            else:
                logits, _ = model(images)
                
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            
            # Top-1 and Top-5 accuracy
            _, preds_top1 = torch.max(logits, 1)
            _, preds_top5 = logits.topk(min(5, logits.size(1)), dim=1, largest=True, sorted=True)
            
            correct_top1 += (preds_top1 == labels).sum().item()
            correct_top5 += (preds_top5 == labels.view(-1, 1)).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds_top1.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, total)
    top1_acc = (correct_top1 / max(1, total)) * 100.0
    top5_acc = (correct_top5 / max(1, total)) * 100.0
    
    return avg_loss, top1_acc, top5_acc, all_preds, all_targets


# ==========================================
# TRAINING HARNESS
# ==========================================

def run_experiment(exp_name: str, config: Dict[str, Any]):
    print(f"\n{'='*70}")
    print(f"[EXP] RUNNING EXPERIMENT: {exp_name.upper()}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*70}")
    
    # Load Splits
    with open("data/clean_splits/class_to_idx.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    
    train_transform = get_train_transforms(use_heavy_aug=config.get("heavy_aug", True))
    val_test_transform = get_val_test_transforms()
    
    train_ds = IndianCarsDataset("data/clean_splits/train_split.json", class_to_idx, transform=train_transform)
    val_ds = IndianCarsDataset("data/clean_splits/val_split.json", class_to_idx, transform=val_test_transform)
    test_ds = IndianCarsDataset("data/clean_splits/test_split.json", class_to_idx, transform=val_test_transform)
    
    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # -------------------------------------------------------------
    # 1. Baseline Zero-Shot Prototype Metric Matching
    # -------------------------------------------------------------
    if config.get("model_type") == "zero_shot_baseline":
        print("[BASELINE] Extracting Zero-Shot DINOv2 Features...")
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", skip_validation=True, trust_repo=True).to(device)
        dinov2.eval()
        
        # Build Prototypes from Train Set
        train_features = []
        train_labels = []
        with torch.no_grad():
            for img, lbl, _, _ in train_loader:
                f = dinov2(img.to(device)).squeeze().cpu().numpy()
                f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                train_features.append(f_norm)
                train_labels.extend(lbl.numpy())
                
        train_features = np.vstack(train_features)
        train_labels = np.array(train_labels)
        
        # Prototype per class
        prototypes = np.zeros((num_classes, train_features.shape[1]), dtype=np.float32)
        for c in range(num_classes):
            c_mask = (train_labels == c)
            if np.sum(c_mask) > 0:
                p = np.mean(train_features[c_mask], axis=0)
                prototypes[c] = p / (np.linalg.norm(p) + 1e-8)

        # Evaluate on Validation
        val_correct_top1, val_correct_top5, val_total = 0, 0, 0
        with torch.no_grad():
            for img, lbl, _, _ in val_loader:
                f = dinov2(img.to(device)).squeeze().cpu().numpy()
                f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                sims = np.dot(f_norm, prototypes.T)
                top1 = np.argmax(sims, axis=1)
                top5 = np.argsort(sims, axis=1)[:, -5:]
                
                val_correct_top1 += np.sum(top1 == lbl.numpy())
                for i, target in enumerate(lbl.numpy()):
                    if target in top5[i]:
                        val_correct_top5 += 1
                val_total += len(lbl)
                
        val_top1 = (val_correct_top1 / val_total) * 100.0
        val_top5 = (val_correct_top5 / val_total) * 100.0

        # Evaluate on Isolated Test
        test_correct_top1, test_correct_top5, test_total = 0, 0, 0
        with torch.no_grad():
            for img, lbl, _, _ in test_loader:
                f = dinov2(img.to(device)).squeeze().cpu().numpy()
                f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                sims = np.dot(f_norm, prototypes.T)
                top1 = np.argmax(sims, axis=1)
                top5 = np.argsort(sims, axis=1)[:, -5:]
                
                test_correct_top1 += np.sum(top1 == lbl.numpy())
                for i, target in enumerate(lbl.numpy()):
                    if target in top5[i]:
                        test_correct_top5 += 1
                test_total += len(lbl)
                
        test_top1 = (test_correct_top1 / test_total) * 100.0
        test_top5 = (test_correct_top5 / test_total) * 100.0

        print(f" [BASELINE RESULT] Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | Test Top-1: {test_top1:.2f}% | Test Top-5: {test_top5:.2f}%")
        return {
            "exp_name": exp_name,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_loss": 0.0,
            "test_top1": test_top1,
            "test_top5": test_top5,
            "test_loss": 0.0
        }

    # -------------------------------------------------------------
    # 2. Supervised Training (Linear Probe / Fine-Tuning / ArcFace)
    # -------------------------------------------------------------
    is_arcface = (config.get("model_type") == "arcface")
    if is_arcface:
        model = DINOv2ArcFaceModel(num_classes, unfreeze_blocks=config.get("unfreeze_blocks", 1)).to(device)
    else:
        model = DINOv2Classifier(
            num_classes,
            freeze_backbone=config.get("freeze_backbone", True),
            unfreeze_blocks=config.get("unfreeze_blocks", 0)
        ).to(device)

    # Optimizer & LR Scheduling
    lr = config.get("lr", 3e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    
    # Differential Learning Rates (lower for ViT backbone, higher for head)
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr}
    ], weight_decay=weight_decay)
    
    epochs = config.get("epochs", 8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    
    best_val_acc = 0.0
    best_checkpoint_path = f"models/{exp_name}_best.pt"
    
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if is_arcface:
                logits, _ = model(images, labels=labels)
            else:
                logits, _ = model(images)
                
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        scheduler.step()
        
        train_avg_loss = train_loss / max(1, train_total)
        train_acc = (train_correct / max(1, train_total)) * 100.0
        
        # Validation Evaluation
        val_loss, val_top1, val_top5, _, _ = evaluate_model(model, val_loader, criterion, is_arcface=is_arcface)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_avg_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Top-1: {val_top1:.2f}% Top-5: {val_top5:.2f}%")
        
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_acc": val_top1,
                "class_to_idx": class_to_idx
            }, best_checkpoint_path)

    # Load Best Model for Final Isolated Test Set Evaluation
    print(f"\n[EVALUATION] Loading best validation checkpoint ({best_val_acc:.2f}%) for final Test evaluation...")
    ckpt = torch.load(best_checkpoint_path)
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_loss, test_top1, test_top5, test_preds, test_targets = evaluate_model(model, test_loader, criterion, is_arcface=is_arcface)
    val_loss, val_top1, val_top5, val_preds, val_targets = evaluate_model(model, val_loader, criterion, is_arcface=is_arcface)
    
    print(f"[BEST] [FINAL TEST ACCURACY] Test Top-1: {test_top1:.2f}% | Test Top-5: {test_top5:.2f}% | Test Loss: {test_loss:.4f}")
    
    return {
        "exp_name": exp_name,
        "val_top1": val_top1,
        "val_top5": val_top5,
        "val_loss": val_loss,
        "test_top1": test_top1,
        "test_top5": test_top5,
        "test_loss": test_loss,
        "test_preds": test_preds,
        "test_targets": test_targets
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all")
    args = parser.parse_args()
    
    results = {}
    
    # 1. Baseline Experiment
    if args.exp in ["all", "baseline"]:
        r_base = run_experiment("baseline_zeroshot", {
            "model_type": "zero_shot_baseline"
        })
        results["baseline_zeroshot"] = r_base
        
    # 2. Linear Probe / MLP Head
    if args.exp in ["all", "linear_probe"]:
        r_linear = run_experiment("exp1_linear_probe", {
            "model_type": "linear_probe",
            "freeze_backbone": True,
            "unfreeze_blocks": 0,
            "lr": 5e-4,
            "batch_size": 32,
            "epochs": 8,
            "label_smoothing": 0.1,
            "heavy_aug": True
        })
        results["exp1_linear_probe"] = r_linear

    # 3. Fine-Tuning Top 2 Blocks
    if args.exp in ["all", "finetune"]:
        r_ft = run_experiment("exp2_dinov2_finetune", {
            "model_type": "finetune",
            "freeze_backbone": False,
            "unfreeze_blocks": 2,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "epochs": 8,
            "label_smoothing": 0.1,
            "heavy_aug": True
        })
        results["exp2_dinov2_finetune"] = r_ft

    # 4. ArcFace Metric Learning
    if args.exp in ["all", "arcface"]:
        r_arc = run_experiment("exp3_arcface_metric", {
            "model_type": "arcface",
            "unfreeze_blocks": 2,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "epochs": 8,
            "heavy_aug": True
        })
        results["exp3_arcface_metric"] = r_arc

    # Save Experiment Comparison Report
    with open("data/clean_splits/experiment_results.json", "w", encoding="utf-8") as f:
        # Convert non-serializable elements
        clean_res = {}
        for k, v in results.items():
            clean_res[k] = {
                "exp_name": v["exp_name"],
                "val_top1": float(v["val_top1"]),
                "val_top5": float(v["val_top5"]),
                "test_top1": float(v["test_top1"]),
                "test_top5": float(v["test_top5"]),
                "test_loss": float(v.get("test_loss", 0.0))
            }
        json.dump(clean_res, f, indent=2)
