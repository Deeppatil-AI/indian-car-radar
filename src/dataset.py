"""
Dataset module for Indian Car Classification.
Handles image loading, preprocessing, data augmentation, and DataLoader generation.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.indian_cars_metadata import CLASS_NAMES, LABEL_TO_INDEX

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Data augmentation pipeline for training:
    - RandomResizedCrop: Teaches model scale invariance
    - RandomHorizontalFlip: Mirrors images (cars are symmetric)
    - ColorJitter: Makes model robust to daylight/shadow/lighting changes
    - RandomRotation: Accounts for slight camera tilts
    - ToTensor & Normalize: Standardizes pixel distributions to [-2, 2] approx.
    """
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Deterministic evaluation pipeline for validation and inference.
    """
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class IndianCarDataset(Dataset):
    """
    PyTorch Dataset for Indian car images.
    Expects directory format:
      root_dir/
        class_name_1/
          img1.jpg, img2.png, ...
        class_name_2/
          ...
    """
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        
        self._load_samples()

    def _load_samples(self):
        if not self.root_dir.exists():
            return

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Build class mapping
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in valid_extensions:
                    self.samples.append((file_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to empty black image if corrupted
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    train_dir: str = "data/train",
    val_dir: str = "data/val",
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 0
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Dict[str, int]]:
    """
    Creates train and validation DataLoaders optimized for RTX 3050.
    """
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)

    train_dataset = IndianCarDataset(train_dir, transform=train_transform)
    val_dataset = IndianCarDataset(val_dir, transform=val_transform)

    train_loader = None
    val_loader = None

    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    class_to_idx = train_dataset.class_to_idx if len(train_dataset) > 0 else LABEL_TO_INDEX
    return train_loader, val_loader, class_to_idx
