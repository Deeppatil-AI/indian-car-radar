"""
Domain-Specific Vehicle Data Augmentation Pipeline.
Strictly realistic automotive physics (no inverted cars, no extreme color distortion).
"""

import torch
import torchvision.transforms as transforms
from PIL import Image

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(input_size: int = 224, use_heavy_aug: bool = True) -> transforms.Compose:
    """
    Returns realistic vehicle training transforms.
    """
    if not use_heavy_aug:
        # Standard Light Augmentation
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    # Realistic Heavy Automotive Augmentation
    return transforms.Compose([
        transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Camera angle & perspective shifts (tilted roads, speed bumps, elevated views)
        transforms.RandomAffine(degrees=(-8, 8), translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.4),
        # Natural lighting, noon sun, shadows (Hue=0 to preserve true paint tone)
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Simulates real-world traffic occlusion (poles, road signs)
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), value='random')
    ])

def get_val_test_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Deterministic validation and test transforms.
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
