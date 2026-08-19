"""
Model Architecture Module for Fine-Grained Indian Car Classification.
Supports Transfer Learning with EfficientNet-B0, MobileNetV3, and ResNet.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple


class IndianCarClassifier(nn.Module):
    """
    Fine-Grained Visual Categorization Network for Indian Cars.
    Uses pretrained CNN backbones with customized classifier heads.
    """
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 12,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if "efficientnet" in self.backbone_name:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            # Replace classifier head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features=in_features, out_features=512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(p=dropout_rate / 2.0),
                nn.Linear(in_features=512, out_features=num_classes)
            )
            # Target layer for Grad-CAM activations
            self.target_cam_layer = self.backbone.features[-1]

        elif "mobilenet" in self.backbone_name:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=512),
                nn.Hardswish(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=512, out_features=num_classes)
            )
            self.target_cam_layer = self.backbone.features[-1]

        elif "resnet" in self.backbone_name:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=in_features, out_features=512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2.0),
                nn.Linear(in_features=512, out_features=num_classes)
            )
            self.target_cam_layer = self.backbone.layer4[-1]

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone and classifier head.
        Returns unnormalized logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)

    def freeze_backbone(self, freeze: bool = True):
        """
        Freezes feature extractor layers during initial warmup training epochs.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
        # Keep classifier head trainable
        if "resnet" in self.backbone_name:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True


def build_model(
    backbone_name: str = "efficientnet_b0",
    num_classes: int = 12,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
) -> IndianCarClassifier:
    """
    Factory function to initialize and optionally load trained weights.
    """
    model = IndianCarClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    return model
