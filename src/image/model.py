"""
Image-only classification model: pretrained ResNet-18 + classifier head.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """
    Binary classifier built on top of pretrained ResNet-18.

    Architecture:
        ResNet-18 (backbone) → Global average pool → Dropout → Linear(hidden) → Linear(2)

    Args:
        model_name: torchvision model name ("resnet18", "resnet50", ...)
        hidden_dim: Intermediate projection size
        dropout: Dropout probability
        freeze_encoder: If True, ResNet weights are frozen (only head trains)
        num_labels: Number of output classes
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        num_labels: int = 2,
    ):
        super().__init__()

        # Load pretrained backbone
        backbone = getattr(models, model_name)(weights="IMAGENET1K_V1")
        in_features = backbone.fc.in_features  # 512 for resnet18

        # Replace the final FC layer with Identity to get feature vectors
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, H, W)

        Returns:
            logits: (batch, num_labels)
        """
        features = self.backbone(images)  # (batch, in_features)
        return self.classifier(features)

    def get_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Return backbone feature vector (used by the fusion model)."""
        return self.backbone(images)  # (batch, in_features)
