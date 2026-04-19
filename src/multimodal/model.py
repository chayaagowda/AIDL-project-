"""
Multimodal fusion model: BERT (text) + ResNet (image) → MLP classifier.

Two encoders extract embeddings independently; embeddings are concatenated
and fed into a small MLP for binary classification.
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models


class MultimodalFusionModel(nn.Module):
    """
    Late-fusion multimodal classifier.

    Architecture:
        Text  → BERT → [CLS] embedding (768)  ─┐
                                                 ├─ Concat → MLP → 2 logits
        Image → ResNet backbone → feature (512) ─┘

    Args:
        text_model_name: HuggingFace BERT identifier
        image_model_name: torchvision ResNet identifier ("resnet18", "resnet50")
        fusion_hidden_dim: Hidden size of the MLP after concatenation
        dropout: Dropout probability
        freeze_encoders: Freeze both encoders; only the MLP head is trained
        num_labels: Number of output classes
    """

    TEXT_DIM = 768   # BERT-base hidden size
    IMAGE_DIM = 512  # ResNet-18 feature size (2048 for ResNet-50)

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "resnet18",
        fusion_hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_encoders: bool = False,
        num_labels: int = 2,
    ):
        super().__init__()

        # ── Text encoder ──────────────────────────────────────────────
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        # ── Image encoder ─────────────────────────────────────────────
        backbone = getattr(models, image_model_name)(weights="IMAGENET1K_V1")
        image_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        # Optionally freeze both encoders
        if freeze_encoders:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # ── Fusion MLP ────────────────────────────────────────────────
        fused_dim = text_dim + image_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            images:         (batch, 3, H, W)

        Returns:
            logits: (batch, num_labels)
        """
        # Text: extract [CLS] token
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state[:, 0, :]   # (batch, 768)

        # Image: global average pooled feature
        img_emb = self.image_encoder(images)              # (batch, 512)

        # Concatenate and classify
        fused = torch.cat([text_emb, img_emb], dim=-1)   # (batch, 1280)
        return self.classifier(fused)
