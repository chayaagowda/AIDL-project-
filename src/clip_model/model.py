"""
CLIP-based classifier: frozen CLIP encoder + trainable MLP head.

CLIP produces a joint vision-language embedding. We concatenate the
text and image embeddings and pass them through a small classifier.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPClassifier(nn.Module):
    """
    Binary classifier on top of CLIP.

    Architecture:
        CLIP text encoder  → text embedding (512) ─┐
                                                     ├─ Concat → MLP → 2 logits
        CLIP image encoder → image embedding (512) ─┘

    The CLIP encoder is frozen by default; only the MLP head is trained.

    Args:
        model_name: HuggingFace CLIP identifier
        hidden_dim: MLP hidden size
        dropout: Dropout probability
        freeze_encoder: Freeze CLIP weights (recommended; CLIP is very large)
        num_labels: Number of output classes
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
        num_labels: int = 2,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        clip_dim = self.clip.config.projection_dim  # 512 for ViT-B/32

        if freeze_encoder:
            for param in self.clip.parameters():
                param.requires_grad = False

        fused_dim = clip_dim * 2  # text + image
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)  — CLIP tokenised text
            attention_mask: (batch, seq_len)
            pixel_values:   (batch, 3, 224, 224) — CLIP preprocessed image

        Returns:
            logits: (batch, num_labels)
        """
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        text_emb = outputs.text_embeds    # (batch, 512) — L2 normalised
        image_emb = outputs.image_embeds  # (batch, 512)
        fused = torch.cat([text_emb, image_emb], dim=-1)  # (batch, 1024)
        return self.classifier(fused)
