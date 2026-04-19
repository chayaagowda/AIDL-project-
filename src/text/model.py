"""
Text-only classification model: pretrained BERT + linear classification head.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    Binary classifier built on top of pretrained BERT.

    Architecture:
        BERT [CLS] embedding → Dropout → Linear → 2 logits

    Args:
        model_name: HuggingFace model identifier (default: bert-base-uncased)
        hidden_dim: Size of the intermediate projection layer (0 = skip)
        dropout: Dropout probability before the classifier head
        freeze_encoder: If True, BERT weights are frozen during training
        num_labels: Number of output classes (2 for binary)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        num_labels: int = 2,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_labels)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_embedding)

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return [CLS] embedding (used by the fusion model)."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # (batch, 768)
