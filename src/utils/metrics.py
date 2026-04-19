"""
Evaluation metrics for binary classification.
Provides AUROC, F1, accuracy, and confusion matrix.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    """
    Compute all evaluation metrics from logits and ground-truth labels.

    Args:
        labels: Ground-truth binary labels, shape (N,)
        logits: Raw model output logits, shape (N, 2) or probabilities (N,)

    Returns:
        Dictionary with auroc, f1, accuracy, confusion_matrix keys.
    """
    # Convert logits to probabilities for the positive class
    if logits.ndim == 2:
        probs = softmax(logits)[:, 1]
    else:
        probs = logits  # already probabilities

    preds = (probs >= 0.5).astype(int)

    auroc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    return {
        "auroc": auroc,
        "f1": f1,
        "accuracy": acc,
        "confusion_matrix": cm,
    }


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def print_metrics(metrics: dict, split: str = "val") -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n[{split.upper()} Metrics]")
    print(f"  AUROC    : {metrics['auroc']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
