"""
Checkpoint utilities: save and load model state dicts.
"""

import os
import torch


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, save_dir: str, name: str) -> str:
    """
    Save model + optimizer state to disk.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        metrics: Dict of metric values to store alongside the checkpoint
        save_dir: Directory to save checkpoints into
        name: Base name for the checkpoint file (e.g. "text_bert")

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}_epoch{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"[Checkpoint] Saved → {path}")
    return path


def load_checkpoint(model, optimizer, path: str, device: torch.device):
    """
    Load model + optimizer state from a checkpoint file.

    Args:
        model: PyTorch model (must match architecture)
        optimizer: PyTorch optimizer
        path: Path to the .pt checkpoint file
        device: Device to map tensors to

    Returns:
        Tuple of (epoch, metrics) stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    print(f"[Checkpoint] Loaded ← {path}  (epoch {epoch})")
    return epoch, metrics
