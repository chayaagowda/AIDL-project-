"""
Entry point: train the CLIP-based classifier (frozen encoder + MLP head).

Usage:
    python train_clip.py
    python train_clip.py --unfreeze    # fine-tune CLIP encoder too (GPU-heavy)
"""

import argparse
import torch

from src.utils.config import load_config
from src.clip_model.dataset import get_clip_dataloaders
from src.clip_model.model import CLIPClassifier
from src.clip_model import train as clip_train


def parse_args():
    p = argparse.ArgumentParser(description="Train CLIP classifier on Hateful Memes")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument(
        "--unfreeze",
        action="store_true",
        help="Fine-tune the CLIP encoder (default: frozen)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.unfreeze:
        cfg.clip.freeze_encoder = False
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr

    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Freeze CLIP encoder: {cfg.clip.freeze_encoder}")

    print("\nLoading data...")
    train_loader, val_loader = get_clip_dataloaders(cfg)
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    print("\nBuilding model...")
    model = CLIPClassifier(
        model_name=cfg.clip.model_name,
        hidden_dim=cfg.clip.hidden_dim,
        dropout=cfg.clip.dropout,
        freeze_encoder=cfg.clip.freeze_encoder,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable / Total parameters: {trainable:,} / {total:,}")

    print("\nStarting training...\n")
    clip_train.train(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
