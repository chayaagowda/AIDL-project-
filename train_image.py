"""
Entry point: train the image-only ResNet-18 classifier.

Usage:
    python train_image.py
    python train_image.py --freeze --no-aug
"""

import argparse
import torch

from src.utils.config import load_config
from src.image.dataset import get_image_dataloaders
from src.image.model import ResNetClassifier
from src.image import train as image_train


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet image classifier on Hateful Memes")
    p.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    p.add_argument("--freeze", action="store_true", help="Freeze ResNet encoder")
    p.add_argument("--no-aug", action="store_true", help="Disable image augmentation")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.freeze:
        cfg.image.freeze_encoder = True
    if args.no_aug:
        cfg.image.augmentation = False
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr

    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Freeze encoder: {cfg.image.freeze_encoder}  |  Augmentation: {cfg.image.augmentation}")

    print("\nLoading data...")
    train_loader, val_loader = get_image_dataloaders(cfg)
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    print("\nBuilding model...")
    model = ResNetClassifier(
        model_name=cfg.image.model_name,
        hidden_dim=cfg.image.hidden_dim,
        dropout=cfg.image.dropout,
        freeze_encoder=cfg.image.freeze_encoder,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    print("\nStarting training...\n")
    image_train.train(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
