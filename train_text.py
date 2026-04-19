"""
Entry point: train the text-only BERT classifier.

Usage:
    python train_text.py
    python train_text.py --config config/config.yaml --freeze
"""

import argparse
import torch

from src.utils.config import load_config
from src.text.dataset import get_text_dataloaders
from src.text.model import BertClassifier
from src.text import train as text_train


def parse_args():
    p = argparse.ArgumentParser(description="Train BERT text classifier on Hateful Memes")
    p.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    p.add_argument("--freeze", action="store_true", help="Freeze BERT encoder (override config)")
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate from config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.freeze:
        cfg.text.freeze_encoder = True
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr

    # Reproducibility
    torch.manual_seed(cfg.training.seed)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Freeze encoder: {cfg.text.freeze_encoder}")

    print("\nLoading data...")
    train_loader, val_loader = get_text_dataloaders(cfg)
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    print("\nBuilding model...")
    model = BertClassifier(
        model_name=cfg.text.model_name,
        hidden_dim=cfg.text.hidden_dim,
        dropout=cfg.text.dropout,
        freeze_encoder=cfg.text.freeze_encoder,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    print("\nStarting training...\n")
    text_train.train(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
