"""
Entry point: train the multimodal BERT + ResNet fusion classifier.

Usage:
    python train_fusion.py
    python train_fusion.py --freeze          # freeze both encoders, train only MLP head
    python train_fusion.py --epochs 10 --lr 1e-5
"""

import argparse
import torch

from src.utils.config import load_config
from src.multimodal.dataset import get_multimodal_dataloaders
from src.multimodal.model import MultimodalFusionModel
from src.multimodal import train as fusion_train


def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal fusion model on Hateful Memes")
    p.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    p.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze both BERT and ResNet encoders (only MLP head trained)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.freeze:
        cfg.multimodal.freeze_encoders = True
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr

    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Freeze encoders: {cfg.multimodal.freeze_encoders}")

    print("\nLoading data...")
    train_loader, val_loader = get_multimodal_dataloaders(cfg)
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    print("\nBuilding model...")
    model = MultimodalFusionModel(
        text_model_name=cfg.multimodal.text_model,
        image_model_name=cfg.multimodal.image_model,
        fusion_hidden_dim=cfg.multimodal.fusion_hidden_dim,
        dropout=cfg.multimodal.dropout,
        freeze_encoders=cfg.multimodal.freeze_encoders,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable / Total parameters: {trainable:,} / {total:,}")

    print("\nStarting training...\n")
    fusion_train.train(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
