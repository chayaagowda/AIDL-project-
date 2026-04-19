"""
Core training and evaluation logic for the text-only (BERT) pipeline.
Called by train_text.py entry point.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.utils.metrics import compute_metrics, print_metrics
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import Logger


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    logger: Logger,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients with BERT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        logger.log_step(loss.item())

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate model on a dataloader.

    Returns:
        (avg_loss, metrics_dict)
    """
    model.eval()
    all_labels, all_logits = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    labels_np = np.concatenate(all_labels)
    logits_np = np.concatenate(all_logits)
    metrics = compute_metrics(labels_np, logits_np)
    return total_loss / len(loader), metrics


def train(model, train_loader, val_loader, cfg, device: torch.device):
    """
    Full training loop for the text model.

    Args:
        model: BertClassifier instance
        train_loader / val_loader: DataLoaders
        cfg: Config namespace
        device: torch.device
    """
    logger = Logger(cfg.logging.log_dir, name="text", log_every_n_steps=cfg.logging.log_every_n_steps)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = len(train_loader) * cfg.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps,
    )

    best_auroc = 0.0
    for epoch in range(1, cfg.training.num_epochs + 1):
        logger.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, logger)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print_metrics(val_metrics, split="val")
        logger.log_metrics({"train_loss": train_loss, "val_loss": val_loss, **{k: v for k, v in val_metrics.items() if k != "confusion_matrix"}})

        # Save checkpoint every N epochs
        if epoch % cfg.checkpoints.save_every_n_epochs == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, cfg.checkpoints.save_dir, name="text")

        # Track best model
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            save_checkpoint(model, optimizer, epoch, val_metrics, cfg.checkpoints.save_dir, name="text_best")
            print(f"  [Best] New best AUROC: {best_auroc:.4f}")

    logger.close()
    print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")
    return best_auroc
