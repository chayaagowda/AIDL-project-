"""
Standalone evaluation script.

Loads a saved checkpoint and evaluates on the test (or validation) split.

Usage:
    python evaluate.py --model text   --checkpoint checkpoints/text_best.pt
    python evaluate.py --model image  --checkpoint checkpoints/image_best.pt
    python evaluate.py --model fusion --checkpoint checkpoints/fusion_best.pt
    python evaluate.py --model clip   --checkpoint checkpoints/clip_best.pt
    python evaluate.py --model text   --checkpoint checkpoints/text_best.pt --split test
"""

import argparse
import numpy as np
import torch

from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import compute_metrics, print_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    p.add_argument("--model", required=True, choices=["text", "image", "fusion", "clip"])
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    return p.parse_args()


def build_model_and_loader(model_type: str, cfg, split: str):
    """Instantiate the correct model and dataloader for the given type."""
    if model_type == "text":
        from src.text.model import BertClassifier
        from src.text.dataset import HatefulMemesTextDataset
        from torch.utils.data import DataLoader

        model = BertClassifier(
            model_name=cfg.text.model_name,
            hidden_dim=cfg.text.hidden_dim,
            dropout=cfg.text.dropout,
        )
        ds = HatefulMemesTextDataset(
            dataset_name=cfg.data.dataset_name,
            split=split,
            tokenizer_name=cfg.text.model_name,
            max_length=cfg.data.max_text_length,
        )
        loader = DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)

        def forward_fn(model, batch, device):
            return model(batch["input_ids"].to(device), batch["attention_mask"].to(device))

    elif model_type == "image":
        from src.image.model import ResNetClassifier
        from src.image.dataset import HatefulMemesImageDataset
        from torch.utils.data import DataLoader

        model = ResNetClassifier(
            model_name=cfg.image.model_name,
            hidden_dim=cfg.image.hidden_dim,
            dropout=cfg.image.dropout,
        )
        ds = HatefulMemesImageDataset(
            dataset_name=cfg.data.dataset_name,
            split=split,
            image_size=cfg.data.image_size,
        )
        loader = DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)

        def forward_fn(model, batch, device):
            return model(batch["image"].to(device))

    elif model_type == "fusion":
        from src.multimodal.model import MultimodalFusionModel
        from src.multimodal.dataset import HatefulMemesMultimodalDataset
        from torch.utils.data import DataLoader

        model = MultimodalFusionModel(
            text_model_name=cfg.multimodal.text_model,
            image_model_name=cfg.multimodal.image_model,
            fusion_hidden_dim=cfg.multimodal.fusion_hidden_dim,
            dropout=cfg.multimodal.dropout,
        )
        ds = HatefulMemesMultimodalDataset(
            dataset_name=cfg.data.dataset_name,
            split=split,
            tokenizer_name=cfg.multimodal.text_model,
            max_text_length=cfg.data.max_text_length,
            image_size=cfg.data.image_size,
        )
        loader = DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)

        def forward_fn(model, batch, device):
            return model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["image"].to(device),
            )

    elif model_type == "clip":
        from src.clip_model.model import CLIPClassifier
        from src.clip_model.dataset import HatefulMemesCLIPDataset
        from torch.utils.data import DataLoader

        model = CLIPClassifier(
            model_name=cfg.clip.model_name,
            hidden_dim=cfg.clip.hidden_dim,
            dropout=cfg.clip.dropout,
        )
        ds = HatefulMemesCLIPDataset(
            dataset_name=cfg.data.dataset_name,
            split=split,
            model_name=cfg.clip.model_name,
        )
        loader = DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)

        def forward_fn(model, batch, device):
            return model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
            )

    return model, loader, forward_fn


@torch.no_grad()
def run_evaluation(model, loader, forward_fn, device):
    model.eval()
    all_labels, all_logits = [], []

    for batch in loader:
        labels = batch["label"]
        logits = forward_fn(model, batch, device)
        all_labels.append(labels.numpy())
        all_logits.append(logits.cpu().numpy())

    labels_np = np.concatenate(all_labels)
    logits_np = np.concatenate(all_logits)
    return compute_metrics(labels_np, logits_np)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    print(f"Model type : {args.model}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"Device     : {device}\n")

    model, loader, forward_fn = build_model_and_loader(args.model, cfg, args.split)
    load_checkpoint(model, optimizer=None, path=args.checkpoint, device=device)
    model.to(device)

    metrics = run_evaluation(model, loader, forward_fn, device)
    print_metrics(metrics, split=args.split)


if __name__ == "__main__":
    main()
