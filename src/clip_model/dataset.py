"""
CLIP dataset: preprocesses both text and image using the CLIP processor.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from datasets import load_dataset
from PIL import Image


class HatefulMemesCLIPDataset(Dataset):
    """
    Wraps a Hateful Memes split and applies CLIP preprocessing.

    CLIP has its own processor that handles both image resizing/normalisation
    and text tokenisation in a unified API.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: "train", "validation", or "test"
        model_name: CLIP model name (e.g. "openai/clip-vit-base-patch32")
    """

    def __init__(
        self,
        dataset_name: str = "limjiayi/hateful_memes_expanded",
        split: str = "train",
        model_name: str = "openai/clip-vit-base-patch32",
        data_root: str = "data",
    ):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        raw = load_dataset(dataset_name, split=split)
        pairs = [
            (txt, os.path.normpath(os.path.join(data_root, p)), lbl)
            for txt, p, lbl in zip(raw["text"], raw["img"], raw["label"])
            if os.path.isfile(os.path.normpath(os.path.join(data_root, p)))
        ]
        self.texts, self.img_paths, self.labels = zip(*pairs) if pairs else ([], [], [])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.img_paths[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")

        inputs = self.processor(
            text=self.texts[idx],
            images=img,
            return_tensors="pt",
            padding="max_length",
            max_length=77,        # CLIP's standard context length
            truncation=True,
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_clip_dataloaders(cfg) -> tuple:
    """
    Build train and validation DataLoaders for the CLIP pipeline.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = getattr(cfg.data, "data_root", "data")
    train_ds = HatefulMemesCLIPDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        model_name=cfg.clip.model_name,
        data_root=data_root,
    )
    val_ds = HatefulMemesCLIPDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.val_split,
        model_name=cfg.clip.model_name,
        data_root=data_root,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
