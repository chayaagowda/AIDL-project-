"""
Multimodal dataset: returns tokenised text + processed image + label.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

from src.image.dataset import get_transforms


class HatefulMemesMultimodalDataset(Dataset):
    """
    Combines text tokenisation and image transforms in a single dataset.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: "train", "validation", or "test"
        tokenizer_name: BERT tokeniser identifier
        max_text_length: Maximum token sequence length
        image_size: Target image resolution
        augmentation: Enable image augmentation (train only)
    """

    def __init__(
        self,
        dataset_name: str = "limjiayi/hateful_memes_expanded",
        split: str = "train",
        tokenizer_name: str = "bert-base-uncased",
        max_text_length: int = 128,
        image_size: int = 224,
        augmentation: bool = False,
        data_root: str = "data",
    ):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_text_length = max_text_length
        self.img_transform = get_transforms(image_size, augmentation=augmentation)

        raw = load_dataset(dataset_name, split=split)
        # Filter to samples whose image file actually exists on disk
        triples = [
            (txt, os.path.normpath(os.path.join(data_root, p)), lbl)
            for txt, p, lbl in zip(raw["text"], raw["img"], raw["label"])
            if os.path.isfile(os.path.normpath(os.path.join(data_root, p)))
        ]
        self.texts, self.img_paths, self.labels = zip(*triples) if triples else ([], [], [])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Text
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Image
        img = Image.open(self.img_paths[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": self.img_transform(img),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_multimodal_dataloaders(cfg) -> tuple:
    """
    Build train and validation DataLoaders for the multimodal pipeline.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = getattr(cfg.data, "data_root", "data")
    train_ds = HatefulMemesMultimodalDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        tokenizer_name=cfg.multimodal.text_model,
        max_text_length=cfg.data.max_text_length,
        image_size=cfg.data.image_size,
        augmentation=cfg.image.augmentation,
        data_root=data_root,
    )
    val_ds = HatefulMemesMultimodalDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.val_split,
        tokenizer_name=cfg.multimodal.text_model,
        max_text_length=cfg.data.max_text_length,
        image_size=cfg.data.image_size,
        augmentation=False,
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
