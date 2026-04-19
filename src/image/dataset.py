"""
Image-only dataset for the Hateful Memes task.
Downloads images via the HuggingFace dataset and applies torchvision transforms.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


def get_transforms(image_size: int = 224, augmentation: bool = False) -> transforms.Compose:
    """
    Build image transform pipeline.

    Args:
        image_size: Target spatial resolution (square crop)
        augmentation: If True, add random flips/colour jitter for training

    Returns:
        torchvision.transforms.Compose object
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet statistics
        std=[0.229, 0.224, 0.225],
    )

    if augmentation:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


class HatefulMemesImageDataset(Dataset):
    """
    Wraps a HuggingFace Hateful Memes split and returns processed images + labels.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: "train", "validation", or "test"
        image_size: Target resolution for resizing
        augmentation: Enable data augmentation (train split only)
    """

    def __init__(
        self,
        dataset_name: str = "limjiayi/hateful_memes_expanded",
        split: str = "train",
        image_size: int = 224,
        augmentation: bool = False,
        data_root: str = "data",
    ):
        self.transform = get_transforms(image_size, augmentation=augmentation)
        self.data_root = data_root
        raw = load_dataset(dataset_name, split=split)
        # Filter to samples whose image file actually exists on disk
        pairs = [
            (os.path.normpath(os.path.join(data_root, p)), lbl)
            for p, lbl in zip(raw["img"], raw["label"])
        ]
        pairs = [(p, lbl) for p, lbl in pairs if os.path.isfile(p)]
        self.img_paths, self.labels = zip(*pairs) if pairs else ([], [])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.img_paths[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")
        return {
            "image": self.transform(img),                          # (3, H, W)
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_image_dataloaders(cfg) -> tuple:
    """
    Build train and validation DataLoaders for the image pipeline.

    Args:
        cfg: Config namespace

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = getattr(cfg.data, "data_root", "data")
    train_ds = HatefulMemesImageDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        image_size=cfg.data.image_size,
        augmentation=cfg.image.augmentation,
        data_root=data_root,
    )
    val_ds = HatefulMemesImageDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.val_split,
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
