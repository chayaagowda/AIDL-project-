"""
Text-only dataset for the Hateful Memes task.
Loads text + label from a HuggingFace dataset split and tokenises with BERT.
"""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from datasets import load_dataset


class HatefulMemesTextDataset(Dataset):
    """
    Wraps a HuggingFace Hateful Memes split and tokenises the text field.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: "train", "validation", or "test"
        tokenizer_name: Pretrained tokeniser to use
        max_length: Maximum token sequence length
    """

    def __init__(
        self,
        dataset_name: str = "limjiayi/hateful_memes_expanded",
        split: str = "train",
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
    ):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        raw = load_dataset(dataset_name, split=split)
        # Keep only the columns we need
        self.texts = raw["text"]
        self.labels = raw["label"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_text_dataloaders(cfg) -> tuple:
    """
    Build train and validation DataLoaders for the text pipeline.

    Args:
        cfg: Config namespace (from load_config())

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_ds = HatefulMemesTextDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        tokenizer_name=cfg.text.model_name,
        max_length=cfg.data.max_text_length,
    )
    val_ds = HatefulMemesTextDataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.val_split,
        tokenizer_name=cfg.text.model_name,
        max_length=cfg.data.max_text_length,
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
