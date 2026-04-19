# Hateful Meme Detection — Multimodal Deep Learning

Binary classification of hateful memes using text, image, fusion, and CLIP architectures, trained on the [Hateful Memes Expanded](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded) dataset.

| Model | Architecture | Best Val AUROC | Best Val F1 |
|---|---|---|---|
| Text (BERT) | BERT-base-uncased + MLP | 0.619 | 0.399 |
| Fusion (BERT + ResNet18) | Late fusion + MLP | 0.616 | 0.411 |
| CLIP (frozen) | CLIP ViT-B/32 + MLP | 0.597 | 0.031 |
| Image (ResNet18) | ResNet18 + MLP | 0.508 | 0.237 |

---

## Project Structure

```
AIDL-project-merged/
├── config/
│   └── config.yaml          # All hyperparameters and paths
├── src/
│   ├── text/
│   │   ├── dataset.py       # BERT tokenization + DataLoader
│   │   ├── model.py         # BERT + classifier head
│   │   └── train.py         # Train/eval loop
│   ├── image/
│   │   ├── dataset.py       # Image loading + transforms
│   │   ├── model.py         # ResNet18 + classifier head
│   │   └── train.py         # Train/eval loop
│   ├── multimodal/
│   │   ├── dataset.py       # Combined text + image dataset
│   │   ├── model.py         # BERT + ResNet18 → concat → MLP
│   │   └── train.py         # Train/eval loop
│   ├── clip_model/
│   │   ├── dataset.py       # CLIP processor (text + image)
│   │   ├── model.py         # Frozen CLIP + MLP head
│   │   └── train.py         # Train/eval loop
│   └── utils/
│       ├── config.py        # YAML config loader
│       ├── metrics.py       # AUROC, F1, accuracy
│       ├── checkpoint.py    # Save/load checkpoints
│       └── logger.py        # Step + epoch logger
├── data/                    # Downloaded dataset (images + JSONL splits)
├── checkpoints/             # Saved model checkpoints
├── logs/                    # Training logs per model
├── results/                 # Saved evaluation metrics
├── train_text.py            # Entry point — text model
├── train_image.py           # Entry point — image model
├── train_fusion.py          # Entry point — fusion model
├── train_clip.py            # Entry point — CLIP model
├── evaluate.py              # Evaluate any saved checkpoint
├── download_dataset.py      # One-time dataset download
└── requirements.txt
```

---

## Dependencies

Python 3.9+ is required.

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
Pillow>=9.0.0
pyyaml>=6.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
huggingface_hub>=0.16.0
```

For GPU training, install the appropriate CUDA build of PyTorch from https://pytorch.org before running `pip install -r requirements.txt`. All experiments in this project were run on CPU; GPU is optional but strongly recommended for reasonable training times.

> **Note for Windows users:** HuggingFace's cache system uses symlinks by default. To avoid warnings, either enable Developer Mode in Windows settings or set the environment variable `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. This does not affect functionality.

---

## Dataset Instructions

The dataset is hosted on HuggingFace and must be downloaded before training. The download script fetches all splits (train, validation, test) and the full image archive into a local `data/` folder.

```bash
python download_dataset.py
```

This creates the following structure:

```
data/
├── img/           # ~13,300 meme images (.png)
├── train.jsonl
├── dev_seen.jsonl
├── dev_unseen.jsonl
└── test_seen.jsonl
```

**Expected size:** ~1–2 GB total.

> The expanded dataset includes a `covid_memes` subset. Approximately 500 of those images may be missing from the HuggingFace archive. The training scripts automatically skip missing files — this affects less than 4% of training data and does not require any manual intervention.

---

## Reproduction Steps

Follow these steps in order to reproduce the results reported in the final report.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset

```bash
python download_dataset.py
```

### Step 3 — Train each model

Run each training script from the project root directory. Checkpoints are saved to `checkpoints/` and logs to `logs/` after every epoch.

```bash
# Text model (BERT)
python train_text.py

# Image model (ResNet18)
python train_image.py

# Fusion model (BERT + ResNet18)
python train_fusion.py

# CLIP model (frozen encoder)
python train_clip.py
```

Each script runs for 5 epochs by default (configured in `config/config.yaml`). On CPU, expect roughly 1–2 hours per epoch for the text and fusion models.

### Step 4 — Evaluate a checkpoint

```bash
python evaluate.py --model text   --checkpoint checkpoints/text_best_epoch4.pt
python evaluate.py --model image  --checkpoint checkpoints/image_best_epoch5.pt
python evaluate.py --model fusion --checkpoint checkpoints/fusion_best_epoch4.pt
python evaluate.py --model clip   --checkpoint checkpoints/clip_best_epoch5.pt
```

Use `--split test` to evaluate on the test split instead of validation:

```bash
python evaluate.py --model text --checkpoint checkpoints/text_best_epoch4.pt --split test
```

Evaluation results are printed to the console and saved to `results/`.

---

## Configuration

All hyperparameters and paths are in `config/config.yaml`. Key options:

| Section | Key | Default | Description |
|---|---|---|---|
| `data` | `dataset_name` | `limjiayi/hateful_memes_expanded` | HuggingFace dataset ID |
| `training` | `batch_size` | `32` | Batch size for all models |
| `training` | `num_epochs` | `5` | Training epochs |
| `training` | `learning_rate` | `2e-5` | AdamW learning rate |
| `training` | `device` | `"cuda"` | Set to `"cpu"` if no GPU |
| `text` | `freeze_encoder` | `false` | Freeze BERT weights |
| `image` | `augmentation` | `true` | Enable train-time augmentation |
| `multimodal` | `freeze_encoders` | `false` | Freeze both encoders |
| `clip` | `freeze_encoder` | `true` | Freeze CLIP encoder |

To switch to CPU training (if no GPU is available), set `device: "cpu"` in `config/config.yaml`.

---

## Outputs

| Path | Contents |
|---|---|
| `checkpoints/` | `.pt` files saved each epoch and at best AUROC |
| `logs/` | Per-step and per-epoch loss/metrics for each model |
| `results/` | JSON/TXT files with final evaluation metrics |
| `presentation/` | Final report (`.docx`, `.pdf`) and slides (`.pptx`) |
