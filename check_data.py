import pandas as pd
from PIL import Image
import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = os.path.join(DATA_ROOT, "train.jsonl")

train_df = pd.read_json(TRAIN_FILE, lines=True)
print(train_df.head())
print(train_df.columns)
print(train_df.shape)

sample_img_path = os.path.join(DATA_ROOT, train_df.iloc[0]["img"])
print("Image path:", sample_img_path)
print("Exists:", os.path.exists(sample_img_path))

if os.path.exists(sample_img_path):
    img = Image.open(sample_img_path)
    print("Image size:", img.size)
else:
    print("Image not found — run download_dataset.py first.")
