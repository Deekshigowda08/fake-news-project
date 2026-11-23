import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_MAP = {"real": 0, "fake": 1, 0: 0, 1: 1}


def tokenize_text(text: str):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return [t for t in text.split() if t]


def build_vocab(texts, max_vocab=50000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize_text(t))
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


class MultimodalFakeNewsDataset(Dataset):
    """Dataset providing tokenized IDs for CNN-LSTM and image tensors."""

    def __init__(
        self,
        csv_path,
        image_dirs,
        vocab=None,
        max_len=128,
    ):
        self.csv_path = Path(csv_path)
        self.image_dirs = [Path(p) for p in image_dirs]
        self.max_len = max_len

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.data = pd.read_csv(self.csv_path).dropna(subset=["post_text", "label"])
        self.data = self.data[self.data["label"].isin(LABEL_MAP)].reset_index(drop=True)

        if vocab is None:
            self.vocab = build_vocab(self.data["post_text"].tolist())
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def text_to_ids(self, text: str):
        tokens = tokenize_text(text)[:self.max_len]
        ids = [self.vocab.get(t, 1) for t in tokens]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def resolve_image_path(self, image_id):
        if pd.isna(image_id):
            return None
        candidates = [f"{image_id}.jpg", f"{image_id}.jpeg", f"{image_id}.png", f"{image_id}.gif"]
        for img_dir in self.image_dirs:
            for c in candidates:
                p = img_dir / c
                if p.exists():
                    return p
        return None

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_ids = self.text_to_ids(str(row["post_text"]))

        img_path = self.resolve_image_path(row.get("image_id"))
        if img_path is not None:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
                has_image = 1
            except Exception:
                img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
                has_image = 0
        else:
            img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
            has_image = 0

        raw_label = row["label"]
        label = LABEL_MAP[raw_label]

        return {
            "input_ids": input_ids,
            "image": img_tensor,
            "has_image": torch.tensor(has_image, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "post_id": row.get("post_id", ""),
        }


def summarize_csv(csv_path):
    data = pd.read_csv(csv_path)
    label_counts = data["label"].value_counts(dropna=False).to_dict()
    missing = data.isna().sum().to_dict()
    return {
        "rows": len(data),
        "columns": list(data.columns),
        "label_counts": label_counts,
        "missing_required_fields": missing,
    }
