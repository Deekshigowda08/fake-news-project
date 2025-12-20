from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


LABEL_MAP = {"real": 0, "fake": 1}


class MultimodalFakeNewsDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dirs,
        max_len=128,
        tokenizer_name="bert-base-uncased",
    ):
        self.csv_path = Path(csv_path)
        self.image_dirs = [Path(path) for path in image_dirs]
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.data = pd.read_csv(self.csv_path).dropna(
            subset=["post_text", "image_id", "label"]
        )
        self.data = self.data[self.data["label"].isin(LABEL_MAP)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = self.resolve_image_path(str(row["image_id"]).strip())

        encoding = self.tokenizer(
            str(row["post_text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        image = Image.open(image_path).convert("RGB")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": self.transform(image),
            "label": torch.tensor(LABEL_MAP[row["label"]], dtype=torch.long),
            "image_id": row["image_id"],
            "post_id": row.get("post_id", ""),
        }

    def resolve_image_path(self, image_id):
        candidates = [f"{image_id}.jpg", f"{image_id}.jpeg", f"{image_id}.png", f"{image_id}.gif"]
        for image_dir in self.image_dirs:
            for filename in candidates:
                image_path = image_dir / filename
                if image_path.exists():
                    return image_path
        raise FileNotFoundError(f"Image not found for image_id={image_id}")


def summarize_csv(csv_path):
    data = pd.read_csv(csv_path)
    label_counts = data["label"].value_counts(dropna=False).to_dict()
    missing = data[["post_text", "image_id", "label"]].isna().sum().to_dict()
    return {
        "rows": len(data),
        "columns": list(data.columns),
        "label_counts": label_counts,
        "missing_required_fields": missing,
        "unique_images": data["image_id"].nunique(dropna=True),
    }
