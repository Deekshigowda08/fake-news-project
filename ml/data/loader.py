"""Modular dataset loader for multimodal fake news detection.

Supports multiple data formats:
1. Canonical JSONL / CSV processed schema
2. Fakeddit TSV/CSV format
3. Twitter / Weibo social media format (matching original baseline)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ml.data.schema import (
    BinaryLabel,
    FAKEDDIT_6WAY_TO_BINARY,
    STRING_LABEL_TO_BINARY,
    ProcessedSample,
)

logger = logging.getLogger(__name__)


def compute_text_hash(text: str) -> str:
    """Computes SHA256 of normalized text to detect duplicates and prevent leakage."""
    normalized = " ".join(text.lower().strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ModularNewsDataset(Dataset):
    """PyTorch Dataset for multimodal fake news detection with robust error handling.
    
    Guarantees:
    - Never throws an unhandled crash on missing or corrupted images.
    - Explicitly sets `has_image` flag so models can apply modality-specific routing.
    - Preserves data traceability with sample_id and metadata.
    """

    def __init__(
        self,
        samples: List[ProcessedSample],
        tokenizer=None,
        image_transform=None,
        max_seq_length: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 1. Text tokenization if tokenizer provided
        text_data = {}
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                sample.text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            text_data["input_ids"] = encoding["input_ids"].squeeze(0)
            text_data["attention_mask"] = encoding["attention_mask"].squeeze(0)
            if "token_type_ids" in encoding:
                text_data["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        # 2. Image loading and validation
        image_tensor = None
        has_valid_image = False
        if sample.image_path and Path(sample.image_path).is_file():
            try:
                with Image.open(sample.image_path) as img:
                    img = img.convert("RGB")
                    image_tensor = self.image_transform(img)
                    has_valid_image = True
            except Exception as e:
                logger.warning("Failed to decode image %s: %s", sample.image_path, e)
                has_valid_image = False

        if image_tensor is None:
            # Explicit zero placeholder for tensor batching, but has_image flag is False
            image_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        return {
            "sample_id": sample.sample_id,
            "text": sample.text,
            "has_image": torch.tensor(1 if has_valid_image else 0, dtype=torch.long),
            "image": image_tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            **text_data,
        }


def load_dataset_from_records(
    data_path: Union[str, Path],
    images_dir: Optional[Union[str, Path]] = None,
    dataset_type: str = "auto",
    split: str = "train",
) -> List[ProcessedSample]:
    """Inspects, cleans, and loads samples into canonical ProcessedSample objects."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Determine file type
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    samples: List[ProcessedSample] = []
    images_root = Path(images_dir) if images_dir else path.parent

    # Auto-detect schema
    columns = set(df.columns)

    for idx, row in df.iterrows():
        # Case A: Canonical schema
        if "sample_id" in columns and "text" in columns and "label" in columns:
            sample_id = str(row["sample_id"])
            text = str(row["text"])
            raw_label = row["label"]
            label = int(raw_label)
            img_path = str(row["image_path"]) if pd.notna(row.get("image_path")) else None

        # Case B: Twitter / Weibo schema (df_train2.csv, train_posts.csv)
        elif "post_text" in columns and "image_id" in columns and "label" in columns:
            sample_id = str(row.get("post_id", f"sample_{idx}"))
            text = str(row["post_text"])
            raw_label = str(row["label"]).lower().strip()
            label = STRING_LABEL_TO_BINARY.get(raw_label, 1)

            # Resolve image filename candidates
            img_id = str(row["image_id"]).strip()
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".gif"]:
                candidate = images_root / f"{img_id}{ext}"
                if candidate.is_file():
                    img_path = str(candidate)
                    break
                # Also check subdirectories images_train / images_test
                for sub in ["images_train", "images_test"]:
                    candidate_sub = images_root / sub / f"{img_id}{ext}"
                    if candidate_sub.is_file():
                        img_path = str(candidate_sub)
                        break
                if img_path:
                    break

        # Case C: Fakeddit schema
        elif "clean_title" in columns or "title" in columns:
            sample_id = str(row.get("id", f"fakeddit_{idx}"))
            text = str(row.get("clean_title", row.get("title", "")))
            
            # Fakeddit 2-way vs 6-way label handling
            if "2_way_label" in columns:
                label = int(row["2_way_label"])
            elif "6_way_label" in columns:
                label = FAKEDDIT_6WAY_TO_BINARY.get(int(row["6_way_label"]), 1)
            else:
                label = int(row.get("label", 1))

            img_id = str(row.get("image_id", sample_id))
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = images_root / f"{img_id}{ext}"
                if candidate.is_file():
                    img_path = str(candidate)
                    break

        else:
            continue

        if not text.strip() or pd.isna(text):
            continue

        has_img = bool(img_path and Path(img_path).is_file())

        samples.append(
            ProcessedSample(
                sample_id=sample_id,
                text=text.strip(),
                image_path=img_path if has_img else None,
                label=label,
                dataset_source=dataset_type,
                split=split,
                has_image=has_img,
                metadata={"original_row_index": idx},
            )
        )

    return samples


def analyze_dataset_statistics(samples: List[ProcessedSample]) -> Dict[str, Any]:
    """Performs deep dataset analysis: label distribution, missing image rate, duplicates."""
    total = len(samples)
    if total == 0:
        return {"total_samples": 0}

    real_count = sum(1 for s in samples if s.label == BinaryLabel.REAL.value)
    fake_count = sum(1 for s in samples if s.label == BinaryLabel.FAKE.value)
    with_image = sum(1 for s in samples if s.has_image)

    text_hashes = [compute_text_hash(s.text) for s in samples]
    unique_texts = len(set(text_hashes))
    duplicate_texts = total - unique_texts

    return {
        "total_samples": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "real_ratio": round(real_count / total, 4),
        "fake_ratio": round(fake_count / total, 4),
        "samples_with_image": with_image,
        "samples_without_image": total - with_image,
        "missing_image_frequency": round((total - with_image) / total, 4),
        "duplicate_or_near_duplicate_texts": duplicate_texts,
        "unique_text_count": unique_texts,
    }
