"""Generates a reproducible multimodal benchmark dataset for research and testing.

Includes realistic news headlines, fabricated claims, and generated sample images
to allow automated CI testing, baseline validation, and sanity checks without
requiring manual downloading of multi-gigabyte external datasets.
"""

from pathlib import Path
import json
import random
from PIL import Image, ImageDraw, ImageFont


BENCHMARK_SAMPLES = [
    # REAL NEWS SAMPLES
    {
        "sample_id": "real_001",
        "text": "NASA's James Webb Space Telescope reveals detailed atmospheric composition of a distant exoplanet.",
        "label": 0,
        "split": "train",
        "image_theme": "space",
    },
    {
        "sample_id": "real_002",
        "text": "World Health Organization announces updated international guidelines on seasonal influenza vaccination.",
        "label": 0,
        "split": "train",
        "image_theme": "medical",
    },
    {
        "sample_id": "real_003",
        "text": "European Central Bank holds benchmark interest rates steady amid moderating inflation metrics.",
        "label": 0,
        "split": "train",
        "image_theme": "finance",
    },
    {
        "sample_id": "real_004",
        "text": "Renewable energy capacity expanded by a record 50% globally over the previous calendar year.",
        "label": 0,
        "split": "train",
        "image_theme": "energy",
    },
    {
        "sample_id": "real_005",
        "text": "Archaeologists discover well-preserved Roman mosaic floor beneath a modern construction site in Verona.",
        "label": 0,
        "split": "val",
        "image_theme": "history",
    },
    {
        "sample_id": "real_006",
        "text": "Global seismology networks record magnitude 6.2 offshore earthquake with no immediate tsunami threat.",
        "label": 0,
        "split": "test",
        "image_theme": "nature",
    },
    {
        "sample_id": "real_007",
        "text": "National Weather Service issues severe thunderstorm warning for eastern coastal counties.",
        "label": 0,
        "split": "test",
        "image_theme": "weather",
    },
    {
        "sample_id": "real_008",
        "text": "Public transit agency unveils battery-electric bus fleet scheduled to begin operations on Monday.",
        "label": 0,
        "split": "train",
        "image_theme": "transport",
    },

    # FAKE NEWS SAMPLES
    {
        "sample_id": "fake_001",
        "text": "Secret Antarctic pyramid discovered housing alien energy crystals confirmed by unclassified defense files.",
        "label": 1,
        "split": "train",
        "image_theme": "conspiracy",
    },
    {
        "sample_id": "fake_002",
        "text": "Drinking boiled lemon water with baking soda completely cures all stages of cancer in 48 hours.",
        "label": 1,
        "split": "train",
        "image_theme": "medical_hoax",
    },
    {
        "sample_id": "fake_003",
        "text": "Government secretly passes law banning all cash transactions starting next month to control citizenry.",
        "label": 1,
        "split": "train",
        "image_theme": "finance_hoax",
    },
    {
        "sample_id": "fake_004",
        "text": "NASA scientist reveals the Moon is hollow and artificial radio transmitters were installed inside.",
        "label": 1,
        "split": "train",
        "image_theme": "space_hoax",
    },
    {
        "sample_id": "fake_005",
        "text": "Scientists discover giant 50-foot prehistoric snake alive in the Amazon river basin captured on tourist camera.",
        "label": 1,
        "split": "val",
        "image_theme": "cryptozoology",
    },
    {
        "sample_id": "fake_006",
        "text": "5G cell towers cause sudden cloud vaporization and artificial tornadoes according to whistleblower memo.",
        "label": 1,
        "split": "test",
        "image_theme": "telecom_hoax",
    },
    {
        "sample_id": "fake_007",
        "text": "Celebrity billionaire donates 99% of wealth directly to social media followers who share this post.",
        "label": 1,
        "split": "test",
        "image_theme": "social_scam",
    },
    {
        "sample_id": "fake_008",
        "text": "Sun will completely shut down for three days in December due to cosmic alignment blackout protocol.",
        "label": 1,
        "split": "train",
        "image_theme": "solar_hoax",
    },
]


def create_theme_image(path: Path, theme: str, sample_id: str):
    """Generates a 224x224 RGB image with distinct geometric patterns and labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Deterministic palette based on theme
    random.seed(hash(theme) % 10000)
    bg_color = (
        random.randint(40, 200),
        random.randint(40, 200),
        random.randint(40, 200),
    )
    img = Image.new("RGB", (224, 224), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw shapes
    for _ in range(5):
        shape_color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
        x0 = random.randint(10, 180)
        y0 = random.randint(10, 180)
        x1 = x0 + random.randint(20, 80)
        y1 = y0 + random.randint(20, 80)
        draw.rectangle([x0, y0, x1, y1], fill=shape_color, outline=(255, 255, 255))

    # Add text label
    draw.text((10, 10), f"ID: {sample_id}", fill=(255, 255, 255))
    draw.text((10, 30), f"Theme: {theme}", fill=(240, 240, 240))

    img.save(path, format="JPEG", quality=90)


def generate_benchmark_dataset(base_dir: Path = Path("data/benchmark")) -> Path:
    """Generates train.jsonl, val.jsonl, test.jsonl and associated images."""
    base_dir.mkdir(parents=True, exist_ok=True)
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}

    for item in BENCHMARK_SAMPLES:
        sample_id = item["sample_id"]
        img_name = f"{sample_id}.jpg"
        img_path = images_dir / img_name
        
        # Create image
        create_theme_image(img_path, item["image_theme"], sample_id)

        record = {
            "sample_id": sample_id,
            "text": item["text"],
            "image_path": str(img_path.resolve()),
            "label": item["label"],
            "dataset_source": "multimodal_benchmark",
            "split": item["split"],
            "has_image": True,
            "metadata": {"theme": item["image_theme"]},
        }
        splits[item["split"]].append(record)

    for split_name, records in splits.items():
        split_file = base_dir / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    return base_dir


if __name__ == "__main__":
    out_dir = generate_benchmark_dataset()
    print(f"Generated benchmark multimodal dataset at: {out_dir}")
