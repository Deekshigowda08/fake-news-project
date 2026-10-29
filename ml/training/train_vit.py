"""Fine-tuning script for Vision Transformer (ViT) on Fake News / Image Manipulation datasets.

Supports:
- ImageFolder directory format (dataset/train/real, dataset/train/fake)
- CSV / JSONL format with image_path and label
- Layer-wise unfreezing of top transformer blocks
- PyTorch AMP mixed precision (ideal for RTX 3050 and modern GPUs)
- Validation calibration and model export compatible with ViTInferencePipeline
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import ViTImageProcessor

from ml.models.vit_classifier import ViTNewsClassifier, ViTInferencePipeline
from ml.preprocessing.image import get_vit_transform
from ml.calibration.metrics import compute_comprehensive_metrics
from ml.calibration.temperature import TemperatureScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_vit")


class CSVImageDataset(Dataset):
    """Loads image dataset from CSV/JSONL/TSV containing image paths/IDs and labels."""

    def __init__(
        self,
        data_file: Path,
        images_dir: Optional[Path] = None,
        label_col: Optional[str] = None,
        num_classes: int = 2,
        transform=None,
    ):
        import pandas as pd
        from PIL import Image

        self.transform = transform or get_vit_transform()
        suffix = data_file.suffix.lower()
        if suffix == ".jsonl":
            df = pd.read_json(data_file, lines=True)
        elif suffix in (".tsv", ".txt"):
            df = pd.read_csv(data_file, sep="\t")
        else:
            df = pd.read_csv(data_file)

        cols = list(df.columns)
        target_lbl_col = label_col
        if not target_lbl_col:
            if "label" in cols:
                target_lbl_col = "label"
            elif "2_way_label" in cols and num_classes == 2:
                target_lbl_col = "2_way_label"
            elif "6_way_label" in cols and num_classes == 6:
                target_lbl_col = "6_way_label"
            elif "2_way_label" in cols:
                target_lbl_col = "2_way_label"
            elif "6_way_label" in cols:
                target_lbl_col = "6_way_label"
            else:
                raise ValueError(f"Could not determine label column from available columns: {cols}")

        img_dir_path = Path(images_dir) if images_dir else data_file.parent
        self.samples = []

        for _, row in df.iterrows():
            lbl_raw = row[target_lbl_col]
            if pd.isna(lbl_raw):
                continue

            # Support 0/1, real/fake strings, or integer categorical classes
            if isinstance(lbl_raw, str):
                lbl_str = lbl_raw.lower().strip()
                if lbl_str in ("fake", "manipulated", "1", "true"):
                    lbl = 1
                elif lbl_str in ("real", "0", "false"):
                    lbl = 0
                else:
                    try:
                        lbl = int(float(lbl_raw))
                    except ValueError:
                        continue
            else:
                lbl = int(lbl_raw)

            # Locate image path
            img_path = None
            if "image_path" in row and pd.notna(row["image_path"]):
                cand = Path(str(row["image_path"]))
                if cand.is_file():
                    img_path = str(cand)
                elif (img_dir_path / cand.name).is_file():
                    img_path = str(img_dir_path / cand.name)

            if not img_path:
                # Try Fakeddit style 'id' or 'image_id'
                for id_col in ("id", "image_id"):
                    if id_col in row and pd.notna(row[id_col]):
                        raw_id = str(row[id_col]).strip()
                        for ext in (".jpg", ".jpeg", ".png", ".webp"):
                            cand = img_dir_path / f"{raw_id}{ext}"
                            if cand.is_file():
                                img_path = str(cand)
                                break
                            for sub in ("images", "images_train", "images_val", "images_test"):
                                cand_sub = img_dir_path / sub / f"{raw_id}{ext}"
                                if cand_sub.is_file():
                                    img_path = str(cand_sub)
                                    break
                            if img_path:
                                break
                        if img_path:
                            break

            if img_path and Path(img_path).is_file():
                self.samples.append((img_path, lbl))

        logger.info(
            "Loaded %d valid image samples (out of %d rows) from %s using label column '%s'",
            len(self.samples),
            len(df),
            data_file,
            target_lbl_col,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return {"image": tensor, "label": torch.tensor(label, dtype=torch.long)}



def get_imagefolder_loaders(
    data_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Builds train and val loaders from ImageFolder structure."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not val_dir.exists():
        val_dir = data_dir / "test"

    transform = get_vit_transform()

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=transform)

    logger.info("ImageFolder train classes: %s (%d samples)", train_ds.class_to_idx, len(train_ds))
    logger.info("ImageFolder val classes: %s (%d samples)", val_ds.class_to_idx, len(val_ds))

    def collate_fn(batch):
        images, labels = zip(*batch)
        return {
            "image": torch.stack(images),
            "label": torch.tensor(labels, dtype=torch.long),
        }

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    return train_loader, val_loader


def train_vit_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    unfreeze_layers: int = 4,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
):
    """Executes ViT fine-tuning with validation checks and calibration."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Target device: %s", dev)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate model with partial layer unfreezing
    model = ViTNewsClassifier(
        pretrained_model_name="google/vit-base-patch16-224",
        num_classes=num_classes,
        freeze_encoder=True,
        unfreeze_last_n_layers=unfreeze_layers,
    )
    model.to(dev)

    # Log trainable parameter breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Total params: %d | Trainable params: %d (%.2f%%) [Top %d layers unfrozen]",
        total_params,
        trainable_params,
        100.0 * trainable_params / total_params,
        unfreeze_layers,
    )

    # Differential learning rate: lower lr for backbone, slightly higher for classification head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr},
            {"params": head_params, "lr": lr * 3.0},
        ],
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if (use_amp and dev.type == "cuda") else None

    best_val_f1 = -1.0
    best_weights_path = output_dir / "pytorch_model.bin"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = batch["image"].to(dev)
            labels = batch["label"].to(dev)

            with torch.amp.autocast(device_type=dev.type, enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()

            if step % 20 == 0 or step == len(train_loader):
                logger.info(
                    "Epoch [%d/%d] Step [%d/%d] - Loss: %.4f",
                    epoch,
                    epochs,
                    step,
                    len(train_loader),
                    running_loss / step,
                )

        # Validation phase
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(dev)
                lbls = batch["label"].cpu().numpy()
                with torch.amp.autocast(device_type=dev.type, enabled=(scaler is not None)):
                    logits = model(images)
                all_logits.append(logits.cpu().numpy())
                all_labels.extend(lbls)

        raw_logits = np.vstack(all_logits)
        y_true = np.array(all_labels)

        # Probabilities & metrics
        exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        if num_classes == 2:
            y_prob_fake = probs[:, 1]
            y_pred = (y_prob_fake >= 0.5).astype(int)
            val_metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob_fake)
        else:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            y_pred = np.argmax(probs, axis=-1)
            val_metrics = {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "f1_score": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
                "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
                "brier_score": 0.0,
                "ece": 0.0,
            }

        logger.info(
            "Epoch %d Val Results -> Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | Brier: %.4f | ECE: %.4f",
            epoch,
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1_score"],
            val_metrics["brier_score"],
            val_metrics["ece"],
        )

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            logger.info("--> New best validation F1 (%.4f)! Saving weights to %s", best_val_f1, best_weights_path)
            torch.save(model.state_dict(), best_weights_path)

            # Fit temperature scaling calibration on validation logits
            calibrated_temp = 1.0
            if num_classes == 2:
                scaler_calib = TemperatureScaler()
                val_tensor_logits = torch.tensor(raw_logits, dtype=torch.float32)
                val_tensor_labels = torch.tensor(y_true, dtype=torch.long)
                calibrated_temp = scaler_calib.fit(val_tensor_logits, val_tensor_labels)

            # Save processor and metadata
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            processor.save_pretrained(str(output_dir))

            meta = {
                "model_name": "vit",
                "pretrained_model_name": "google/vit-base-patch16-224",
                "num_classes": num_classes,
                "unfreeze_layers": unfreeze_layers,
                "temperature": float(calibrated_temp),
                "best_val_metrics": val_metrics,
            }
            with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    logger.info("Training complete! Best Checkpoint saved at %s", output_dir)
    return best_weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ViT on Fake News / Image Manipulation / Fakeddit dataset")
    parser.add_argument("--data_dir", type=str, default="data/image_benchmark", help="Directory with train/val subfolders")
    parser.add_argument("--csv_train", type=str, default=None, help="Optional CSV/JSONL file for training")
    parser.add_argument("--csv_val", type=str, default=None, help="Optional CSV/JSONL file for validation")
    parser.add_argument("--fakeddit_train", type=str, default=None, help="Path to Fakeddit train TSV/CSV")
    parser.add_argument("--fakeddit_val", type=str, default=None, help="Path to Fakeddit val TSV/CSV")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory containing images (<id>.jpg)")
    parser.add_argument("--label_col", type=str, default=None, help="Column name for labels (e.g. 2_way_label, 6_way_label)")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of target classes (2 for binary, 6 for Fakeddit 6-way)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/vit_finetuned", help="Output directory for saved weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (reduce to 8 if VRAM is limited)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for ViT backbone")
    parser.add_argument("--unfreeze_layers", type=int, default=4, help="Number of top transformer layers to unfreeze (1-12)")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)

    # 1. Direct Fakeddit TSV/CSV support
    train_file = args.fakeddit_train or args.csv_train
    val_file = args.fakeddit_val or args.csv_val

    if train_file and Path(train_file).is_file():
        img_dir = Path(args.images_dir) if args.images_dir else Path(train_file).parent
        train_ds = CSVImageDataset(
            data_file=Path(train_file),
            images_dir=img_dir,
            label_col=args.label_col,
            num_classes=args.num_classes,
        )
        if val_file and Path(val_file).is_file():
            val_ds = CSVImageDataset(
                data_file=Path(val_file),
                images_dir=img_dir,
                label_col=args.label_col,
                num_classes=args.num_classes,
            )
        else:
            # Simple 90/10 split if separate val file not provided
            val_size = max(1, int(len(train_ds) * 0.1))
            train_size = len(train_ds) - val_size
            train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

        t_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        v_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    elif data_path.exists():
        t_loader, v_loader = get_imagefolder_loaders(data_path, batch_size=args.batch_size)
    else:
        logger.error(
            "Dataset not found at '%s'. Provide valid --fakeddit_train, --csv_train, or --data_dir.",
            data_path,
        )
        exit(1)

    train_vit_model(
        train_loader=t_loader,
        val_loader=v_loader,
        output_dir=out_path,
        epochs=args.epochs,
        lr=args.lr,
        unfreeze_layers=args.unfreeze_layers,
        num_classes=args.num_classes,
        use_amp=not args.no_amp,
    )
