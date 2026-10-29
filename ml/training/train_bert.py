"""Fine-tuning script for BERT on Fake News Text Classification datasets.

Supports:
- CSV, TSV (e.g. Fakeddit all_train.tsv), JSONL, and JSON formats
- Automatic or custom text and label column detection
- Layer-wise freezing of bottom BERT encoder layers (conserves VRAM & speeds up training)
- Differential learning rates (backbone vs classification head)
- PyTorch AMP mixed precision (FP16) on CUDA
- Validation calibration via Temperature Scaling
- Comprehensive metrics evaluation (Accuracy, F1, Precision, Recall, AUC, Brier Score, ECE)
- Exports checkpoint ready for BertInferencePipeline and backend inference service
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertTokenizer

from ml.calibration.metrics import compute_comprehensive_metrics
from ml.calibration.temperature import TemperatureScaler
from ml.models.bert_classifier import BertNewsClassifier, BertInferencePipeline
from ml.preprocessing.text import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_bert")


class TextClassificationDataset(Dataset):
    """Loads and tokenizes text dataset from CSV / TSV / JSONL / DataFrame."""

    def __init__(
        self,
        data_source: Union[Path, pd.DataFrame],
        tokenizer: Any,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        max_seq_length: int = 128,
        num_classes: int = 2,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")

            suffix = path.suffix.lower()
            if suffix == ".jsonl":
                df = pd.read_json(path, lines=True)
            elif suffix == ".json":
                df = pd.read_json(path)
            elif suffix in (".tsv", ".txt"):
                df = pd.read_csv(path, sep="\t", low_memory=False)
            else:
                df = pd.read_csv(path, low_memory=False)

        cols = list(df.columns)

        # 1. Detect Text Column
        chosen_text_col = text_col
        if not chosen_text_col:
            candidates = [
                "clean_title",
                "title",
                "post_text",
                "text",
                "content",
                "headline",
                "claim",
                "statement",
                "body",
                "article",
            ]
            for c in candidates:
                if c in cols:
                    chosen_text_col = c
                    break
            if not chosen_text_col:
                # Fallback: check first object/string column
                for c in cols:
                    if df[c].dtype == object:
                        chosen_text_col = c
                        break

        if not chosen_text_col or chosen_text_col not in cols:
            raise ValueError(f"Could not find valid text column in columns: {cols}. Specify via --text_col.")

        # 2. Detect Label Column
        chosen_label_col = label_col
        if not chosen_label_col:
            if num_classes == 6 and "6_way_label" in cols:
                chosen_label_col = "6_way_label"
            elif num_classes == 2 and "2_way_label" in cols:
                chosen_label_col = "2_way_label"
            else:
                candidates = [
                    "2_way_label",
                    "label",
                    "target",
                    "fake",
                    "is_fake",
                    "class",
                    "6_way_label",
                    "category",
                ]
                for c in candidates:
                    if c in cols:
                        chosen_label_col = c
                        break

        if not chosen_label_col or chosen_label_col not in cols:
            raise ValueError(f"Could not find valid label column in columns: {cols}. Specify via --label_col.")

        logger.info(
            "Selected text column: '%s' | label column: '%s' (Total raw rows: %d)",
            chosen_text_col,
            chosen_label_col,
            len(df),
        )

        # 3. Clean and map samples
        self.texts: List[str] = []
        self.labels: List[int] = []

        for _, row in df.iterrows():
            raw_t = row[chosen_text_col]
            if pd.isna(raw_t):
                continue

            cleaned_t = clean_text(str(raw_t).strip())
            if not cleaned_t:
                continue

            raw_l = row[chosen_label_col]
            if pd.isna(raw_l):
                continue

            # Standardize label to int
            if isinstance(raw_l, str):
                s = raw_l.lower().strip()
                if s in ("fake", "false", "manipulated", "1"):
                    lbl = 1
                elif s in ("real", "true", "authentic", "0"):
                    lbl = 0
                else:
                    try:
                        lbl = int(float(s))
                    except ValueError:
                        continue
            else:
                lbl = int(raw_l)

            # Validate range for num_classes
            if 0 <= lbl < num_classes:
                self.texts.append(cleaned_t)
                self.labels.append(lbl)

            if max_samples and len(self.texts) >= max_samples:
                break

        from collections import Counter
        class_counts = dict(sorted(Counter(self.labels).items()))
        logger.info(
            "Successfully indexed %d valid samples | Class distribution: %s",
            len(self.texts),
            class_counts,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return item


def train_bert_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    pretrained_model_name: str = "bert-base-uncased",
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    freeze_layers: int = 6,
    num_classes: int = 2,
    max_seq_length: int = 128,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> Path:
    """Executes BERT fine-tuning loop, evaluations, and saves calibrated checkpoint."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", dev)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate model
    model = BertNewsClassifier(
        pretrained_model_name=pretrained_model_name,
        num_classes=num_classes,
        freeze_encoder_layers=freeze_layers,
    )
    model.to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Total params: %d | Trainable params: %d (%.2f%%) [Bottom %d encoder layers frozen]",
        total_params,
        trainable_params,
        100.0 * trainable_params / total_params,
        freeze_layers,
    )

    # 2. Differential learning rate
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
            {"params": head_params, "lr": lr * 4.0},
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

            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels = batch["label"].to(dev)

            with torch.amp.autocast(device_type=dev.type, enabled=(scaler is not None)):
                logits = model(input_ids, attention_mask)
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

            if step % 25 == 0 or step == len(train_loader):
                logger.info(
                    "Epoch [%d/%d] Step [%d/%d] - Average Loss: %.4f",
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
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                labels = batch["label"].cpu().numpy()

                with torch.amp.autocast(device_type=dev.type, enabled=(scaler is not None)):
                    logits = model(input_ids, attention_mask)

                all_logits.append(logits.cpu().numpy())
                all_labels.extend(labels)

        raw_logits = np.vstack(all_logits)
        y_true = np.array(all_labels)

        # Compute metrics
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
            "Epoch %d Validation -> Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ECE: %.4f",
            epoch,
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1_score"],
            val_metrics.get("ece", 0.0),
        )

        # Track and checkpoint best model based on F1
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            logger.info("--> New best validation F1 (%.4f)! Saving weights to %s", best_val_f1, best_weights_path)
            torch.save(model.state_dict(), best_weights_path)

            # Fit post-hoc Temperature Calibration on validation logits
            calibrated_temp = 1.0
            if num_classes == 2:
                scaler_calib = TemperatureScaler()
                scaler_calib.fit(raw_logits, y_true)
                calibrated_temp = float(scaler_calib.temperature)
                logger.info("--> Optimal calibrated temperature: %.4f", calibrated_temp)

            # Save Tokenizer files into checkpoint directory
            if tokenizer is not None:
                tokenizer.save_pretrained(str(output_dir))

            label_map = (
                {0: "true", 1: "satire", 2: "misleading", 3: "manipulated", 4: "false_context", 5: "fabricated"}
                if num_classes == 6
                else {0: "real", 1: "fake"}
            )
            # Save Metadata
            meta = {
                "model_name": "bert",
                "pretrained_model_name": pretrained_model_name,
                "num_classes": num_classes,
                "label_map": label_map,
                "max_seq_length": max_seq_length,
                "freeze_layers": freeze_layers,
                "temperature": calibrated_temp,
                "best_val_metrics": val_metrics,
            }
            with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    logger.info("Training complete! Best Checkpoint saved in %s", output_dir)
    return best_weights_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on Fake News Text Classification")
    parser.add_argument("--data_file", type=str, default="data/fakeddit/all_train.tsv", help="Path to input CSV / TSV / JSONL file")
    parser.add_argument("--val_file", type=str, default=None, help="Optional separate validation CSV / TSV file")
    parser.add_argument("--text_col", type=str, default=None, help="Column name containing text/headline")
    parser.add_argument("--label_col", type=str, default=None, help="Column name containing class label (0/1 or categorical)")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (2 for binary real/fake, 6 for Fakeddit 6-way)")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased", help="Hugging Face model backbone")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum token sequence length")
    parser.add_argument("--freeze_layers", type=int, default=6, help="Bottom encoder layers to freeze (0-12, default 6)")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional row limit for quick training/experimentation")
    parser.add_argument("--output_dir", type=str, default="checkpoints/bert_finetuned", help="Directory where checkpoint is saved")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (e.g. 16 or 32)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for BERT backbone")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split fraction if val_file not provided")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")

    args = parser.parse_args()

    data_path = Path(args.data_file)
    out_path = Path(args.output_dir)

    if not data_path.is_file():
        logger.error("Dataset not found at '%s'. Please provide a valid file path.", data_path)
        exit(1)

    logger.info("Initializing tokenizer '%s'...", args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    if args.val_file and Path(args.val_file).is_file():
        train_ds = TextClassificationDataset(
            data_source=data_path,
            tokenizer=tokenizer,
            text_col=args.text_col,
            label_col=args.label_col,
            max_seq_length=args.max_seq_length,
            num_classes=args.num_classes,
            max_samples=args.max_samples,
        )
        val_ds = TextClassificationDataset(
            data_source=Path(args.val_file),
            tokenizer=tokenizer,
            text_col=args.text_col,
            label_col=args.label_col,
            max_seq_length=args.max_seq_length,
            num_classes=args.num_classes,
            max_samples=args.max_samples,
        )
    else:
        full_ds = TextClassificationDataset(
            data_source=data_path,
            tokenizer=tokenizer,
            text_col=args.text_col,
            label_col=args.label_col,
            max_seq_length=args.max_seq_length,
            num_classes=args.num_classes,
            max_samples=args.max_samples,
        )
        val_size = max(1, int(len(full_ds) * args.val_split))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info("Dataset split into %d training samples and %d validation samples.", train_size, val_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    train_bert_model(
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=out_path,
        pretrained_model_name=args.pretrained_model,
        epochs=args.epochs,
        lr=args.lr,
        freeze_layers=args.freeze_layers,
        num_classes=args.num_classes,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
