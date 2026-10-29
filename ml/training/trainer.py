"""Training and evaluation routines for PyTorch models in the multimodal fake news pipeline."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.calibration.metrics import compute_comprehensive_metrics
from ml.calibration.temperature import TemperatureScaler

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_multimodal: bool = False,
    is_text: bool = True,
    is_image: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Trains a single epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    use_amp = scaler is not None and device.type == "cuda"

    for batch in dataloader:
        optimizer.zero_grad()
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if is_text and not is_multimodal:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
            elif is_image and not is_multimodal:
                pixel_values = batch["image"].to(device)
                logits = model(pixel_values)
            else:
                # Multimodal forward
                input_ids = batch.get("input_ids", None)
                if input_ids is not None:
                    input_ids = input_ids.to(device)
                    attention_mask = batch["attention_mask"].to(device)
                else:
                    attention_mask = None
                image = batch["image"].to(device)
                has_image = batch["has_image"].to(device)
                logits = model(input_ids, attention_mask, image, has_image)

            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_multimodal: bool = False,
    is_text: bool = True,
    is_image: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluates model on dataloader and calculates full metrics.
    
    Returns:
        (metrics_dict, raw_logits, ground_truth_labels)
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].numpy()
            if is_text and not is_multimodal:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
            elif is_image and not is_multimodal:
                pixel_values = batch["image"].to(device)
                logits = model(pixel_values)
            else:
                input_ids = batch.get("input_ids", None)
                if input_ids is not None:
                    input_ids = input_ids.to(device)
                    attention_mask = batch["attention_mask"].to(device)
                else:
                    attention_mask = None
                image = batch["image"].to(device)
                has_image = batch["has_image"].to(device)
                logits = model(input_ids, attention_mask, image, has_image)

            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels)

    raw_logits = np.vstack(all_logits)
    y_true = np.array(all_labels)

    # Compute softmax probabilities
    exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    y_prob_positive = probs[:, 1]
    y_pred = (y_prob_positive >= 0.5).astype(int)

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob_positive)
    return metrics, raw_logits, y_true
