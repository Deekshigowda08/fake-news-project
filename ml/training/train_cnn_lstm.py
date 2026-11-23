"""
train_cnn_lstm.py
=================
Training script for the TextCNNLSTMClassifier (CNN + LSTM hybrid) used in the
fake-news-detection pipeline.

Usage
-----
    python -m ml.training.train_cnn_lstm \\
        --data_file data/fakeddit/all_train.tsv \\
        --num_classes 2 \\
        --epochs 10 \\
        --batch_size 64 \\
        --output_dir checkpoints/cnn_lstm \\
        --glove_path glove.6B.100d.txt

The script:
  1. Loads and preprocesses a dataset (CSV / TSV / JSONL or pd.DataFrame).
  2. Builds a word vocabulary from the training corpus.
  3. Optionally loads GloVe embeddings to initialise the embedding layer.
  4. Trains TextCNNLSTMClassifier with Adam + ReduceLROnPlateau + early-stopping.
  5. Saves the best checkpoint, vocabulary and metadata to *output_dir*.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAD_IDX = 0
UNK_IDX = 1

_TEXT_COL_CANDIDATES: List[str] = [
    "clean_title", "title", "post_text", "text", "content", "headline"
]
_LABEL_COL_CANDIDATES: List[str] = [
    "2_way_label", "label", "target", "fake", "is_fake", "class"
]

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Lowercase and strip punctuation from *text*, returning a cleaned string.

    Parameters
    ----------
    text:
        Raw input string.

    Returns
    -------
    str
        Cleaned string with only alphabetic tokens separated by single spaces.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    """Split a cleaned string into whitespace-delimited tokens.

    Parameters
    ----------
    text:
        Pre-cleaned string.

    Returns
    -------
    List[str]
        List of tokens.
    """
    return text.split()


def _encode(
    tokens: List[str],
    word2idx: Dict[str, int],
    max_seq_length: int,
) -> List[int]:
    """Map tokens to integer IDs, truncate or pad to *max_seq_length*.

    Unknown tokens are mapped to UNK_IDX (1).  Padding uses PAD_IDX (0).

    Parameters
    ----------
    tokens:
        Pre-tokenised text.
    word2idx:
        Vocabulary mapping token → integer index.
    max_seq_length:
        Fixed output length.

    Returns
    -------
    List[int]
        Integer sequence of exactly *max_seq_length* elements.
    """
    ids = [word2idx.get(t, UNK_IDX) for t in tokens[:max_seq_length]]
    ids += [PAD_IDX] * (max_seq_length - len(ids))
    return ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CNNLSTMDataset(Dataset):
    """PyTorch Dataset for the CNN+LSTM text classifier.

    Accepts a file path (CSV / TSV / JSONL) **or** a pre-loaded
    ``pd.DataFrame``.  Text and label columns are auto-detected when not
    supplied explicitly.

    Parameters
    ----------
    data_source:
        Path to a data file *or* a ``pd.DataFrame``.
    word2idx:
        Vocabulary mapping token → integer index.  PAD=0, UNK=1.
    text_col:
        Name of the column containing the raw text.  Detected automatically
        from :data:`_TEXT_COL_CANDIDATES` when *None*.
    label_col:
        Name of the column containing integer labels.  Detected automatically
        from :data:`_LABEL_COL_CANDIDATES` when *None*.
    max_seq_length:
        Every sequence is truncated / padded to this length.
    num_classes:
        Number of target classes (used only for validation).
    max_samples:
        If set, subsample this many rows from the dataset (random, no seed).
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        word2idx: Dict[str, int],
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        max_seq_length: int = 300,
        num_classes: int = 2,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

        # ---- load dataframe ------------------------------------------------
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            df = self._load_file(Path(data_source))

        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples).reset_index(drop=True)
            logger.info("Subsampled to %d rows.", max_samples)

        # ---- auto-detect columns -------------------------------------------
        cols = [c.lower() for c in df.columns]
        df.columns = cols  # normalise to lower

        self.text_col = self._resolve_col(text_col, _TEXT_COL_CANDIDATES, cols, "text")
        self.label_col = self._resolve_col(label_col, _LABEL_COL_CANDIDATES, cols, "label")

        logger.info(
            "Using text_col=%r  label_col=%r  rows=%d",
            self.text_col, self.label_col, len(df),
        )

        # ---- pre-process ---------------------------------------------------
        self.texts: List[str] = df[self.text_col].fillna("").tolist()
        raw_labels = df[self.label_col].tolist()
        self.labels: List[int] = [int(lbl) for lbl in raw_labels]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> pd.DataFrame:
        """Load a CSV, TSV or JSONL file into a DataFrame.

        Parameters
        ----------
        path:
            Path to the file.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If the file extension is not recognised.
        """
        suffix = path.suffix.lower()
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            return pd.read_json(path)
        raise ValueError(f"Unsupported file format: {suffix!r}")

    @staticmethod
    def _resolve_col(
        explicit: Optional[str],
        candidates: List[str],
        actual_cols: List[str],
        role: str,
    ) -> str:
        """Return the first matching column name or raise.

        Parameters
        ----------
        explicit:
            User-supplied column name (may be *None*).
        candidates:
            Ordered list of fallback column names to try.
        actual_cols:
            Columns present in the DataFrame (already lower-cased).
        role:
            Human-readable label used in error messages.

        Returns
        -------
        str
            The resolved column name.

        Raises
        ------
        ValueError
            If no candidate is found.
        """
        if explicit is not None:
            if explicit.lower() in actual_cols:
                return explicit.lower()
            raise ValueError(
                f"Specified {role} column {explicit!r} not found in data. "
                f"Available: {actual_cols}"
            )
        for cand in candidates:
            if cand in actual_cols:
                return cand
        raise ValueError(
            f"Could not auto-detect {role} column. "
            f"Tried {candidates}. Available: {actual_cols}"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D105
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample.

        Parameters
        ----------
        idx:
            Sample index.

        Returns
        -------
        dict
            ``{"input_ids": LongTensor[max_seq_length], "label": LongTensor[]}``
        """
        raw = self.texts[idx]
        cleaned = _clean_text(raw)
        tokens = _tokenize(cleaned)
        ids = _encode(tokens, self.word2idx, self.max_seq_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Vocabulary builder  (standalone so callers can import it too)
# ---------------------------------------------------------------------------

def build_vocab_from_texts(
    texts: List[str],
    max_vocab: int = 50_000,
    min_freq: int = 2,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build a word-level vocabulary from a list of raw text strings.

    Special tokens: ``[PAD]`` → 0, ``[UNK]`` → 1.

    Parameters
    ----------
    texts:
        Iterable of raw text strings.
    max_vocab:
        Maximum vocabulary size (including special tokens).
    min_freq:
        Minimum token frequency to be included.

    Returns
    -------
    word2idx : Dict[str, int]
    idx2word : Dict[int, str]
    """
    counter: Counter = Counter()
    for text in texts:
        tokens = _tokenize(_clean_text(text))
        counter.update(tokens)

    word2idx: Dict[str, int] = {"[PAD]": PAD_IDX, "[UNK]": UNK_IDX}
    for word, freq in counter.most_common(max_vocab - 2):
        if freq < min_freq:
            break
        word2idx[word] = len(word2idx)

    idx2word: Dict[int, str] = {v: k for k, v in word2idx.items()}
    logger.info("Vocabulary size: %d tokens.", len(word2idx))
    return word2idx, idx2word


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_cnn_lstm_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Union[str, Path],
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    vocab_size: int,
    epochs: int = 10,
    lr: float = 0.001,
    embedding_dim: int = 100,
    num_filters: int = 100,
    filter_sizes: List[int] = None,
    lstm_hidden: int = 128,
    lstm_layers: int = 2,
    num_classes: int = 2,
    max_seq_length: int = 300,
    glove_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> Path:
    """Train a :class:`TextCNNLSTMClassifier` and save the best checkpoint.

    Parameters
    ----------
    train_loader:
        DataLoader yielding batches from the training split.
    val_loader:
        DataLoader yielding batches from the validation split.
    output_dir:
        Directory where the best checkpoint, vocabulary and metadata are saved.
    word2idx:
        Vocabulary mapping token → index.
    idx2word:
        Reverse mapping index → token.
    vocab_size:
        Total vocabulary size (len(word2idx)).
    epochs:
        Maximum number of training epochs.
    lr:
        Initial learning rate for Adam.
    embedding_dim:
        Embedding vector dimensionality.
    num_filters:
        Number of convolutional filters per kernel size.
    filter_sizes:
        Kernel sizes for the CNN block.  Defaults to ``[2, 3, 4, 5]``.
    lstm_hidden:
        LSTM hidden state size (per direction).
    lstm_layers:
        Number of stacked LSTM layers.
    num_classes:
        Number of output classes.
    max_seq_length:
        Maximum input sequence length.
    glove_path:
        Optional path to a GloVe text file (e.g. ``glove.6B.100d.txt``).
    device:
        Torch device.  Auto-detected when *None*.
    use_amp:
        Whether to use automatic mixed precision (requires CUDA).

    Returns
    -------
    Path
        Absolute path to the best model checkpoint file.
    """
    if filter_sizes is None:
        filter_sizes = [2, 3, 4, 5]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # ------------------------------------------------------------------ model
    # Import here to avoid circular imports when this module is used standalone
    try:
        from ml.models.cnn_lstm_classifier import (
            TextCNNLSTMClassifier,
            load_glove_embeddings,
        )
    except ImportError as exc:
        raise ImportError(
            "Cannot import TextCNNLSTMClassifier. "
            "Make sure ml/models/cnn_lstm_classifier.py exists."
        ) from exc

    model = TextCNNLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_sizes=filter_sizes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        num_classes=num_classes,
    )

    # Optionally load GloVe weights
    if glove_path is not None and Path(glove_path).is_file():
        logger.info("Loading GloVe embeddings from %s …", glove_path)
        glove_matrix = load_glove_embeddings(glove_path, word2idx, embedding_dim)
        model.embedding.weight.data.copy_(
            torch.tensor(glove_matrix, dtype=torch.float)
        )
        logger.info("GloVe embeddings loaded.")
    elif glove_path is not None:
        logger.warning("GloVe path %r not found — training from scratch.", glove_path)

    model.to(device)

    # ---------------------------------------------------------- class weights
    all_labels: List[int] = []
    for batch in train_loader:
        all_labels.extend(batch["label"].tolist())
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (num_classes * label_counts.get(c, 1)) for c in range(num_classes)],
        dtype=torch.float,
    ).to(device)
    logger.info("Class weights: %s", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -------------------------------------------------- optimiser / scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # ---------------------------------------------------------- AMP scaler
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -------------------------------------------------- training state
    best_val_f1: float = -1.0
    patience_counter: int = 0
    early_stop_patience: int = 5
    best_ckpt_path: Path = output_dir / "best_model.pt"

    history: List[Dict] = []

    # ================================================================ loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # -------------------------------------------------------- TRAIN
        model.train()
        train_loss = 0.0
        train_preds: List[int] = []
        train_targets: List[int] = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} [train]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1).cpu().tolist()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(
            train_targets, train_preds, average="weighted", zero_division=0
        )

        # -------------------------------------------------------- VAL
        model.eval()
        val_loss = 0.0
        val_preds: List[int] = []
        val_targets: List[int] = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{epochs} [val]",
                leave=False,
                dynamic_ncols=True,
            ):
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(input_ids)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().tolist())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(
            val_targets, val_preds, average="weighted", zero_division=0
        )
        val_precision = precision_score(
            val_targets, val_preds, average="weighted", zero_division=0
        )
        val_recall = recall_score(
            val_targets, val_preds, average="weighted", zero_division=0
        )

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %d/%d | "
            "train_loss=%.4f train_acc=%.4f train_f1=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_f1=%.4f val_prec=%.4f val_rec=%.4f | "
            "time=%.1fs",
            epoch, epochs,
            train_loss, train_acc, train_f1,
            val_loss, val_acc, val_f1, val_precision, val_recall,
            elapsed,
        )

        scheduler.step(val_f1)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
        }
        history.append(epoch_record)

        # -------------------------------------------------- checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": {
                        "vocab_size": vocab_size,
                        "embedding_dim": embedding_dim,
                        "num_filters": num_filters,
                        "filter_sizes": filter_sizes,
                        "lstm_hidden": lstm_hidden,
                        "lstm_layers": lstm_layers,
                        "num_classes": num_classes,
                        "max_seq_length": max_seq_length,
                    },
                },
                best_ckpt_path,
            )
            logger.info("✓ New best val_f1=%.4f — checkpoint saved.", val_f1)
        else:
            patience_counter += 1
            logger.info(
                "No improvement. Early-stop patience: %d/%d.",
                patience_counter, early_stop_patience,
            )
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping triggered after epoch %d.", epoch)
                break

    # ---------------------------------------------------------------- vocab
    vocab_path = output_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as fh:
        json.dump({"word2idx": word2idx, "idx2word": {str(k): v for k, v in idx2word.items()}}, fh)
    logger.info("Vocabulary saved to %s", vocab_path)

    # ------------------------------------------------------------- metadata
    metadata = {
        "model_name": "cnn_lstm",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_filters": num_filters,
        "filter_sizes": filter_sizes,
        "lstm_hidden": lstm_hidden,
        "lstm_layers": lstm_layers,
        "num_classes": num_classes,
        "max_seq_length": max_seq_length,
        "best_val_f1": float(best_val_f1),
        "epochs_trained": len(history),
        "training_history": history,
    }
    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    logger.info("Training complete. Best checkpoint: %s", best_ckpt_path)
    return best_ckpt_path.resolve()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse arguments, build vocab, create DataLoaders, train.

    Command-line arguments
    ----------------------
    --data_file : str
        Path to the training data file (TSV / CSV / JSONL).
        Default: ``data/fakeddit/all_train.tsv``
    --val_file : str, optional
        Path to a separate validation file.  When omitted, *val_split* of
        the training data is held out.
    --text_col : str, optional
        Name of the text column.  Auto-detected when not provided.
    --label_col : str, optional
        Name of the label column.  Auto-detected when not provided.
    --num_classes : int
        Number of target classes.  Default: 2.
    --max_seq_length : int
        Sequence length for padding / truncation.  Default: 300.
    --max_vocab : int
        Maximum vocabulary size.  Default: 50 000.
    --min_freq : int
        Minimum token frequency to include in vocab.  Default: 2.
    --glove_path : str, optional
        Path to a GloVe text file (e.g. ``glove.6B.100d.txt``).
    --embedding_dim : int
        Embedding dimensionality.  Default: 100.
    --num_filters : int
        Number of CNN filters per kernel size.  Default: 100.
    --lstm_hidden : int
        LSTM hidden units (per direction).  Default: 128.
    --lstm_layers : int
        Number of stacked LSTM layers.  Default: 2.
    --epochs : int
        Maximum training epochs.  Default: 10.
    --batch_size : int
        Mini-batch size.  Default: 64.
    --lr : float
        Initial learning rate.  Default: 0.001.
    --output_dir : str
        Directory for checkpoints and artefacts.  Default: ``checkpoints/cnn_lstm``.
    --val_split : float
        Fraction of training data used for validation (when --val_file is not
        provided).  Default: 0.15.
    --max_samples : int, optional
        Subsample this many rows from the training file.
    --num_workers : int
        DataLoader worker processes.  Default: 0.
    """
    parser = argparse.ArgumentParser(
        description="Train the CNN+LSTM fake-news classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data_file", type=str, default="data/fakeddit/all_train.tsv",
        help="Path to training data (CSV / TSV / JSONL).",
    )
    parser.add_argument(
        "--val_file", type=str, default=None,
        help="Optional separate validation file.",
    )
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=300)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.15)

    # Vocab
    parser.add_argument("--max_vocab", type=int, default=50_000)
    parser.add_argument("--min_freq", type=int, default=2)

    # Embeddings
    parser.add_argument(
        "--glove_path", type=str, default=None,
        help="Path to GloVe file (e.g. glove.6B.100d.txt).",
    )
    parser.add_argument("--embedding_dim", type=int, default=100)

    # Model architecture
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="checkpoints/cnn_lstm")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CNN+LSTM Fake-News Classifier — Training")
    logger.info("=" * 60)
    logger.info("Arguments: %s", vars(args))

    # ------------------------------------------------------------------ load
    logger.info("Loading training data from %s …", args.data_file)
    train_df = CNNLSTMDataset._load_file(Path(args.data_file))
    logger.info("Loaded %d rows.", len(train_df))

    if args.max_samples is not None and args.max_samples < len(train_df):
        train_df = train_df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        logger.info("Subsampled training data to %d rows.", len(train_df))

    # ---------------------------------------------------------- val split
    if args.val_file is not None:
        logger.info("Loading validation data from %s …", args.val_file)
        val_df = CNNLSTMDataset._load_file(Path(args.val_file))
        logger.info("Validation rows: %d", len(val_df))
    else:
        logger.info("Splitting %.0f%% for validation …", args.val_split * 100)
        # Need to detect label col first to do stratified split
        cols = [c.lower() for c in train_df.columns]
        train_df.columns = cols
        label_col = CNNLSTMDataset._resolve_col(
            args.label_col, _LABEL_COL_CANDIDATES, cols, "label"
        )
        try:
            train_df, val_df = train_test_split(
                train_df,
                test_size=args.val_split,
                stratify=train_df[label_col],
                random_state=42,
            )
        except ValueError:
            logger.warning("Stratified split failed; falling back to random split.")
            train_df, val_df = train_test_split(
                train_df,
                test_size=args.val_split,
                random_state=42,
            )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        logger.info("Train: %d  Val: %d", len(train_df), len(val_df))

    # ----------------------------------------------------------------- vocab
    logger.info("Building vocabulary (max_vocab=%d, min_freq=%d) …", args.max_vocab, args.min_freq)
    # Detect text column before vocab building
    cols = [c.lower() for c in train_df.columns]
    train_df.columns = cols
    text_col = CNNLSTMDataset._resolve_col(
        args.text_col, _TEXT_COL_CANDIDATES, cols, "text"
    )
    raw_texts = train_df[text_col].fillna("").tolist()
    word2idx, idx2word = build_vocab_from_texts(
        raw_texts, max_vocab=args.max_vocab, min_freq=args.min_freq
    )
    vocab_size = len(word2idx)

    # --------------------------------------------------------------- datasets
    logger.info("Building datasets …")
    train_dataset = CNNLSTMDataset(
        data_source=train_df,
        word2idx=word2idx,
        text_col=args.text_col,
        label_col=args.label_col,
        max_seq_length=args.max_seq_length,
        num_classes=args.num_classes,
    )
    val_dataset = CNNLSTMDataset(
        data_source=val_df,
        word2idx=word2idx,
        text_col=args.text_col,
        label_col=args.label_col,
        max_seq_length=args.max_seq_length,
        num_classes=args.num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(
        "DataLoaders ready. Train batches: %d  Val batches: %d",
        len(train_loader), len(val_loader),
    )

    # --------------------------------------------------------------- train
    best_ckpt = train_cnn_lstm_model(
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        word2idx=word2idx,
        idx2word=idx2word,
        vocab_size=vocab_size,
        epochs=args.epochs,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        filter_sizes=[2, 3, 4, 5],
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        num_classes=args.num_classes,
        max_seq_length=args.max_seq_length,
        glove_path=args.glove_path,
        use_amp=True,
    )

    logger.info("Best checkpoint: %s", best_ckpt)
    logger.info("Training finished successfully.")


if __name__ == "__main__":
    main()
