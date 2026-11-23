import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local imports work when imported as package or executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dataset import MultimodalFakeNewsDataset, summarize_csv, build_vocab
    from model import MultimodalCNNLSTM
    from utils import evaluate_predictions, print_metrics, set_seed
except ImportError:
    from src.dataset import MultimodalFakeNewsDataset, summarize_csv, build_vocab
    from src.model import MultimodalCNNLSTM
    from src.utils import evaluate_predictions, print_metrics, set_seed


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            images = batch["image"].to(device)
            logits = model(input_ids, images)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
    return evaluate_predictions(all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Multimodal Fake News Classifier")
    parser.add_argument("--data-dir", default="data", help="Directory containing CSVs and image folders.")
    parser.add_argument("--train-csv", default="df_train2.csv", help="Training CSV filename.")
    parser.add_argument("--test-csv", default="df_test2.csv", help="Test CSV filename.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints/cnn_lstm_src", help="Checkpoint directory")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    image_dirs = [data_dir / "images_train", data_dir / "images_test"]
    train_csv = data_dir / args.train_csv
    test_csv = data_dir / args.test_csv

    train_ds = MultimodalFakeNewsDataset(train_csv, image_dirs)
    vocab = train_ds.vocab
    test_ds = MultimodalFakeNewsDataset(test_csv, image_dirs, vocab=vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MultimodalCNNLSTM(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} | Test F1: {metrics.get('f1', 0.0):.4f}")

        if metrics.get("f1", 0.0) > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            with open(out_dir / "vocab.json", "w") as f:
                json.dump(vocab, f)
            print(f"--> Saved best model with F1: {best_f1:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
