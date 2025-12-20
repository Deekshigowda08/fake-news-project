import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultimodalFakeNewsDataset, summarize_csv
from model import MultimodalEncoder
from utils import evaluate_predictions, print_metrics, set_seed


def extract_embeddings(dataloader, encoder, device):
    features = []
    labels = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            embeddings = encoder(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["image"].to(device),
            )
            features.append(embeddings.cpu().numpy())
            labels.extend(batch["label"].numpy())

    return np.vstack(features), np.array(labels)


def build_dataloader(csv_path, image_dirs, batch_size, max_len):
    dataset = MultimodalFakeNewsDataset(
        csv_path=csv_path,
        image_dirs=image_dirs,
        max_len=max_len,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_svm(X_train, y_train):
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                ),
            ),
        ]
    )
    classifier.fit(X_train, y_train)
    return classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BERT + ViT feature extractor with a LinearSVC classifier."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing CSVs and image folders.")
    parser.add_argument("--train-csv", default="df_train2.csv", help="Training CSV filename.")
    parser.add_argument("--test-csv", default="df_test2.csv", help="Test CSV filename.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding extraction batch size.")
    parser.add_argument("--max-len", type=int, default=128, help="BERT token sequence length.")
    parser.add_argument("--image-weight", type=float, default=0.5, help="Weight applied to ViT features.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    data_dir = Path(args.data_dir)
    image_dirs = [data_dir / "images_train", data_dir / "images_test"]
    train_csv = data_dir / args.train_csv
    test_csv = data_dir / args.test_csv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Train summary:", summarize_csv(train_csv))
    print("Test summary:", summarize_csv(test_csv))

    train_loader = build_dataloader(train_csv, image_dirs, args.batch_size, args.max_len)
    test_loader = build_dataloader(test_csv, image_dirs, args.batch_size, args.max_len)

    encoder = MultimodalEncoder(image_weight=args.image_weight).to(device)
    X_train, y_train = extract_embeddings(train_loader, encoder, device)
    X_test, y_test = extract_embeddings(test_loader, encoder, device)

    classifier = train_svm(X_train, y_train)
    test_predictions = classifier.predict(X_test)
    print_metrics("Test", evaluate_predictions(y_test, test_predictions))


if __name__ == "__main__":
    main()
