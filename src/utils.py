import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_predictions(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(name, metrics):
    print(f"\n{name} Results")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_fake']:.4f}")
    print(f"Recall   : {metrics['recall_fake']:.4f}")
    print(f"F1-score : {metrics['f1_fake']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
