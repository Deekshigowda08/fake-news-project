"""Research evaluation metrics including calibration (ECE, Brier Score) and ranking metrics."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Computes Expected Calibration Error (ECE) and bin reliability statistics.
    
    Args:
        y_true: 1D binary array of ground truth labels (0 or 1).
        probabilities: 1D array of predicted probabilities for positive class (1).
        n_bins: Number of equal-width probability bins.
    
    Returns:
        Dictionary with ECE score, maximum calibration error (MCE), and per-bin stats.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.maximum(probabilities, 1.0 - probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    mce = 0.0
    bin_stats = []

    n_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += bin_error * prop_in_bin
            mce = max(mce, bin_error)
            bin_stats.append({
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": int(np.sum(in_bin)),
                "accuracy": round(float(accuracy_in_bin), 4),
                "confidence": round(float(avg_confidence_in_bin), 4),
                "error": round(float(bin_error), 4),
            })
        else:
            bin_stats.append({
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "error": 0.0,
            })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "n_bins": n_bins,
        "bins": bin_stats,
    }


def compute_comprehensive_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[float], np.ndarray]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calculates full suite of classification and calibration metrics.
    
    Returns:
        accuracy, precision, recall, f1, confusion_matrix,
        roc_auc (if prob provided), pr_auc (if prob provided),
        brier_score (if prob provided), ece (if prob provided).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "total_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # Brier Score
        brier = float(brier_score_loss(y_true, y_prob))
        metrics["brier_score"] = round(brier, 4)

        # ROC-AUC & PR-AUC
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            metrics["roc_auc"] = round(roc_auc, 4)
        except Exception:
            metrics["roc_auc"] = None

        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
            metrics["pr_auc"] = round(pr_auc, 4)
        except Exception:
            metrics["pr_auc"] = None

        # Expected Calibration Error
        calib_stats = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        metrics["ece"] = calib_stats["ece"]
        metrics["mce"] = calib_stats["mce"]
        metrics["calibration_bins"] = calib_stats["bins"]

    return metrics
