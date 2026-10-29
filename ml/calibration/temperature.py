"""Post-hoc probability calibration using Temperature Scaling.

Learns a single scalar parameter T > 0 on validation set logits such that:
P_calibrated = Softmax(logits / T)
"""

from typing import Union
import numpy as np
from scipy.optimize import minimize


class TemperatureScaler:
    """Finds optimal temperature T using negative log-likelihood on validation data."""

    def __init__(self, initial_temp: float = 1.0):
        self.temperature = float(initial_temp)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Optimizes temperature on validation logits and binary labels (0 or 1)."""
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if logits.ndim == 1:
            # Binary logit (log-odds)
            logits_2d = np.stack([-logits / 2.0, logits / 2.0], axis=1)
        elif logits.shape[1] == 1:
            logits_2d = np.concatenate([-logits / 2.0, logits / 2.0], axis=1)
        else:
            logits_2d = logits

        def nll_loss(t_array: np.ndarray) -> float:
            t = t_array[0]
            scaled = logits_2d / t
            # Stable log-softmax
            max_s = np.max(scaled, axis=1, keepdims=True)
            log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
            log_probs = scaled - log_sum_exp
            # Negative log likelihood of true class
            chosen_log_probs = log_probs[np.arange(len(labels)), labels]
            return float(-np.mean(chosen_log_probs))

        res = minimize(nll_loss, [self.temperature], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        if res.success:
            self.temperature = float(res.x[0])

        return self

    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        """Applies temperature to raw logits."""
        return logits / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Computes calibrated probabilities for [class_0, class_1]."""
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits_2d = np.stack([-logits / 2.0, logits / 2.0], axis=1)
        elif logits.shape[1] == 1:
            logits_2d = np.concatenate([-logits / 2.0, logits / 2.0], axis=1)
        else:
            logits_2d = logits

        scaled = logits_2d / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        return probs
