"""Consistent validation-threshold selection and binary classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, brier_score_loss, f1_score, precision_recall_curve


def select_macro_f1_threshold(y_validation, validation_probabilities) -> float:
    """Choose the decision threshold on validation labels only."""

    labels = np.asarray(y_validation, dtype=int)
    probabilities = np.asarray(validation_probabilities, dtype=float).reshape(-1)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = np.array(
        [
            f1_score(labels, probabilities >= threshold, average="macro")
            for threshold in candidates
        ]
    )
    best_score = float(scores.max())
    best_candidates = candidates[np.isclose(scores, best_score)]
    return float(best_candidates[np.argmin(np.abs(best_candidates - 0.5))])


def binary_metrics(y_true, probabilities, threshold: float) -> dict[str, float]:
    """Return explicitly named, common metrics for every model variant."""

    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {
        "PR-AUC-Trapezoidal": float(auc(recall, precision)),
        "F1-Macro": float(f1_score(labels, predictions, average="macro")),
        "F1-Quark": float(f1_score(labels, predictions, pos_label=1)),
        "Brier-Score": float(brier_score_loss(labels, probabilities)),
        "Decision-Threshold": float(threshold),
    }
