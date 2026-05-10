"""
Evaluation Metrics
==================

All metrics for the 12 PRNU experiments: accuracy, FAR/FRR, ECE,
Brier Score, selective risk, confusion matrix, error taxonomy.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from typing import Optional, List, Tuple, Dict


# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------

def top_k_accuracy(y_true, y_probs, k=5):
    """Compute Top-K accuracy."""
    y_true = np.asarray(y_true)
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return float(correct.mean())


def classification_report_dict(y_true, y_pred, label_names=None):
    """Wrapper around sklearn classification_report returning a dict."""
    return classification_report(y_true, y_pred, target_names=label_names,
                                 output_dict=True, zero_division=0)


# ---------------------------------------------------------------------------
# FAR / FRR
# ---------------------------------------------------------------------------

def compute_far_frr(genuine_scores, impostor_scores, threshold):
    """Compute False Acceptance Rate and False Rejection Rate."""
    genuine_scores = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float64)
    far = float(np.mean(impostor_scores >= threshold))
    frr = float(np.mean(genuine_scores < threshold))
    return far, frr


def compute_far_frr_curve(genuine_scores, impostor_scores, num_thresholds=1000):
    """Compute FAR/FRR over a range of thresholds."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
    fars = np.array([np.mean(impostor_scores >= t) for t in thresholds])
    frrs = np.array([np.mean(genuine_scores < t) for t in thresholds])
    return thresholds, fars, frrs


def compute_eer(genuine_scores, impostor_scores):
    """Compute Equal Error Rate (where FAR ~ FRR)."""
    thresholds, fars, frrs = compute_far_frr_curve(genuine_scores, impostor_scores)
    diffs = np.abs(fars - frrs)
    idx = np.argmin(diffs)
    eer = (fars[idx] + frrs[idx]) / 2.0
    return float(eer), float(thresholds[idx])


# ---------------------------------------------------------------------------
# Calibration metrics (Experiment C3)
# ---------------------------------------------------------------------------

def compute_ece(y_true, y_probs, n_bins=15):
    """Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true)
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * np.abs(bin_acc - bin_conf)

    return float(ece / len(y_true))


def compute_brier_score(y_true, y_probs):
    """Brier Score (multi-class). Lower is better."""
    y_true = np.asarray(y_true)
    one_hot = np.zeros_like(y_probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))


def selective_risk(y_true, y_probs, coverage=0.9):
    """Selective prediction risk at a given coverage level."""
    y_true = np.asarray(y_true)
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)

    sorted_indices = np.argsort(-confidences)
    n_keep = int(np.ceil(coverage * len(y_true)))
    kept = sorted_indices[:n_keep]

    accuracy = float(np.mean(predictions[kept] == y_true[kept]))
    threshold = float(confidences[kept[-1]]) if n_keep > 0 else 0.0
    return accuracy, threshold


# ---------------------------------------------------------------------------
# Confusion matrix & error taxonomy
# ---------------------------------------------------------------------------

def compute_confusion_matrix(y_true, y_pred, normalize=None):
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def error_taxonomy(y_true, y_pred, device_to_model):
    """Categorise errors into same-model vs cross-model confusion."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_true != y_pred

    same_model = cross_model = 0
    for i in np.where(errors)[0]:
        true_model = device_to_model.get(int(y_true[i]), "unknown")
        pred_model = device_to_model.get(int(y_pred[i]), "unknown")
        if true_model == pred_model:
            same_model += 1
        else:
            cross_model += 1

    total = same_model + cross_model
    return {"total_errors": total, "same_model_errors": same_model,
            "cross_model_errors": cross_model,
            "same_model_rate": same_model / max(total, 1)}


def per_device_frr(y_true, y_pred):
    """Compute per-device False Rejection Rate (Experiment D1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results = {}
    for device in np.unique(y_true):
        mask = y_true == device
        correct = np.sum(y_pred[mask] == device)
        results[int(device)] = 1.0 - correct / max(np.sum(mask), 1)
    return results
