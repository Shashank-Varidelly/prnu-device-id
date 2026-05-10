"""
NCC Baseline — Normalized Cross-Correlation Matching
=====================================================

Classical PRNU-based device attribution using normalized cross-correlation
between query noise residuals and stored device fingerprints.

References
----------
Lukas, J., Fridrich, J., & Goljan, M. (2006). Digital camera
identification from sensor pattern noise. IEEE TIFS, 1(2), 205-214.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def ncc_score(
    residual: np.ndarray,
    fingerprint: np.ndarray,
    image: Optional[np.ndarray] = None,
) -> float:
    """Normalised Cross-Correlation between a query residual and a fingerprint.

    If ``image`` is provided, the correlation is computed against
    ``I * K_d`` (the signal-dependent fingerprint), following Eq.6
    of Lukas et al.  Otherwise, it correlates directly against K_d.

    Parameters
    ----------
    residual : np.ndarray
        Query noise residual W_q.
    fingerprint : np.ndarray
        Device fingerprint K_d.
    image : np.ndarray or None
        Query image I_q (optional, for signal-dependent matching).

    Returns
    -------
    float
        NCC score in [-1, 1].
    """
    if image is not None:
        img_f = image.astype(np.float64)
        if img_f.max() > 1.0:
            img_f /= 255.0
        reference = img_f * fingerprint
    else:
        reference = fingerprint

    # Flatten for correlation
    w = residual.ravel().astype(np.float64)
    r = reference.ravel().astype(np.float64)

    # Zero-mean
    w = w - w.mean()
    r = r - r.mean()

    # NCC
    denom = np.sqrt(np.sum(w ** 2) * np.sum(r ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(w * r) / denom)


def ncc_score_batch(
    residual: np.ndarray,
    fingerprints: dict[str, np.ndarray],
    image: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Score a query residual against multiple device fingerprints.

    Parameters
    ----------
    residual : np.ndarray
        Query noise residual.
    fingerprints : dict[str, np.ndarray]
        Mapping device_id -> fingerprint K_d.
    image : np.ndarray or None
        Query image (optional).

    Returns
    -------
    dict[str, float]
        Mapping device_id -> NCC score, sorted descending.
    """
    scores = {}
    for device_id, fp in fingerprints.items():
        scores[device_id] = ncc_score(residual, fp, image)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def compute_threshold(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: float = 0.01,
) -> float:
    """Compute NCC decision threshold via Neyman-Pearson criterion.

    Selects the threshold t such that FAR <= target_far.

    Parameters
    ----------
    genuine_scores : np.ndarray
        NCC scores for genuine (same-device) pairs.
    impostor_scores : np.ndarray
        NCC scores for impostor (different-device) pairs.
    target_far : float
        Target false-acceptance rate. Default 0.01 (1%).

    Returns
    -------
    float
        Decision threshold t.
    """
    impostor_sorted = np.sort(impostor_scores)[::-1]
    idx = max(int(np.ceil(target_far * len(impostor_sorted))) - 1, 0)
    return float(impostor_sorted[idx])
