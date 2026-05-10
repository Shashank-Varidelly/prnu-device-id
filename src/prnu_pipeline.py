"""
PRNU Pipeline — Main Orchestrator
===================================

End-to-end PRNU device identification pipeline:
1. Noise residual extraction: W = I - F(I)
2. Fingerprint estimation: K_d = MLE from M training images
3. Attribution: NCC / CNN / Siamese matching

Usage
-----
    from src.prnu_pipeline import PRNUPipeline

    pipeline = PRNUPipeline(denoiser="wavelet")
    fingerprint = pipeline.estimate_fingerprint(training_images)
    scores = pipeline.identify(query_image, fingerprint_gallery)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

from src.utils.helpers import wavelet_denoise, wiener_denoise
from src.algorithms.ncc_baseline import ncc_score, ncc_score_batch, compute_threshold


class PRNUPipeline:
    """End-to-end PRNU camera identification pipeline.

    Parameters
    ----------
    denoiser : str
        'wavelet' (default, Lukas et al.) or 'wiener' (ablation).
    **denoise_kwargs
        Forwarded to the denoiser function.
    """

    def __init__(self, denoiser: str = "wavelet", **denoise_kwargs):
        self.denoiser = denoiser
        self.denoise_kwargs = denoise_kwargs

    def extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """Extract PRNU noise residual W = I - F(I).

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W) or (H, W, C), uint8 or float.

        Returns
        -------
        np.ndarray
            Noise residual, float64, same shape as input.
        """
        img = image.astype(np.float64)
        if img.max() > 1.0:
            img /= 255.0

        if self.denoiser == "wavelet":
            denoised = wavelet_denoise(img, **self.denoise_kwargs)
        elif self.denoiser == "wiener":
            denoised = wiener_denoise(img, **self.denoise_kwargs)
        else:
            raise ValueError(f"Unknown denoiser: {self.denoiser!r}")

        return img - denoised

    def estimate_fingerprint(
        self,
        images: List[np.ndarray],
        return_individual: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray | Tuple[np.ndarray, List[np.ndarray]]:
        """Estimate camera fingerprint K_d from M training images.

        Uses the MLE (Eq. 4 in Lukas et al.):
            K_d = sum(W_i * I_i) / sum(I_i^2)

        Parameters
        ----------
        images : list of np.ndarray
            Training images from a single device.
        return_individual : bool
            Also return individual noise residuals.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        fingerprint : np.ndarray
            Estimated fingerprint K_d.
        residuals : list  (only if return_individual=True)
        """
        if not images:
            raise ValueError("Need at least one image.")

        iterator = tqdm(images, desc="Estimating fingerprint",
                        disable=not show_progress)

        numerator = None
        denominator = None
        residuals = []

        for img in iterator:
            img_f = img.astype(np.float64)
            if img_f.max() > 1.0:
                img_f /= 255.0

            w_i = self.extract_noise_residual(img_f)
            residuals.append(w_i)

            prod = w_i * img_f
            sq = img_f ** 2

            if numerator is None:
                numerator = prod
                denominator = sq
            else:
                numerator += prod
                denominator += sq

        denominator = np.where(denominator == 0, 1.0, denominator)
        fingerprint = numerator / denominator

        if return_individual:
            return fingerprint, residuals
        return fingerprint

    def identify(
        self,
        query_image: np.ndarray,
        fingerprint_gallery: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Identify the source device of a query image.

        Parameters
        ----------
        query_image : np.ndarray
            Query image.
        fingerprint_gallery : dict
            Mapping device_id -> fingerprint K_d.

        Returns
        -------
        dict[str, float]
            Device scores sorted descending.
        """
        residual = self.extract_noise_residual(query_image)
        return ncc_score_batch(residual, fingerprint_gallery, image=query_image)

    @staticmethod
    def compute_threshold(genuine_scores, impostor_scores, target_far=0.01):
        """Compute decision threshold via Neyman-Pearson criterion."""
        return compute_threshold(genuine_scores, impostor_scores, target_far)
