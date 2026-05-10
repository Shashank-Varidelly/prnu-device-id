"""
Helper Functions
=================

Denoising filters, compression utilities, and plotting tools used
across the PRNU pipeline.
"""

from __future__ import annotations

import numpy as np
import pywt
import cv2
from scipy.ndimage import uniform_filter
from typing import Optional, List, Tuple


# ===========================================================================
# Wavelet denoising (Lukas, Fridrich & Goljan, 2006, Appendix A)
# ===========================================================================

def _estimate_noise_sigma(detail_coeffs):
    """Robust noise variance estimator: sigma = median(|d|) / 0.6745"""
    return float(np.median(np.abs(detail_coeffs)) / 0.6745)


def _bayes_shrink_threshold(detail_coeffs, sigma_n):
    """BayesShrink adaptive threshold (Chang et al., 2000)."""
    sigma_d_sq = max(np.var(detail_coeffs) - sigma_n ** 2, 0.0)
    if sigma_d_sq == 0.0:
        return float(np.max(np.abs(detail_coeffs)))
    return sigma_n ** 2 / np.sqrt(sigma_d_sq)


def _soft_threshold(coeffs, threshold):
    """Soft thresholding: sign(x) * max(|x| - T, 0)."""
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def _wavelet_denoise_channel(channel, wavelet, levels):
    """Denoise a single grayscale channel via DWT + BayesShrink."""
    max_level = pywt.dwt_max_level(min(channel.shape), pywt.Wavelet(wavelet).dec_len)
    levels = min(levels, max_level)

    coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=levels)
    sigma_n = _estimate_noise_sigma(coeffs[-1][2])

    denoised_coeffs = [coeffs[0]]
    for detail_tuple in coeffs[1:]:
        denoised_detail = []
        for subband in detail_tuple:
            threshold = _bayes_shrink_threshold(subband, sigma_n)
            denoised_detail.append(_soft_threshold(subband, threshold))
        denoised_coeffs.append(tuple(denoised_detail))

    reconstructed = pywt.waverec2(denoised_coeffs, wavelet=wavelet)
    return np.clip(reconstructed[:channel.shape[0], :channel.shape[1]], 0.0, 1.0)


def wavelet_denoise(image, wavelet="db4", levels=4, per_channel=True):
    """Denoise an image using multi-level DWT + BayesShrink.

    This is the F(I) filter from Lukas et al. (2006), Appendix A.
    The noise residual is then W = I - F(I).

    Parameters
    ----------
    image : np.ndarray
        Input image, (H, W) or (H, W, C). uint8 or float.
    wavelet : str
        Mother wavelet (default 'db4').
    levels : int
        Decomposition levels (default 4).
    per_channel : bool
        Denoise each colour channel independently.

    Returns
    -------
    np.ndarray
        Denoised image, float64 in [0, 1].
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    if img.ndim == 2:
        return _wavelet_denoise_channel(img, wavelet, levels)

    if not per_channel:
        gray = np.mean(img, axis=2)
        return np.stack([_wavelet_denoise_channel(gray, wavelet, levels)] * img.shape[2], axis=2)

    channels = [_wavelet_denoise_channel(img[:, :, c], wavelet, levels)
                for c in range(img.shape[2])]
    return np.stack(channels, axis=2)


def _wiener_channel(channel, window_size, noise_variance):
    """Wiener filter on a single channel."""
    local_mean = uniform_filter(channel, size=window_size, mode="reflect")
    local_sq_mean = uniform_filter(channel ** 2, size=window_size, mode="reflect")
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    if noise_variance is None:
        noise_variance = float(np.mean(local_var))

    ratio = np.where(local_var > noise_variance,
                     (local_var - noise_variance) / local_var, 0.0)
    return np.clip(local_mean + ratio * (channel - local_mean), 0.0, 1.0)


def wiener_denoise(image, window_size=5, noise_variance=None, per_channel=True):
    """Local Wiener filter denoising (ablation baseline, Experiment B2).

    Parameters
    ----------
    image : np.ndarray
        Input image, (H, W) or (H, W, C).
    window_size : int
        Local window size for variance estimation.
    noise_variance : float or None
        If None, estimate globally.
    per_channel : bool
        Filter each channel independently.

    Returns
    -------
    np.ndarray
        Denoised image, float64 in [0, 1].
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    if img.ndim == 2:
        return _wiener_channel(img, window_size, noise_variance)

    if not per_channel:
        gray = np.mean(img, axis=2)
        out = _wiener_channel(gray, window_size, noise_variance)
        return np.stack([out] * img.shape[2], axis=2)

    channels = [_wiener_channel(img[:, :, c], window_size, noise_variance)
                for c in range(img.shape[2])]
    return np.stack(channels, axis=2)


# ===========================================================================
# JPEG / Resize compression utilities
# ===========================================================================

def _ensure_uint8(image):
    """Convert image to uint8 if needed."""
    if image.dtype == np.uint8:
        return image
    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def jpeg_compress(image, quality=75, return_array=True):
    """Apply JPEG compression at a given quality factor.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8 or float).
    quality : int
        JPEG quality 1-100.
    return_array : bool
        If True, decode back to ndarray. If False, return bytes.

    Returns
    -------
    np.ndarray or bytes
    """
    img = _ensure_uint8(image)
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("JPEG encoding failed.")
    if not return_array:
        return buf.tobytes()
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def resize_image(image, scale_factor=None, target_size=None,
                 interpolation=cv2.INTER_AREA):
    """Resize image by scale factor or to target size."""
    if scale_factor is not None and target_size is not None:
        raise ValueError("Specify scale_factor OR target_size, not both.")
    if scale_factor is None and target_size is None:
        raise ValueError("Must specify scale_factor or target_size.")

    img = _ensure_uint8(image)
    if scale_factor is not None:
        return cv2.resize(img, None, fx=scale_factor, fy=scale_factor,
                          interpolation=interpolation)
    return cv2.resize(img, target_size, interpolation=interpolation)


def jpeg_quality_sweep(image, quality_range=None):
    """Apply JPEG at multiple quality levels (Experiment C1)."""
    if quality_range is None:
        quality_range = [95, 90, 85, 80, 70, 60, 50, 40, 30]
    return {q: jpeg_compress(image, quality=q) for q in quality_range}


def resize_sweep(image, scale_factors=None):
    """Resize at multiple scale factors (Experiment C2)."""
    if scale_factors is None:
        scale_factors = [1.0, 0.75, 0.5, 0.375, 0.25]
    results = {}
    for sf in scale_factors:
        results[sf] = image.copy() if sf == 1.0 else resize_image(image, scale_factor=sf)
    return results


class JPEGAugmentation:
    """On-the-fly JPEG compression augmentation for training (Experiment A3)."""

    def __init__(self, quality_range=(30, 95), probability=0.5):
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, image):
        if np.random.rand() > self.probability:
            return image
        quality = np.random.randint(self.quality_range[0], self.quality_range[1] + 1)
        return jpeg_compress(image, quality=quality)

    def __repr__(self):
        return (f"JPEGAugmentation(quality_range={self.quality_range}, "
                f"probability={self.probability})")


def compute_jpeg_file_size(image, quality):
    """Return JPEG-encoded file size in bytes."""
    return len(jpeg_compress(image, quality=quality, return_array=False))
