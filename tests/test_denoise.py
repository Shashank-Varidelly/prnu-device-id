"""
Tests for PRNU denoising module.

Tests the wavelet and Wiener denoising filters against known properties:
- Output shape preservation
- Value range [0, 1]
- Noise reduction (denoised should have lower variance than input)
- Residual is non-trivial (not all zeros)
"""

import numpy as np
import pytest
from src.utils.helpers import wavelet_denoise, wiener_denoise


@pytest.fixture
def synthetic_noisy_image():
    """Create a synthetic noisy 256×256 RGB image."""
    rng = np.random.RandomState(42)
    # Clean gradient pattern
    clean = np.zeros((256, 256, 3), dtype=np.float64)
    for c in range(3):
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        clean[:, :, c] = np.outer(y, x) * (0.5 + 0.2 * c)
    # Add Gaussian noise
    noise = rng.normal(0, 0.05, clean.shape)
    noisy = np.clip(clean + noise, 0, 1)
    return noisy, clean


@pytest.fixture
def synthetic_grayscale():
    """Create a synthetic noisy 128×128 grayscale image."""
    rng = np.random.RandomState(123)
    clean = np.outer(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
    noisy = np.clip(clean + rng.normal(0, 0.08, clean.shape), 0, 1)
    return noisy


class TestWaveletDenoise:
    """Tests for wavelet_denoise()."""

    def test_output_shape_rgb(self, synthetic_noisy_image):
        noisy, _ = synthetic_noisy_image
        denoised = wavelet_denoise(noisy)
        assert denoised.shape == noisy.shape

    def test_output_shape_grayscale(self, synthetic_grayscale):
        denoised = wavelet_denoise(synthetic_grayscale)
        assert denoised.shape == synthetic_grayscale.shape

    def test_value_range(self, synthetic_noisy_image):
        noisy, _ = synthetic_noisy_image
        denoised = wavelet_denoise(noisy)
        assert denoised.min() >= 0.0
        assert denoised.max() <= 1.0

    def test_noise_reduction(self, synthetic_noisy_image):
        """Denoised image should have lower noise (variance) near edges."""
        noisy, clean = synthetic_noisy_image
        denoised = wavelet_denoise(noisy)

        # Residual noise: compare variance
        noisy_residual = noisy - clean
        denoised_residual = denoised - clean

        assert np.var(denoised_residual) < np.var(noisy_residual)

    def test_residual_is_nontrivial(self, synthetic_noisy_image):
        """The denoised image should differ from the input."""
        noisy, _ = synthetic_noisy_image
        denoised = wavelet_denoise(noisy)
        residual = noisy - denoised
        assert np.std(residual) > 1e-6

    def test_uint8_input(self):
        """Should handle uint8 input (0-255 range)."""
        rng = np.random.RandomState(0)
        img = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        denoised = wavelet_denoise(img)
        assert denoised.shape == (64, 64, 3)
        assert denoised.dtype == np.float64
        assert denoised.min() >= 0.0
        assert denoised.max() <= 1.0

    def test_different_wavelets(self, synthetic_grayscale):
        """Should work with different wavelet families."""
        for wavelet in ["db2", "db4", "db8", "haar"]:
            denoised = wavelet_denoise(synthetic_grayscale, wavelet=wavelet)
            assert denoised.shape == synthetic_grayscale.shape

    def test_different_levels(self, synthetic_grayscale):
        """Different decomposition levels should all work."""
        for levels in [1, 2, 3, 4]:
            denoised = wavelet_denoise(synthetic_grayscale, levels=levels)
            assert denoised.shape == synthetic_grayscale.shape

    def test_small_image(self):
        """Should handle images smaller than typical DWT requirements."""
        rng = np.random.RandomState(0)
        small = rng.rand(16, 16)
        denoised = wavelet_denoise(small, levels=4)
        assert denoised.shape == small.shape


class TestWienerDenoise:
    """Tests for wiener_denoise()."""

    def test_output_shape(self, synthetic_noisy_image):
        noisy, _ = synthetic_noisy_image
        denoised = wiener_denoise(noisy)
        assert denoised.shape == noisy.shape

    def test_value_range(self, synthetic_noisy_image):
        noisy, _ = synthetic_noisy_image
        denoised = wiener_denoise(noisy)
        assert denoised.min() >= 0.0
        assert denoised.max() <= 1.0

    def test_noise_reduction(self, synthetic_noisy_image):
        noisy, clean = synthetic_noisy_image
        denoised = wiener_denoise(noisy)
        assert np.var(denoised - clean) < np.var(noisy - clean)

    def test_grayscale(self, synthetic_grayscale):
        denoised = wiener_denoise(synthetic_grayscale)
        assert denoised.shape == synthetic_grayscale.shape

    def test_custom_window(self, synthetic_grayscale):
        for ws in [3, 5, 7, 11]:
            denoised = wiener_denoise(synthetic_grayscale, window_size=ws)
            assert denoised.shape == synthetic_grayscale.shape
