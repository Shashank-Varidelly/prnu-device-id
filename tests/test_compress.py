"""
Tests for JPEG / resize compression utilities.
"""

import numpy as np
import pytest
from src.utils.helpers import (
    jpeg_compress,
    resize_image,
    jpeg_quality_sweep,
    resize_sweep,
    JPEGAugmentation,
    compute_jpeg_file_size,
)


@pytest.fixture
def test_image_rgb():
    """128×128 RGB uint8 test image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def test_image_gray():
    """128×128 grayscale uint8 test image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (128, 128), dtype=np.uint8)


class TestJPEGCompress:

    def test_output_shape(self, test_image_rgb):
        compressed = jpeg_compress(test_image_rgb, quality=75)
        assert compressed.shape == test_image_rgb.shape

    def test_output_dtype(self, test_image_rgb):
        compressed = jpeg_compress(test_image_rgb, quality=75)
        assert compressed.dtype == np.uint8

    def test_quality_affects_output(self, test_image_rgb):
        """Higher quality should be closer to original."""
        high_q = jpeg_compress(test_image_rgb, quality=95)
        low_q = jpeg_compress(test_image_rgb, quality=20)

        mse_high = np.mean((test_image_rgb.astype(float) - high_q.astype(float)) ** 2)
        mse_low = np.mean((test_image_rgb.astype(float) - low_q.astype(float)) ** 2)
        assert mse_high < mse_low

    def test_returns_bytes(self, test_image_rgb):
        result = jpeg_compress(test_image_rgb, quality=75, return_array=False)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_grayscale(self, test_image_gray):
        compressed = jpeg_compress(test_image_gray, quality=75)
        assert compressed.ndim == 2

    def test_float_input(self):
        img = np.random.rand(64, 64, 3)
        compressed = jpeg_compress(img, quality=75)
        assert compressed.dtype == np.uint8


class TestResizeImage:

    def test_scale_factor(self, test_image_rgb):
        resized = resize_image(test_image_rgb, scale_factor=0.5)
        assert resized.shape == (64, 64, 3)

    def test_target_size(self, test_image_rgb):
        resized = resize_image(test_image_rgb, target_size=(64, 32))
        assert resized.shape == (32, 64, 3)  # OpenCV: target_size is (w, h)

    def test_both_params_raises(self, test_image_rgb):
        with pytest.raises(ValueError, match="not both"):
            resize_image(test_image_rgb, scale_factor=0.5, target_size=(64, 64))

    def test_neither_params_raises(self, test_image_rgb):
        with pytest.raises(ValueError, match="Must specify"):
            resize_image(test_image_rgb)


class TestJPEGQualitySweep:

    def test_default_qualities(self, test_image_rgb):
        results = jpeg_quality_sweep(test_image_rgb)
        expected_qs = [95, 90, 85, 80, 70, 60, 50, 40, 30]
        assert set(results.keys()) == set(expected_qs)

    def test_custom_qualities(self, test_image_rgb):
        results = jpeg_quality_sweep(test_image_rgb, quality_range=[50, 75])
        assert set(results.keys()) == {50, 75}

    def test_all_outputs_valid(self, test_image_rgb):
        results = jpeg_quality_sweep(test_image_rgb)
        for q, img in results.items():
            assert img.shape == test_image_rgb.shape
            assert img.dtype == np.uint8


class TestResizeSweep:

    def test_default_scales(self, test_image_rgb):
        results = resize_sweep(test_image_rgb)
        expected = [1.0, 0.75, 0.5, 0.375, 0.25]
        assert set(results.keys()) == set(expected)

    def test_identity_scale(self, test_image_rgb):
        results = resize_sweep(test_image_rgb)
        np.testing.assert_array_equal(results[1.0], test_image_rgb)


class TestJPEGAugmentation:

    def test_augmentation_returns_valid(self, test_image_rgb):
        aug = JPEGAugmentation(quality_range=(30, 95), probability=1.0)
        result = aug(test_image_rgb)
        assert result.shape == test_image_rgb.shape

    def test_no_augmentation(self, test_image_rgb):
        """With probability=0, output should match input."""
        aug = JPEGAugmentation(probability=0.0)
        result = aug(test_image_rgb)
        np.testing.assert_array_equal(result, test_image_rgb)

    def test_repr(self):
        aug = JPEGAugmentation(quality_range=(30, 90), probability=0.7)
        r = repr(aug)
        assert "30" in r
        assert "90" in r


class TestFileSize:

    def test_lower_quality_smaller_file(self, test_image_rgb):
        size_high = compute_jpeg_file_size(test_image_rgb, quality=95)
        size_low = compute_jpeg_file_size(test_image_rgb, quality=30)
        assert size_low < size_high
