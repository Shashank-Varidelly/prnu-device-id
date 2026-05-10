"""
Tests for patch extraction module.
"""

import numpy as np
import pytest
from src.utils.data_loader import extract_patches, extract_center_patch


@pytest.fixture
def residual_256():
    """256×256 grayscale residual."""
    rng = np.random.RandomState(42)
    return rng.randn(256, 256)


@pytest.fixture
def residual_rgb():
    """256×256×3 colour residual."""
    rng = np.random.RandomState(42)
    return rng.randn(256, 256, 3)


class TestExtractPatches:

    def test_non_overlapping_count(self, residual_256):
        patches = extract_patches(residual_256, patch_size=64)
        # 256 / 64 = 4 per dimension → 16 patches
        assert patches.shape == (16, 64, 64)

    def test_non_overlapping_128(self, residual_256):
        patches = extract_patches(residual_256, patch_size=128)
        # 256 / 128 = 2 per dim → 4 patches
        assert patches.shape == (4, 128, 128)

    def test_single_patch_256(self, residual_256):
        patches = extract_patches(residual_256, patch_size=256)
        assert patches.shape == (1, 256, 256)

    def test_rgb_patches(self, residual_rgb):
        patches = extract_patches(residual_rgb, patch_size=64)
        assert patches.shape == (16, 64, 64, 3)

    def test_stride_overlap(self, residual_256):
        patches = extract_patches(residual_256, patch_size=64, stride=32)
        # (256 - 64) / 32 + 1 = 7 per dim → 49 patches
        assert patches.shape == (49, 64, 64)

    def test_max_patches(self, residual_256):
        patches = extract_patches(residual_256, patch_size=64, max_patches=5,
                                  random_state=42)
        assert patches.shape == (5, 64, 64)

    def test_max_patches_reproducible(self, residual_256):
        p1 = extract_patches(residual_256, patch_size=64, max_patches=3,
                             random_state=42)
        p2 = extract_patches(residual_256, patch_size=64, max_patches=3,
                             random_state=42)
        np.testing.assert_array_equal(p1, p2)

    def test_too_small_image_raises(self):
        small = np.random.randn(16, 16)
        with pytest.raises(ValueError, match="too small"):
            extract_patches(small, patch_size=64)


class TestExtractCenterPatch:

    def test_center_patch_shape(self, residual_256):
        patch = extract_center_patch(residual_256, patch_size=128)
        assert patch.shape == (128, 128)

    def test_center_patch_rgb(self, residual_rgb):
        patch = extract_center_patch(residual_rgb, patch_size=128)
        assert patch.shape == (128, 128, 3)

    def test_too_large_raises(self):
        small = np.random.randn(32, 32)
        with pytest.raises(ValueError, match="smaller than"):
            extract_center_patch(small, patch_size=64)

    def test_center_location(self, residual_256):
        """Center patch values should match the actual centre of the image."""
        patch = extract_center_patch(residual_256, patch_size=64)
        expected = residual_256[96:160, 96:160]
        np.testing.assert_array_equal(patch, expected)
