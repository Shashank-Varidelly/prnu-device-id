"""
Data Loader — Dataset loading, splits, and patch extraction
=============================================================

Handles data pipeline: loading images, device-stratified splitting,
patch extraction, and PyTorch Dataset classes.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Patch extraction (pure numpy)
# ---------------------------------------------------------------------------

def extract_patches(
    residual: np.ndarray,
    patch_size: int = 128,
    stride: Optional[int] = None,
    max_patches: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Extract non-overlapping or strided patches from a noise residual.

    Parameters
    ----------
    residual : np.ndarray
        Noise residual image, shape (H, W) or (H, W, C).
    patch_size : int
        Side length of square patches. One of {64, 128, 256}.
    stride : int or None
        Stride between patches. Defaults to patch_size (non-overlapping).
    max_patches : int or None
        Maximum patches to return (random subset if more available).
    random_state : int or None
        RNG seed for reproducible patch selection.

    Returns
    -------
    np.ndarray
        Array of patches, shape (N, patch_size, patch_size[, C]).
    """
    if stride is None:
        stride = patch_size

    h, w = residual.shape[:2]

    row_positions = list(range(0, h - patch_size + 1, stride))
    col_positions = list(range(0, w - patch_size + 1, stride))

    patches = []
    for r in row_positions:
        for c in col_positions:
            if residual.ndim == 2:
                patches.append(residual[r:r + patch_size, c:c + patch_size])
            else:
                patches.append(residual[r:r + patch_size, c:c + patch_size, :])

    if not patches:
        raise ValueError(
            f"Image too small ({h}x{w}) for patch_size={patch_size}."
        )

    patches = np.array(patches)

    if max_patches is not None and len(patches) > max_patches:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(patches), size=max_patches, replace=False)
        patches = patches[indices]

    return patches


def extract_center_patch(
    residual: np.ndarray,
    patch_size: int = 256,
) -> np.ndarray:
    """Extract the single centre patch from a residual image.

    Parameters
    ----------
    residual : np.ndarray
        Noise residual, (H, W) or (H, W, C).
    patch_size : int
        Side length of the square patch.

    Returns
    -------
    np.ndarray
        Centre patch, shape (patch_size, patch_size[, C]).
    """
    h, w = residual.shape[:2]
    r = (h - patch_size) // 2
    c = (w - patch_size) // 2

    if r < 0 or c < 0:
        raise ValueError(
            f"Image ({h}x{w}) smaller than patch_size={patch_size}."
        )

    if residual.ndim == 2:
        return residual[r:r + patch_size, c:c + patch_size]
    return residual[r:r + patch_size, c:c + patch_size, :]


# ---------------------------------------------------------------------------
# Splits management
# ---------------------------------------------------------------------------

def load_splits(splits_path: str | Path) -> dict:
    """Load device-stratified splits from JSON file.

    Returns
    -------
    dict
        Split configuration with train/val/test device assignments.
    """
    with open(splits_path) as f:
        return json.load(f)


def build_device_stratified_split(
    image_index: dict,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Build train/val/test splits stratified by device identity.

    CRITICAL: Images from the same device must all go to the same split.
    This prevents data leakage from shared PRNU patterns.

    Parameters
    ----------
    image_index : dict
        Mapping device_id -> list of image paths.
    train_ratio, val_ratio, test_ratio : float
        Split ratios (must sum to 1.0).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        {'train': {device: [paths]}, 'val': {device: [paths]},
         'test': {device: [paths]}, 'metadata': {...}}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    devices = sorted(image_index.keys())
    rng.shuffle(devices)

    n = len(devices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_devices = devices[:n_train]
    val_devices = devices[n_train:n_train + n_val]
    test_devices = devices[n_train + n_val:]

    splits = {
        "train": {d: image_index[d] for d in train_devices},
        "val": {d: image_index[d] for d in val_devices},
        "test": {d: image_index[d] for d in test_devices},
        "metadata": {
            "seed": seed,
            "num_devices": n,
            "num_train_devices": len(train_devices),
            "num_val_devices": len(val_devices),
            "num_test_devices": len(test_devices),
        },
    }

    return splits


# ---------------------------------------------------------------------------
# PyTorch Datasets (lazy torch import)
# ---------------------------------------------------------------------------

try:
    from torch.utils.data import Dataset as _DatasetBase
except (ImportError, OSError):
    _DatasetBase = object


class PatchDataset(_DatasetBase):
    """PyTorch Dataset for PRNU residual patches.

    Loads pre-extracted residuals from .npy files, extracts patches
    on-the-fly, and pairs them with device labels.
    """

    def __init__(self, residual_paths, labels, patch_size=128,
                 patches_per_image=4, transform=None, random_seed=42):
        assert len(residual_paths) == len(labels)
        self.residual_paths = [Path(p) for p in residual_paths]
        self.labels = labels
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform
        self.rng = np.random.RandomState(random_seed)

        self._index = []
        for img_idx in range(len(self.residual_paths)):
            for p in range(patches_per_image):
                self._index.append((img_idx, p))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        import torch
        img_idx, patch_num = self._index[idx]
        residual = np.load(self.residual_paths[img_idx])
        seed = self.rng.randint(0, 2**31) + patch_num
        patches = extract_patches(residual, patch_size=self.patch_size,
                                  max_patches=1, random_state=seed)
        patch = patches[0]
        if patch.ndim == 2:
            tensor = torch.from_numpy(patch).float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(patch).float().permute(2, 0, 1)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, self.labels[img_idx]


class SiamesePatchDataset(_DatasetBase):
    """Dataset yielding (anchor, other, label) pairs for contrastive training."""

    def __init__(self, residual_paths, labels, patch_size=128,
                 pairs_per_epoch=10_000, transform=None, random_seed=42):
        self.residual_paths = [Path(p) for p in residual_paths]
        self.labels = np.array(labels)
        self.patch_size = patch_size
        self.pairs_per_epoch = pairs_per_epoch
        self.transform = transform
        self.rng = np.random.RandomState(random_seed)

        self.label_to_indices = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            self.label_to_indices[int(lbl)].append(i)
        self.unique_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return self.pairs_per_epoch

    def _load_random_patch(self, idx):
        import torch
        _f = np.load(self.residual_paths[idx])
        residual = _f['residual'].astype(np.float16) if 'residual' in _f else _f
        patches = extract_patches(residual, patch_size=self.patch_size,
                                  max_patches=1,
                                  random_state=self.rng.randint(0, 2**31))
        patch = patches[0]
        if patch.ndim == 2:
            return torch.from_numpy(patch).float().unsqueeze(0)
        return torch.from_numpy(patch).float().permute(2, 0, 1)

    def __getitem__(self, idx):
        is_positive = self.rng.rand() < 0.5
        label1 = self.rng.choice(self.unique_labels)
        idx1 = self.rng.choice(self.label_to_indices[label1])

        if is_positive:
            idx2 = self.rng.choice(self.label_to_indices[label1])
            pair_label = 1
        else:
            label2 = self.rng.choice([l for l in self.unique_labels if l != label1])
            idx2 = self.rng.choice(self.label_to_indices[label2])
            pair_label = 0

        anchor = self._load_random_patch(idx1)
        other = self._load_random_patch(idx2)
        if self.transform:
            anchor = self.transform(anchor)
            other = self.transform(other)
        return anchor, other, pair_label
