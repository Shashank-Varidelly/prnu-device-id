"""
Build Device-Stratified Train/Val/Test Splits
===============================================

CRITICAL: Splits are by DEVICE IDENTITY, not by image. If the same
camera's images appear in both train and test, every result is invalid.

This script:
1. Reads indexed datasets (Dresden + VISION)
2. Assigns devices to train/val/test (70/15/15 by default)
3. Writes splits.json — READ-ONLY after Sprint 1

Usage
-----
    python data/scripts/build_splits.py \\
        --dresden-dir data/raw/dresden \\
        --vision-dir data/raw/vision \\
        --output data/splits.json \\
        --seed 42

After running, commit splits.json and do not modify it.
All 12 experiments MUST use the same fixed splits.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from download_dresden import index_dresden_devices
from download_vision import index_vision_devices


def stratified_device_split(
    device_index: Dict[str, list],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_images_per_device: int = 10,
    seed: int = 42,
) -> Dict[str, Dict[str, list]]:
    """Split images by device identity.

    ALL images from a given device go to exactly ONE split.
    Within each device's images, images are further split for
    the train/val/test partition.

    Parameters
    ----------
    device_index : dict
        device_id → list of image paths.
    train_ratio, val_ratio, test_ratio : float
        Split ratios (must sum to 1.0).
    min_images_per_device : int
        Devices with fewer images are excluded.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {"train": {dev: [paths]}, "val": {dev: [paths]}, "test": {dev: [paths]}}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = np.random.RandomState(seed)

    # Filter devices with enough images
    eligible = {
        dev: imgs for dev, imgs in device_index.items()
        if len(imgs) >= min_images_per_device
    }

    if len(eligible) < len(device_index):
        excluded = len(device_index) - len(eligible)
        logger.warning(
            f"Excluded {excluded} devices with < {min_images_per_device} images"
        )

    logger.info(f"Splitting {len(eligible)} devices...")

    splits = {"train": {}, "val": {}, "test": {}}

    for device_id, all_paths in eligible.items():
        # Shuffle images for this device
        paths = list(all_paths)
        rng.shuffle(paths)

        n = len(paths)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Test gets the remainder
        n_test = n - n_train - n_val

        if n_test < 1:
            # Redistribute: ensure at least 1 per split
            n_train = max(1, n - 2)
            n_val = 1
            n_test = n - n_train - n_val

        splits["train"][device_id] = paths[:n_train]
        splits["val"][device_id] = paths[n_train : n_train + n_val]
        splits["test"][device_id] = paths[n_train + n_val:]

    return splits


def stratified_device_split_vision(
    device_index: Dict[str, Dict[str, list]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_images: int = 5,
    seed: int = 42,
) -> dict:
    """Split VISION dataset by device, preserving variant information.

    For VISION, we split original images into train/val/test, and
    variant images (WhatsApp, Flickr) go entirely to test for
    distribution shift evaluation.

    Returns
    -------
    dict
        {
            "train": {dev: [original_train_paths]},
            "val": {dev: [original_val_paths]},
            "test": {dev: [original_test_paths]},
            "test_whatsapp": {dev: [whatsapp_paths]},
            "test_flickr": {dev: [flickr_paths]},
        }
    """
    rng = np.random.RandomState(seed)

    splits = {
        "train": {},
        "val": {},
        "test": {},
        "test_whatsapp": {},
        "test_flickr": {},
    }

    for device_id, variants in device_index.items():
        # Split original images
        original = variants.get("original", [])
        if len(original) < min_images:
            continue

        rng.shuffle(original)
        n = len(original)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        splits["train"][device_id] = original[:n_train]
        splits["val"][device_id] = original[n_train : n_train + n_val]
        splits["test"][device_id] = original[n_train + n_val:]

        # Variant images → test only
        if "whatsapp" in variants:
            splits["test_whatsapp"][device_id] = variants["whatsapp"]
        if "flickr" in variants:
            splits["test_flickr"][device_id] = variants["flickr"]

    return splits


def build_splits(
    dresden_dir: Path,
    vision_dir: Path,
    output_path: Path,
    seed: int = 42,
) -> dict:
    """Build complete splits.json from both datasets.

    Returns
    -------
    dict
        Complete splits structure with both datasets.
    """
    all_splits = {}

    # --- Dresden ---
    if dresden_dir.exists():
        logger.info("Building Dresden splits...")
        dresden_index = index_dresden_devices(dresden_dir)
        if dresden_index:
            all_splits["dresden"] = stratified_device_split(
                dresden_index, seed=seed
            )
            n_devs = len(all_splits["dresden"]["train"])
            logger.info(f"  Dresden: {n_devs} devices split")

    # --- VISION ---
    if vision_dir.exists():
        logger.info("Building VISION splits...")
        vision_index = index_vision_devices(vision_dir)
        if vision_index:
            all_splits["vision"] = stratified_device_split_vision(
                vision_index, seed=seed
            )
            n_devs = len(all_splits["vision"]["train"])
            logger.info(f"  VISION: {n_devs} devices split")

    # --- Write splits.json ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_splits, f, indent=2)

    logger.info(f"Splits written to {output_path}")
    logger.info("⚠️  This file is READ-ONLY after Sprint 1!")

    # Summary
    _print_split_summary(all_splits)

    return all_splits


def _print_split_summary(splits: dict):
    """Print a summary of the splits."""
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)

    for dataset_name, dataset_splits in splits.items():
        print(f"\n{dataset_name.upper()}:")
        for split_name, devices in dataset_splits.items():
            n_devices = len(devices)
            n_images = sum(len(imgs) if isinstance(imgs, list) else 0
                          for imgs in devices.values())
            print(f"  {split_name:20s}: {n_devices:3d} devices, {n_images:6d} images")

    print("\n" + "=" * 60)
    print("⚠️  CRITICAL: Do not modify splits.json after Sprint 1!")
    print("   All 12 experiments must use identical splits.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Build device-stratified train/val/test splits"
    )
    parser.add_argument("--dresden-dir", type=Path, default=Path("data/raw/dresden"))
    parser.add_argument("--vision-dir", type=Path, default=Path("data/raw/vision"))
    parser.add_argument("--output", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_splits(args.dresden_dir, args.vision_dir, args.output, seed=args.seed)
