"""
RAISE Dataset Download Script
===============================

Supplemental dataset: ~8,000 RAW images from 30 cameras.
Used as a clean-image upper-bound baseline for PRNU validation.

Usage
-----
    python data/scripts/download_raise.py --output-dir data/raw/raise
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RAISE_INFO = {
    "url": "http://loki.disi.unitn.it/RAISE/",
    "paper": "Dang-Nguyen et al., MMSys 2015",
    "size_gb": "~200 GB (RAW format)",
    "n_images": 8156,
    "n_cameras": 3,  # Nikon D40, D90, D7000 in original release
    "format": "NEF (Nikon RAW)",
}


def setup_raise(output_dir: Path) -> dict:
    """Setup RAISE dataset.

    Returns device index if data exists, otherwise prints instructions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing data
    raw_extensions = {".nef", ".cr2", ".arw", ".dng", ".tif", ".tiff"}
    image_count = sum(
        1 for f in output_dir.rglob("*")
        if f.suffix.lower() in raw_extensions
    )

    if image_count > 50:
        logger.info(f"RAISE dataset present: {image_count} RAW images")
        return _index_raise(output_dir)

    print("\n" + "=" * 70)
    print("RAISE DATASET — Download Instructions (SUPPLEMENTAL)")
    print("=" * 70)
    print()
    print(f"URL: {RAISE_INFO['url']}")
    print(f"Size: {RAISE_INFO['size_gb']} (RAW format — very large)")
    print()
    print("RAISE provides uncompressed RAW images with the strongest")
    print("possible PRNU signal. Use to validate PRNU extraction")
    print("before testing on compressed images.")
    print()
    print("This dataset is OPTIONAL — needed only for baseline validation.")
    print(f"Extract to: {output_dir}")
    print("=" * 70)

    return {}


def _index_raise(data_dir: Path) -> dict:
    """Index RAISE images by device."""
    raw_extensions = {".nef", ".cr2", ".arw", ".dng", ".tif", ".tiff"}
    device_index = {}

    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() not in raw_extensions:
            continue
        # RAISE naming: r{camera_id}{image_number}.NEF
        device_id = img_path.parent.name
        device_index.setdefault(device_id, []).append(str(img_path))

    return device_index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Setup RAISE Dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/raise"))
    args = parser.parse_args()

    setup_raise(args.output_dir)
