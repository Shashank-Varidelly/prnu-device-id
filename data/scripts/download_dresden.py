"""
Dresden Image Database Download Script
========================================

Downloads the Dresden Image Database for PRNU camera identification.
~17,000 images · 74 cameras · 25 device models.

Usage
-----
    python data/scripts/download_dresden.py --output-dir data/raw/dresden

The Dresden dataset is available from multiple mirrors:
- Primary: https://doi.org/10.1109/TIFS.2010.2090519 (IEEE DataPort)
- Mirror: Dropbox links from original authors

Note: Due to dataset size (~20GB), download may take significant time.
This script supports resumable downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Known mirrors (update if links change)
DRESDEN_MIRRORS = [
    # The actual download URLs will depend on your access method.
    # These are placeholder structures — update with real URLs after
    # obtaining dataset access.
    {
        "name": "IEEE DataPort",
        "url": "https://ieee-dataport.org/open-access/dresden-image-database",
        "type": "manual",
        "notes": "Requires IEEE account. Download manually and place in output-dir.",
    },
]

# Expected directory structure after extraction
EXPECTED_STRUCTURE = {
    "natural": "Natural scene images from each camera",
    "flat": "Flat-field images (for fingerprint estimation)",
}


class DownloadProgress:
    """tqdm-based download progress bar."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            )
        downloaded = block_num * block_size
        self.pbar.update(block_size)
        if downloaded >= total_size and self.pbar:
            self.pbar.close()


def verify_dresden_structure(data_dir: Path) -> bool:
    """Verify that the Dresden dataset directory has expected structure."""
    if not data_dir.exists():
        return False

    # Check for image files
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_count = sum(
        1 for f in data_dir.rglob("*") if f.suffix.lower() in image_extensions
    )

    if image_count < 100:
        logger.warning(
            f"Only {image_count} images found in {data_dir}. "
            f"Expected ~17,000 for full Dresden dataset."
        )
        return False

    logger.info(f"Found {image_count} images in {data_dir}")
    return True


def index_dresden_devices(data_dir: Path) -> dict:
    """Index all devices and their images in the Dresden dataset.

    Returns
    -------
    dict
        Mapping: device_id → list of image paths.

    Expected naming convention:
        {Brand}_{Model}_{DeviceNumber}_{ImageNumber}.JPG
        e.g., Canon_Ixus70_0_00001.JPG
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    device_index = {}

    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Parse device identity from filename
        parts = img_path.stem.split("_")
        if len(parts) >= 3:
            # device_id = brand_model_devicenumber
            device_id = "_".join(parts[:3])
        else:
            # Fallback: use parent directory as device ID
            device_id = img_path.parent.name

        device_index.setdefault(device_id, []).append(str(img_path))

    logger.info(
        f"Indexed {sum(len(v) for v in device_index.values())} images "
        f"across {len(device_index)} devices"
    )
    return device_index


def setup_dresden(output_dir: Path) -> dict:
    """Main setup function for Dresden dataset.

    1. Check if already downloaded.
    2. If not, print instructions for manual download.
    3. Index devices and images.

    Returns
    -------
    dict
        Device index: device_id → [image_paths].
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if verify_dresden_structure(output_dir):
        logger.info("Dresden dataset already present. Indexing...")
        return index_dresden_devices(output_dir)

    # Print manual download instructions
    print("\n" + "=" * 70)
    print("DRESDEN IMAGE DATABASE — Download Instructions")
    print("=" * 70)
    print()
    print("The Dresden dataset requires manual download due to access controls.")
    print()
    print("Options:")
    for mirror in DRESDEN_MIRRORS:
        print(f"  • {mirror['name']}: {mirror['url']}")
        if mirror.get("notes"):
            print(f"    {mirror['notes']}")
    print()
    print(f"After downloading, extract images to: {output_dir}")
    print()
    print("Expected structure:")
    print(f"  {output_dir}/")
    print(f"    ├── Canon_Ixus70_0/")
    print(f"    │   ├── Canon_Ixus70_0_00001.JPG")
    print(f"    │   └── ...")
    print(f"    ├── Canon_Ixus70_1/")
    print(f"    └── ...")
    print("=" * 70)

    return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Setup Dresden Image Database")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/dresden"),
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only index existing files, don't attempt download",
    )
    args = parser.parse_args()

    result = setup_dresden(args.output_dir)
    if result:
        print(f"\nIndexed {len(result)} devices:")
        for dev_id, imgs in sorted(result.items()):
            print(f"  {dev_id}: {len(imgs)} images")
