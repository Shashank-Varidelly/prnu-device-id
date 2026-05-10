"""
VISION Dataset Download Script
================================

Downloads the VISION dataset for PRNU camera identification under
social media compression.

~34,427 images · 35 smartphones · WhatsApp + Flickr sharing variants.

Usage
-----
    python data/scripts/download_vision.py --output-dir data/raw/vision

Reference: Shullani et al. (2017). VISION: a video and image dataset
for source identification. EURASIP J. Image and Video Processing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# VISION dataset information
VISION_INFO = {
    "url": "https://lesc.dinfo.unifi.it/VISION/",
    "paper": "Shullani et al., EURASIP JIVP, 2017",
    "size_gb": "~40 GB (full dataset with videos)",
    "images_only_gb": "~15 GB",
    "n_devices": 35,
    "n_images": 34427,
    "variants": ["original", "whatsapp", "flickr", "facebook", "youtube"],
}

# Subset structure for images
VISION_SUBSETS = {
    "flat": "Flat-field images for fingerprint estimation",
    "nat": "Natural scene images (original quality)",
    "natWA": "Natural scenes shared via WhatsApp",
    "natFB": "Natural scenes shared via Facebook",
    "natFL": "Natural scenes shared via Flickr",
}


def verify_vision_structure(data_dir: Path) -> bool:
    """Check if VISION dataset is properly extracted."""
    if not data_dir.exists():
        return False

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_count = sum(
        1 for f in data_dir.rglob("*") if f.suffix.lower() in image_extensions
    )

    if image_count < 100:
        logger.warning(f"Only {image_count} images found. Expected ~34,000.")
        return False

    logger.info(f"Found {image_count} images in {data_dir}")
    return True


def index_vision_devices(data_dir: Path) -> dict:
    """Index all devices and their images, separated by variant.

    Returns
    -------
    dict
        Nested mapping:
        {
            device_id: {
                "original": [paths],
                "whatsapp": [paths],
                "flickr": [paths],
            }
        }
    """
    image_extensions = {".jpg", ".jpeg", ".png"}
    device_index = {}

    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # VISION structure: device_brand_model/variant/image.jpg
        # Determine variant from parent directory name
        parent = img_path.parent.name.lower()
        if "wa" in parent or "whatsapp" in parent:
            variant = "whatsapp"
        elif "fl" in parent or "flickr" in parent:
            variant = "flickr"
        elif "fb" in parent or "facebook" in parent:
            variant = "facebook"
        elif "flat" in parent:
            variant = "flat"
        else:
            variant = "original"

        # Device ID from grandparent or file naming
        device_parts = img_path.parts
        # Try to find device identifier in path
        device_id = None
        for part in device_parts:
            if any(brand in part.lower() for brand in
                   ["samsung", "apple", "huawei", "lg", "sony", "motorola",
                    "oneplus", "xiaomi", "nokia", "htc", "asus", "lenovo",
                    "iphone", "galaxy", "pixel"]):
                device_id = part
                break

        if device_id is None:
            # Fallback: use parent of variant dir
            if len(device_parts) >= 3:
                device_id = device_parts[-3]
            else:
                device_id = img_path.stem.split("_")[0]

        if device_id not in device_index:
            device_index[device_id] = {}
        device_index[device_id].setdefault(variant, []).append(str(img_path))

    n_total = sum(
        len(imgs)
        for dev in device_index.values()
        for imgs in dev.values()
    )
    logger.info(f"Indexed {n_total} images across {len(device_index)} devices")
    return device_index


def setup_vision(output_dir: Path) -> dict:
    """Main setup function for VISION dataset.

    Returns
    -------
    dict
        Device index with variant separation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if verify_vision_structure(output_dir):
        logger.info("VISION dataset present. Indexing...")
        return index_vision_devices(output_dir)

    print("\n" + "=" * 70)
    print("VISION DATASET — Download Instructions")
    print("=" * 70)
    print()
    print(f"URL: {VISION_INFO['url']}")
    print(f"Paper: {VISION_INFO['paper']}")
    print(f"Size: {VISION_INFO['images_only_gb']} (images only)")
    print()
    print("The VISION dataset includes images from 35 devices with")
    print("social media sharing variants (WhatsApp, Flickr, Facebook).")
    print()
    print("Steps:")
    print("  1. Visit the URL above and request access")
    print("  2. Download the image subsets (flat + natural + variants)")
    print(f"  3. Extract to: {output_dir}")
    print()
    print("Required subsets for this project:")
    for subset, desc in VISION_SUBSETS.items():
        print(f"  • {subset}: {desc}")
    print()
    print("Expected structure:")
    print(f"  {output_dir}/")
    print(f"    ├── D01_Samsung_GalaxyS3/")
    print(f"    │   ├── nat/       (original images)")
    print(f"    │   ├── natWA/     (WhatsApp shared)")
    print(f"    │   ├── natFL/     (Flickr shared)")
    print(f"    │   └── flat/      (flat-field)")
    print(f"    ├── D02_Apple_iPhone4s/")
    print(f"    └── ...")
    print("=" * 70)

    return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Setup VISION Dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/vision"),
    )
    args = parser.parse_args()

    result = setup_vision(args.output_dir)
    if result:
        print(f"\nIndexed {len(result)} devices:")
        for dev_id, variants in sorted(result.items()):
            variant_str = ", ".join(f"{k}: {len(v)}" for k, v in variants.items())
            print(f"  {dev_id}: {variant_str}")
