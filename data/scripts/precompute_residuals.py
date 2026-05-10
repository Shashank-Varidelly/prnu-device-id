"""
PRNU Noise Residual Pre-extraction Script
===========================================

Pre-computes and caches noise residuals to .npy files for fast
training and evaluation. Run this once after datasets are downloaded.

Usage
-----
    python data/scripts/precompute_residuals.py \\
        --splits data/splits.json \\
        --output-dir data/processed/residuals \\
        --denoiser wavelet \\
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Import after path setup
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.prnu_pipeline import PRNUPipeline


def process_single_image(
    image_path: str,
    output_dir: Path,
    denoiser: str = "wavelet",
) -> str | None:
    """Extract and save noise residual for a single image.

    Returns the output .npy path, or None on failure.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        pipeline = PRNUPipeline(denoiser=denoiser)
        residual = pipeline.extract_noise_residual(img)

        # Create output path preserving directory structure
        img_path = Path(image_path)
        rel_name = img_path.stem + ".npy"
        device_dir = output_dir / img_path.parent.name
        device_dir.mkdir(parents=True, exist_ok=True)
        out_path = device_dir / rel_name

        np.save(out_path, residual.astype(np.float32))
        return str(out_path)

    except Exception as e:
        logger.warning(f"Failed on {image_path}: {e}")
        return None


def precompute_residuals(
    splits_path: Path,
    output_dir: Path,
    denoiser: str = "wavelet",
    workers: int = 4,
    datasets: list[str] | None = None,
):
    """Pre-compute noise residuals for all images in splits.json."""
    with open(splits_path) as f:
        splits = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = list(splits.keys())

    for dataset_name in datasets:
        dataset_splits = splits.get(dataset_name, {})
        logger.info(f"Processing {dataset_name}...")

        # Collect all image paths
        all_paths = []
        for split_name, devices in dataset_splits.items():
            for device_id, paths in devices.items():
                if isinstance(paths, list):
                    all_paths.extend(paths)

        logger.info(f"  {len(all_paths)} images to process")

        ds_output = output_dir / dataset_name
        ds_output.mkdir(parents=True, exist_ok=True)

        # Process with parallel workers
        completed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_image, p, ds_output, denoiser): p
                for p in all_paths
            }

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  {dataset_name}"):
                result = future.result()
                if result:
                    completed += 1
                else:
                    failed += 1

        logger.info(f"  Done: {completed} completed, {failed} failed")

    # Save index mapping
    index_path = output_dir / "residual_index.json"
    index = {}
    for npy_file in output_dir.rglob("*.npy"):
        rel = str(npy_file.relative_to(output_dir))
        device = npy_file.parent.name
        index.setdefault(device, []).append(rel)

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Index saved: {index_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Pre-compute noise residuals")
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/residuals"))
    parser.add_argument("--denoiser", default="wavelet", choices=["wavelet", "wiener"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    precompute_residuals(
        splits_path=args.splits,
        output_dir=args.output_dir,
        denoiser=args.denoiser,
        workers=args.workers,
        datasets=args.datasets,
    )
