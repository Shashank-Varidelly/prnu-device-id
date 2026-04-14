import json, numpy as np
from pathlib import Path
import sys; sys.path.insert(0, "src")
import cv2, gc
from prnu_pipeline import PRNUPipeline

SPLITS  = Path("data/splits_demo.json")
OUT_DIR = Path("data/processed/fingerprints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

splits   = json.loads(SPLITS.read_text())
pipeline = PRNUPipeline(denoiser='wavelet')

for device, paths in splits["dresden"]["fingerprint"].items():
    if (OUT_DIR / f"{device}.npy").exists():
        print(f"Skipping {device} — already done")
        continue
    print(f"Building fingerprint for {device}...")
    numerator, denominator, TARGET = None, None, None
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        if TARGET is None: TARGET = img.shape[:2]
        img = cv2.resize(img, (TARGET[1], TARGET[0]))
        img_f = img.astype(np.float64) / 255.0
        w = pipeline.extract_noise_residual(img_f)
        if numerator is None:
            numerator = w * img_f
            denominator = img_f ** 2
        else:
            numerator += w * img_f
            denominator += img_f ** 2
        del img, img_f, w; gc.collect()
    denominator = np.where(denominator == 0, 1.0, denominator)
    fp = numerator / denominator
    np.save(OUT_DIR / f"{device}.npy", fp)
    print(f"  Saved: {device}  shape={fp.shape}")
    del numerator, denominator, fp; gc.collect()

print("Done.", list(OUT_DIR.glob("*.npy")))