# PRNU Device-Level Camera Identification Under Social Media Compression

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Machine Learning Course · Final Project · San José State University · Spring 2026**

Source attribution at the device level using Photo Response Non-Uniformity (PRNU) — a sensor-dependent noise pattern caused by microscopic manufacturing variations. This project compares classical normalized-correlation matching with CNN and Siamese-network approaches, and systematically measures robustness under real-world social media compression.

## Team

| Role | Name | Email | Owns |
|------|------|-------|------|
| P1 — PRNU Core Lead | Geeshitha Mamidi | geeshitha.mamidi@sjsu.edu | `prnu/` |
| P2 — ML Models Lead | Vikaramaditya Baddam | srisaivikramadithyareddy.baddam@sjsu.edu | `models/` |
| P3 — Evaluation Lead | Shashank Varidelly | shashank.varidelly@sjsu.edu | `evaluation/` |
| P4 — Infra & Data Lead | Rishikesh Aluguvelli | rishikeshreddy.aluguvelli@sjsu.edu | `data/` + CI |

## Quick Start

### 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/<org>/prnu-device-id.git
cd prnu-device-id

# Create conda environment
conda env create -f environment.yml
conda activate prnu-device-id

# Install PyTorch (CPU-only — use CUDA variant if you have a GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install project in development mode
pip install -e ".[dev]"
```

### 2. Dataset Setup

```bash
# Download and setup datasets (follow printed instructions)
python data/scripts/download_dresden.py --output-dir data/raw/dresden
python data/scripts/download_vision.py --output-dir data/raw/vision

# Build device-stratified splits (READ-ONLY after this point!)
python data/scripts/build_splits.py \
    --dresden-dir data/raw/dresden \
    --vision-dir data/raw/vision \
    --output data/splits.json \
    --seed 42
```

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Run Experiments

```bash
# Group A: Core performance
python experiments/group_a.py a1 --dataset dresden
python experiments/group_a.py a2
python experiments/group_a.py a3 --dataset dresden

# Group B: Method comparison
python experiments/group_b.py b2 --dataset dresden
python experiments/group_b.py b3 --dataset dresden

# Group C: Robustness
python experiments/group_c.py c1 --dataset dresden
python experiments/group_c.py c2 --dataset dresden

# Group D: Analysis
python experiments/group_d.py all
```

### 5. Live Demo

```bash
python demo.py --image path/to/test_image.jpg --fingerprint-dir data/processed/fingerprints/
```

## Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐     ┌────────────┐
│  Raw Image   │────▶│  Wavelet     │────▶│  Noise Residual      │────▶│ Attribution│
│  I           │     │  Denoise F(I)│     │  W = I - F(I)        │     │ Method     │
└──────────────┘     └──────────────┘     └──────────────────────┘     └────────────┘
                                                   │                         │
                                                   ▼                         ▼
                                          ┌────────────────┐       ┌─────────────────┐
                                          │ Fingerprint    │       │ (A) NCC         │
                                          │ K_d = MLE      │       │ (B) CNN         │
                                          │ from M images  │       │ (C) Siamese     │
                                          └────────────────┘       └─────────────────┘
```

## Three Attribution Methods

| Method | Type | How It Works |
|--------|------|-------------|
| **NCC** | Classical | Normalized cross-correlation between query residual and stored fingerprints. Neyman-Pearson threshold. |
| **CNN** | Learned | ResNet-18 on PRNU residual patches → multiclass logits → argmax prediction. |
| **Siamese** | Learned | Contrastive/triplet network → embedding space → nearest-device matching. |

## Experiments (12 Total)

| ID | Description | Key Variable | Dataset |
|----|-------------|-------------|---------|
| A1 | Clean-domain attribution | Baseline accuracy | Dresden + VISION |
| A2 | Train clean → test social-media | Distribution shift | Both |
| A3 | Compression-aware training | JPEG augmentation | Both |
| B1 | NCC vs CNN vs Siamese | Method comparison | Both |
| B2 | Wiener vs wavelet denoiser | Denoiser choice | Dresden |
| B3 | Patch-size ablation (64/128/256) | Resolution | Dresden |
| C1 | JPEG quality sweep Q=95→30 | Quality factor | Dresden |
| C2 | Resize / downsample stress test | Scale factor | Dresden |
| C3 | Confidence calibration (ECE, Brier) | Calibration error | Both |
| D1 | Per-device robustness | Per-device FRR | VISION |
| D2 | Error taxonomy (same vs cross model) | Confusion structure | Both |
| D3 | Runtime profiling | Seconds per image | Both |

## Repository Structure

```
prnu-device-id/
├── data/
│   ├── raw/                    # Symlinks/download scripts — never commit images
│   ├── processed/              # Cached noise residuals (.npy)
│   ├── scripts/                # Download & preprocessing scripts
│   │   ├── download_dresden.py
│   │   ├── download_vision.py
│   │   ├── download_raise.py
│   │   └── build_splits.py
│   └── splits.json             # Device-stratified splits — READ ONLY
├── prnu/                       # Core PRNU library
│   ├── denoise.py              # Wavelet denoising (Lukas et al. Appendix A)
│   ├── fingerprint.py          # K_d estimator + NCC scorer
│   ├── patches.py              # Patch extraction (64/128/256 px)
│   └── compress.py             # JPEG / resize compression tools
├── models/                     # ML models
│   ├── cnn.py                  # ResNet-18 encoder
│   └── siamese.py              # Siamese + contrastive/triplet loss
├── experiments/                # One script per experiment group
│   ├── group_a.py              # A1, A2, A3
│   ├── group_b.py              # B1, B2, B3
│   ├── group_c.py              # C1, C2, C3
│   └── group_d.py              # D1, D2, D3
├── evaluation/                 # Metrics & visualization
│   ├── metrics.py              # FAR, FRR, ECE, Brier, confusion
│   └── plots.py                # Publication-quality figures
├── tests/                      # pytest unit tests
├── notebooks/                  # Exploratory analysis (NOT final code)
├── journal/                    # Weekly sprint journals
├── docs/                       # Architecture diagrams, report PDF
├── demo.py                     # Live attribution demo
├── environment.yml             # Pinned conda environment
├── pyproject.toml              # Project metadata + dependencies
└── README.md
```

## Datasets

| Dataset | Role | Images | Devices | Download |
|---------|------|--------|---------|----------|
| **Dresden** | Primary | ~17,000 | 74 (25 models) | IEEE DataPort |
| **VISION** | Primary | ~34,427 | 35 smartphones | lesc.dinfo.unifi.it/VISION |
| **RAISE** | Supplemental | ~8,000 RAW | 30 cameras | loki.disi.unitn.it/RAISE |

> ⚠️ **CRITICAL**: Split by DEVICE IDENTITY, not by image. `splits.json` is read-only after Sprint 1.

## Key References

1. Lukáš, J., Fridrich, J., & Goljan, M. (2006). Digital camera identification from sensor pattern noise. *IEEE TIFS*, 1(2), 205-214.
2. Chen, M., Fridrich, J., Goljan, M., & Lukáš, J. (2008). Determining image origin and integrity using sensor noise. *IEEE TIFS*, 3(1), 74-90.
3. Shullani, D., et al. (2017). VISION: a video and image dataset for source identification. *EURASIP JIVP*.

## License

MIT License — see [LICENSE](LICENSE) for details.
