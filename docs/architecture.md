# Architecture & Pipeline Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRNU Camera Identification Pipeline             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐             │
│  │ Raw      │───▶│ Preprocessing│───▶│ PRNU          │             │
│  │ Image    │    │ (resize,     │    │ Extraction    │             │
│  │ I(x,y)   │    │  color norm) │    │ W = I - F(I)  │             │
│  └──────────┘    └──────────────┘    └───────┬───────┘             │
│                                              │                      │
│                          ┌───────────────────┼───────────────┐      │
│                          ▼                   ▼               ▼      │
│                  ┌──────────────┐   ┌──────────────┐ ┌────────────┐│
│                  │ Method A:    │   │ Method B:    │ │ Method C:  ││
│                  │ NCC Baseline │   │ CNN (ResNet  │ │ Siamese    ││
│                  │              │   │ -18)         │ │ Network    ││
│                  │ Compare W_q  │   │ Classify     │ │ Embed +    ││
│                  │ against K_d  │   │ patch into   │ │ nearest    ││
│                  │ for all d    │   │ device class │ │ neighbor   ││
│                  └──────┬───────┘   └──────┬───────┘ └─────┬──────┘│
│                         │                  │               │        │
│                         ▼                  ▼               ▼        │
│                  ┌─────────────────────────────────────────────────┐│
│                  │              Attribution Decision               ││
│                  │  Device ID + Confidence + Calibrated Score      ││
│                  └─────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## PRNU Extraction Pipeline

### Step 1: Wavelet Denoising — F(I)

Following Lukas, Fridrich & Goljan (2006), Appendix A:

1. Apply multi-level (L=4) DWT using Daubechies-4 ("db4") wavelet
2. Estimate noise variance σ_n from finest-level HH sub-band:
   σ_n = median(|d|) / 0.6745  (Donoho & Johnstone, 1994)
3. Apply BayesShrink threshold to each detail sub-band:
   T = σ_n² / σ_x  where σ_x = sqrt(max(σ_d² - σ_n², 0))
4. Soft-threshold: sign(x) · max(|x| - T, 0)
5. Inverse DWT → denoised image F(I)

### Step 2: Noise Residual Extraction

W(x,y) = I(x,y) - F(I(x,y))

Applied independently per color channel (RGB).

### Step 3: Fingerprint Estimation — K_d

Maximum-likelihood estimator from M training images:

K̂_d(x,y) = Σ_i W_i(x,y) · I_i(x,y) / Σ_i I_i²(x,y)

### Step 4: Attribution — NCC

ρ(W_q, I_q · K_d) = (W_q - W̄_q) · (I_q·K_d - mean(I_q·K_d)) / 
                      ||W_q - W̄_q|| · ||I_q·K_d - mean(I_q·K_d)||

## Data Pipeline

```
datasets (external)          data/raw/           data/processed/
  ┌──────────────┐     ┌──────────────────┐    ┌────────────────┐
  │ Dresden DB   │────▶│ dresden/         │───▶│ residuals/     │
  │ VISION       │────▶│ vision/          │    │   *.npy        │
  │ RAISE        │────▶│ raise/           │    │ fingerprints/  │
  └──────────────┘     └──────────────────┘    │   *.npy        │
       (download          (symlinks,            └────────────────┘
        scripts)           raw images)            (cached for
                                                   fast loading)
```

## Model Architectures

### CNN (ResNet-18)

```
Input: 1×128×128 noise residual patch
  │
  ├── Conv2d(1→64, 7×7, stride=2) + BN + ReLU
  ├── MaxPool(3×3, stride=2)
  ├── ResBlock × 2 (64→64)
  ├── ResBlock × 2 (64→128, stride=2)
  ├── ResBlock × 2 (128→256, stride=2)
  ├── ResBlock × 2 (256→512, stride=2)
  ├── AdaptiveAvgPool → 512-dim
  ├── Dropout(0.3)
  └── Linear(512→num_classes)
Output: logits (num_classes)
```

### Siamese/Triplet Network

```
Shared Encoder (ResNet-18 backbone):
  Input: 1×128×128 patch → 512-dim features
  │
  ├── Linear(512→256) + ReLU
  └── Linear(256→128) + L2-normalize
Output: 128-dim unit embedding

Loss: Contrastive or Triplet Margin
```

## Experiment Dependencies

```
Sprint 1 ──▶ Sprint 2 ──▶ Sprint 3 ──▶ Sprint 4 ──▶ Sprint 5
  │             │             │             │             │
  ├─ Repo       ├─ A1 (NCC)  ├─ C1 (JPEG)  ├─ A3 (aug)  ├─ Final
  ├─ Data       ├─ B2 (abl)  ├─ C2 (resize)├─ C3 (cal)  ├─ Report
  ├─ PRNU       ├─ CNN base  ├─ Siamese    ├─ D1 (dev)  ├─ Demo
  └─ splits     └─ B1 (part) ├─ A2 (shift) ├─ D2 (err)  └─ Polish
                              └─ B3 (patch) └─ D3 (time)
```
