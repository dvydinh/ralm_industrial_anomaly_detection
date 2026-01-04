# RAML: Resolution-Adaptive Margin Learning for Industrial Anomaly Detection

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official source code for **RAML (Resolution-Adaptive Margin Learning)**, a novel method for industrial anomaly detection that combines visual features from CLIP with adaptive margin learning.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key contributions](#key-contributions)
- [Datasets](#datasets)
- [Pre-trained model](#pre-trained-model)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental results](#experimental-results)
- [Project structure](#project-structure)
- [References](#references)
- [License](#license)

---

## Overview

Industrial anomaly detection faces several challenges: high-resolution images with very small defects, diverse defect types across product categories, and extremely limited labeled anomaly samples. Existing zero-shot methods such as WinCLIP lack the ability to learn decision boundaries, while supervised methods typically use fixed margins that cannot adapt to the characteristics of each category.

RAML addresses these limitations through:
- Adaptive margin that self-adjusts based on feature distribution and resolution loss
- Multi-scale feature extraction to detect defects at various sizes
- Cross-scale attention to aggregate information across scales
- Auto-difficulty weighting to balance training across categories

**Summary of results:**
| Evaluation | AUROC (%) |
|------------|-----------|
| MVTec AD (20% test split) | 94.09 |
| VisA (zero-shot transfer) | 72.44 |

---

## Architecture

### System overview

```
Input image (H x W)
       |
       v
+------------------+
| Multi-Scale      |  Scale 1: 1x1 (global)    -> 1 x 512-dim
| Feature Pyramid  |  Scale 2: 2x2 patches     -> 4 x 512-dim  
|                  |  Scale 3: 4x4 patches     -> 16 x 512-dim
+------------------+
       |
       v (21 CLIP forward passes total)
+------------------+
| Cross-Scale      |  Bidirectional attention across scales
| Attention        |  Combining global <-> local context
+------------------+
       |
       v
+------------------+
| Text-Visual      |  Text prompts: "flawless {category}" vs
| Fusion           |  "damaged {category} with defects"
+------------------+
       |
       v
+------------------+
| MACCL Loss       |  Center Loss + Margin Loss + Contrastive Loss
+------------------+
       |
       v
   Anomaly Score
```

### Multi-scale feature pyramid

CLIP processes fixed 224x224 inputs, causing significant information loss when applied to high-resolution industrial images (typically 700-1024 pixels). The multi-scale pyramid extracts features at three levels:

| Scale | Patch grid | Number of patches | Effective resolution |
|-------|------------|-------------------|---------------------|
| 1 | 1x1 | 1 | 224x224 (global context) |
| 2 | 2x2 | 4 | equivalent to 448x448 |
| 3 | 4x4 | 16 | equivalent to 896x896 |

Each patch is resized to 224x224 and processed through CLIP, yielding 21 feature vectors (1 + 4 + 16) per image.

### Cross-scale attention

Features from different scales interact through cross-attention:

```
f1' = CrossAttn(f1 -> F2)    # Global attends to 2x2 patches
f2' = CrossAttn(f2 -> F3)    # 2x2 attends to 4x4 patches  
f3' = CrossAttn(f3 -> f1)    # 4x4 attends to global
```

This allows local patches to understand global context and vice versa.

### Resolution-adaptive margin formula

The core contribution is a margin that adapts to both data distribution and preprocessing effects:

```
m = m_base + lambda_sigma * sigma + lambda_resolution * (1 - r/r_max)
```

Where:
- `m_base = 0.5`: base margin
- `lambda_sigma = 0.3`: weight for feature standard deviation
- `lambda_resolution = 0.3`: weight for resolution penalty
- `sigma`: running standard deviation of normal features (updated via EMA)
- `r = 224`: model input resolution
- `r_max = 900`: original image resolution (MVTec average)

**Computation example:**
```
sigma = 0.6 (measured from normal features)
resolution_penalty = 1 - 224/900 = 0.751

m = 0.5 + 0.3 * 0.6 + 0.3 * 0.751
  = 0.5 + 0.18 + 0.225
  = 0.905
```

### MACCL loss function

Margin-Aware Center-Contrastive Loss combines three components:

```
L_total = alpha * L_center + beta * L_margin + gamma * L_contrastive
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| Center loss | alpha = 1.0 | Pull normal features toward the center |
| Margin loss | beta = 1.0 | Push anomaly features beyond the adaptive margin |
| Contrastive loss | gamma = 0.5 | Improve normal/anomaly discriminability |

### Auto-difficulty tracker

Categories with above-average loss receive increased weights:

```
w_cat = 1 + scale * (L_cat - L_global) / L_global
```

Configuration: `momentum = 0.7`, `weight_scale = 0.5`, weights are clipped to the range [0.5, 2.0].

### Inference logic

The final anomaly score combines the visual classifier and text similarity:

```
s_final = 0.85 * s_visual + 0.15 * s_text
```

Where:
- `s_visual`: sigmoid output from the trained classifier
- `s_text`: cosine similarity difference between anomaly and normal text embeddings

---

## Key contributions

1. **Resolution-adaptive margin**: the first formulation combining feature distribution (sigma) and resolution loss into margin computation
2. **Multi-scale feature pyramid**: extracting 21 patches to achieve effective resolution of 896x896 with a 224x224 backbone
3. **Cross-scale attention**: bidirectional attention fusion across scales
4. **Auto-difficulty tracker**: automatically adjusting loss weights based on per-category difficulty
5. **Text-visual fusion**: residual cross-attention with learnable fusion scale

---

## Datasets

### MVTec Anomaly Detection Dataset (primary)

| Property | Value |
|----------|-------|
| Authors | P. Bergmann, M. Fauser, D. Sattlegger, C. Steger |
| Published | CVPR 2019 |
| Paper | "MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" |
| Images | 5,354 total (3,629 train, 1,725 test) |
| Categories | 15 (5 textures, 10 objects) |
| Resolution | 700x700 to 1024x1024 |
| License | CC BY-NC-SA 4.0 |
| Link | https://www.mvtec.com/company/research/datasets/mvtec-ad |

**Category list:** bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

**Data split used:**
- Training: 100% official train set + 80% official test set (stratified)
- Testing: remaining 20% of official test set (held out)

### VisA Dataset (transfer evaluation)

| Property | Value |
|----------|-------|
| Authors | Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer |
| Published | ECCV 2022 |
| Paper | "SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation" |
| Images | 10,821 total |
| Categories | 12 |
| License | CC BY 4.0 |
| Link | https://github.com/amazon-science/spot-diff |

**Category list:** candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum

**Usage:** evaluation only for zero-shot transfer (model trained on MVTec, evaluated on VisA without fine-tuning)

---

## Pre-trained model

### CLIP backbone

| Property | Value |
|----------|-------|
| Model | CLIP ViT-B/16 |
| Provider | OpenAI |
| Parameters | 86M (frozen during training) |
| Input size | 224x224 |
| Feature dimension | 512 |
| Download link | https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ec59a0ca1a1b55/ViT-B-16.pt |

The script automatically downloads the model via the `open_clip` library. For offline environments, download manually and place in the cache directory.

---

## System requirements

### Software

| Package | Version |
|---------|---------|
| Python | 3.8+ |
| PyTorch | 1.10+ |
| open_clip_torch | latest |
| torchvision | compatible with PyTorch |
| numpy | 1.20+ |
| scikit-learn | 0.24+ |
| Pillow | 8.0+ |
| tqdm | 4.60+ |

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 8GB VRAM | Tesla T4/P100 with 16GB VRAM |
| RAM | 16GB | 32GB |
| Storage | 10GB free | 20GB free |

---

## Installation

```bash
# clone the repository
git clone https://github.com/dvydinh/ralm_industrial_anomaly_detection.git
cd ralm_industrial_anomaly_detection

# create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # linux/mac
venv\Scripts\activate     # windows

# install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch scikit-learn pillow tqdm
```
