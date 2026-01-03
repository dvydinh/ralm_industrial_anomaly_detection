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
