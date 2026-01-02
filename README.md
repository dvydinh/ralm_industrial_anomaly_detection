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
