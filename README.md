# Resolution-aware learning with margin-conditioned losses for CLIP-based industrial anomaly detection

## Abstract

Vision-language models such as CLIP have demonstrated strong zero-shot anomaly detection capabilities on industrial inspection benchmarks. However, directly applying frozen CLIP features to anomaly classification yields suboptimal performance because the pre-trained feature space is not aligned to the fine-grained normal/anomaly boundary required by manufacturing inspection. This work proposes a resolution-aware multi-scale learning framework (RAML) that fine-tunes a lightweight detection head on top of frozen CLIP features. The method introduces three key components: (1) a multi-scale cross-attention pyramid that fuses global, medium, and fine-grained CLIP patch features; (2) a text-visual fusion module that integrates category-specific prompt embeddings into the visual representation; and (3) a composite loss function combining center loss, input-size-conditioned margin loss, and supervised contrastive loss. On the MVTec AD benchmark, RAML achieves 94.15% image-level AUROC, improving over the zero-shot WinCLIP baseline by +13.29 percentage points, while maintaining a lightweight trainable parameter count.

## 1. Introduction

Industrial anomaly detection requires identifying defective products from images of manufactured goods. Traditional approaches rely on one-class classification or reconstruction-based methods trained exclusively on normal samples. Recent advances in vision-language models, particularly CLIP (Radford et al., 2021), have enabled zero-shot anomaly detection through text-visual similarity scoring, as demonstrated by WinCLIP (Jeong et al., 2023).

Despite their generality, zero-shot methods suffer from two limitations in the industrial setting. First, the frozen CLIP feature space does not capture the fine-grained distinctions between subtle manufacturing defects and normal surface variations. Second, single-scale feature extraction discards spatial information that is critical for detecting localized anomalies such as scratches, dents, or contamination.

This work addresses both limitations by proposing a multi-scale feature extraction and fusion pipeline that operates on frozen CLIP features, combined with a composite loss function designed to simultaneously tighten the normal feature cluster and push anomaly features beyond an adaptive margin boundary.

## 2. Method

### 2.1 Multi-scale feature extraction

Given an input image, features are extracted at three spatial scales using a frozen CLIP ViT-B/16 encoder:

- **Scale 1 (1x1):** The full image is encoded to produce a single global feature vector (B, 512).
- **Scale 2 (2x2):** The image is divided into 4 non-overlapping patches, each encoded independently, producing (B, 4, 512).
- **Scale 3 (4x4):** The image is divided into 16 non-overlapping patches, each encoded independently, producing (B, 16, 512).

All CLIP parameters remain frozen throughout training. Only the downstream detection modules are optimized.

### 2.2 Cross-scale attention pyramid

The three scale features are fused through a bidirectional cross-attention pyramid:

1. The global feature (scale 1) attends to scale 2 patches via cross-attention.
2. The scale 2 aggregate attends to scale 3 patches.
3. The scale 3 aggregate attends back to the global and scale 2 context.

The three attended outputs are concatenated and projected through a two-layer MLP to produce a single fused representation of dimension 256.

### 2.3 Text-visual fusion

Category-specific text prompts (e.g., "a photo of a normal bottle" / "a photo of a damaged bottle") are encoded by the frozen CLIP text encoder and projected into the visual feature space. The fused visual feature attends to the normal and anomaly text embeddings via cross-attention with a learnable residual scale, allowing the model to incorporate semantic context from the text modality.

### 2.4 Composite loss function

The total training objective combines three losses:

**L_total = alpha * L_center + beta * L_margin + gamma * L_contrastive**

- **Center loss (L_center):** Minimizes the distance between normal-sample features and a learnable center, encouraging a compact normal cluster.
- **Input-size-conditioned margin loss (L_margin):** A hinge loss that pushes anomaly features beyond an adaptive margin from the normal center. The margin is defined as m = m_base + lambda_sigma * sigma + lambda_r * (1 - r/r_max), where sigma is the running standard deviation of normal features and r is the retained resolution ratio.
- **Supervised contrastive loss (L_contrastive):** Attracts features of the same class and repels features of different classes in the embedding space (Khosla et al., 2020).

### 2.5 Inference scoring

The final anomaly score blends two signals:

- **Visual score:** Sigmoid output of the classification head applied to the refined feature.
- **Text score:** Softmax-normalized cosine similarity between patch features and the anomaly text prompt, combining global and local (top-k patch) similarities.

The blended score is computed as: score = (w_v * visual_score + w_t * text_score) / (w_v + w_t), with default weights w_v = 0.85, w_t = 0.15.

## 3. Experiments

### 3.1 Setup

All experiments are conducted on the MVTec Anomaly Detection dataset (Bergmann et al., 2019), which contains 15 categories of industrial products and textures with various defect types. The dataset is split with seed=42 for reproducibility. Models are trained for 25 epochs using AdamW (lr=5e-5, weight decay=0.02) with mixed-precision training on a single GPU.

### 3.2 Evaluation metrics

Models are evaluated using five image-level metrics: area under the receiver operating characteristic curve (AUROC), average precision (AP), maximum F1-score (F1-max), precision at the optimal F1 threshold, and recall at the optimal F1 threshold. All metrics are macro-averaged across the 15 categories.

### 3.3 Main results

| Method | AUROC | AP | F1-max | Precision | Recall |
|--------|------:|---:|-------:|----------:|-------:|
| WinCLIP (zero-shot baseline) | 80.86 | 91.67 | 89.89 | 86.55 | 94.83 |
| RAML (fixed margin ablation) | 93.20 | 96.69 | 93.13 | 94.13 | 93.68 |
| RAML (strong margin ablation) | 93.66 | 96.61 | 94.06 | 93.04 | 95.95 |
| **RAML (full model)** | **94.15** | **96.98** | **94.39** | **95.10** | **94.31** |

### 3.4 Per-category results (full RAML model)

| Category | AUROC | AP | F1-max | Precision | Recall |
|----------|------:|---:|-------:|----------:|-------:|
| bottle | 92.31 | 98.19 | 96.00 | 100.00 | 92.31 |
| cable | 89.47 | 93.66 | 90.48 | 82.61 | 100.00 |
| capsule | 92.73 | 98.38 | 95.65 | 91.67 | 100.00 |
| carpet | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| grid | 96.67 | 98.81 | 95.65 | 100.00 | 91.67 |
| hazelnut | 95.54 | 97.45 | 93.33 | 87.50 | 100.00 |
| leather | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| metal_nut | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| pill | 91.95 | 98.40 | 94.74 | 96.43 | 93.10 |
| screw | 93.52 | 97.51 | 94.12 | 88.89 | 100.00 |
| tile | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| toothbrush | 88.89 | 95.83 | 90.91 | 100.00 | 83.33 |
| transistor | 77.08 | 78.24 | 71.43 | 83.33 | 62.50 |
| wood | 95.83 | 98.81 | 95.65 | 100.00 | 91.67 |
| zipper | 98.21 | 99.48 | 97.96 | 96.00 | 100.00 |

## 4. Ablation study

Two ablation variants isolate the contribution of the margin conditioning mechanism:

- **Fixed margin:** Uses a constant margin m = m_base without the sigma-adaptive or resolution-aware terms. This variant achieves 93.20% AUROC.
- **Strong margin:** Uses a stronger margin conditioning that scales more aggressively with feature dispersion. This variant achieves 93.66% AUROC.

The full model, which balances the margin terms with the center and contrastive losses, achieves the highest AUROC of 94.15%, demonstrating that the three loss components are complementary rather than redundant.

### Training curves

The training curves below show loss (red) and validation AUROC (blue) across 25 epochs for each variant.

<p align="center">
  <img src="results/plots/curve_raml.png" width="32%" alt="RAML training curve" />
  <img src="results/plots/curve_fixed_margin.png" width="32%" alt="Fixed margin training curve" />
  <img src="results/plots/curve_strong_margin.png" width="32%" alt="Strong margin training curve" />
</p>

## 5. Repository structure

```
raml/
  models/
    raml_model.py          # main detector (MultiScaleCLIPDetector)
    multi_scale.py         # frozen CLIP multi-scale feature extractor
    cross_attention.py     # cross-scale attention pyramid
    fusion.py              # text-visual fusion module
    prompts.py             # category-specific prompt generation
  losses/
    combined_loss.py       # composite loss (center + margin + contrastive)
    center_loss.py         # center loss
    margin_loss.py         # input-size-conditioned margin loss
    contrastive_loss.py    # supervised contrastive loss
  data/
    mvtec_dataset.py       # MVTec AD data loader
  utils/
    metrics.py             # evaluation metrics
notebooks/
  run_winclip.ipynb        # zero-shot baseline evaluation
  run_raml.ipynb           # full RAML training and evaluation
  run_ablation_fixed.ipynb # fixed margin ablation
  run_ablation_strong.ipynb # strong margin ablation
configs/
  default.yaml             # default hyperparameters
results/
  metrics/                 # CSV result files per method
  plots/                   # training curve plots
```

## 6. Reproduction

### Requirements

```
torch>=2.0
torchvision
clip (openai)
scikit-learn
pandas
matplotlib
```

### Training

Open the corresponding notebook in Google Colab or a local Jupyter environment:

```bash
# full model
jupyter notebook notebooks/run_raml.ipynb

# ablation variants
jupyter notebook notebooks/run_ablation_fixed.ipynb
jupyter notebook notebooks/run_ablation_strong.ipynb

# zero-shot baseline (no training required)
jupyter notebook notebooks/run_winclip.ipynb
```

All outputs (model checkpoints, CSV metrics, training curve plots) are saved to the configured `SAVE_DIR`.

## 7. Conclusion

This work demonstrates that a lightweight, resolution-aware detection head trained on top of frozen CLIP features can substantially outperform zero-shot methods on the MVTec AD benchmark. The proposed multi-scale cross-attention pyramid captures both global context and local detail, while the composite loss function with input-size-conditioned margins produces well-separated feature clusters. The full RAML model achieves 94.15% image-level AUROC, representing a +13.29 point improvement over the WinCLIP zero-shot baseline with minimal computational overhead.

## References

- Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD -- A comprehensive real-world dataset for unsupervised anomaly detection. CVPR.
- Jeong, J., Zou, Y., Kim, T., Zhang, D., Ravichandran, A., & Dabeer, O. (2023). WinCLIP: Zero-/few-shot anomaly classification and segmentation. CVPR.
- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised contrastive learning. NeurIPS.
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. ICML.
