# ralm industrial anomaly detection

paper under review.

## Experiment Results (MVTec AD)

| Method | Backbone | Mean AUROC | Mean AP | Mean F1 |
|--------|----------|------------|---------|---------|
| WinCLIP (Zero-Shot Baseline) | ViT-B/16 | 80.86% | 91.67% | 89.89% |
| RAML (Fixed Margin Ablation) | ViT-B/16 | 93.20% | 96.69% | 93.13% |
| RAML (Strong Margin Ablation)| ViT-B/16 | 93.66% | 96.61% | 94.06% |
| **RAML (Full Model - Ours)** | ViT-B/16 | **94.15%** | **96.98%** | **94.39%** |

*Note: All models use identical data splits (seed=42) and are evaluated over the entire dataset with matching validation ratio settings.*
