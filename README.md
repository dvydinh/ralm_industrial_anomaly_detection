# ralm industrial anomaly detection

paper under review. pending experiment results.

## Experiment Results (MVTec AD)

| Method | Backbone | Mean AUROC |
|--------|----------|------------|
| WinCLIP (Zero-Shot Baseline) | ViT-B/16 | 80.86% |
| RAML (Ours) | ViT-B/16 | *Training...* |
| RAML (Fixed Margin Ablation) | ViT-B/16 | *Training...* |
| RAML (Strong Margin Ablation)| ViT-B/16 | *Training...* |

<details>
<summary>Click to view Per-Category Details (WinCLIP Baseline)</summary>

- **bottle**: 86.54%
- **cable**: 79.39%
- **capsule**: 65.45%
- **carpet**: 93.52%
- **grid**: 100.00%
- **hazelnut**: 53.57%
- **leather**: 100.00%
- **metal_nut**: 95.79%
- **pill**: 62.64%
- **screw**: 77.31%
- **tile**: 88.24%
- **toothbrush**: 50.00%
- **transistor**: 87.50%
- **wood**: 89.58%
- **zipper**: 83.33%

</details>
