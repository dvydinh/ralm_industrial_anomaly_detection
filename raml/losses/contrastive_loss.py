"""Supervised contrastive loss for anomaly detection.

Reference:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

    L_sup = sum_i  -1/|P(i)| * sum_{p in P(i)} log(
                exp(z_i . z_p / tau) / sum_{a != i} exp(z_i . z_a / tau)
            )

We use the binary label (normal=0, anomaly=1) to define positive pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss with temperature scaling.

    Unlike the original code which looped over every sample, this
    implementation is fully vectorized.

    Args:
        temperature: Scaling factor for cosine similarities.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """Compute supervised contrastive loss.

        Args:
            features: (B, D) feature tensor (will be L2-normalized).
            labels: (B,) integer labels.

        Returns:
            loss: scalar loss.
            per_sample: (B,) per-sample loss.
        """
        device = features.device
        B = features.size(0)
        per_sample = torch.zeros(B, device=device)

        if B < 2:
            return features.sum() * 0.0, per_sample

        features = F.normalize(features, p=2, dim=1)
        sim = torch.mm(features, features.t()) / self.temperature

        # Mask out self-similarity
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, float("-inf"))

        # Positive mask: same label, exclude self
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = labels_eq & ~self_mask

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        # A masked diagonal contains -inf. Set it to zero before reduction so
        # excluded pairs cannot create NaNs through the expression -inf * 0.
        log_prob = log_prob.masked_fill(self_mask, 0.0)

        # Mean of log-prob over positive pairs per sample
        pos_count = pos_mask.sum(dim=1).float()
        mean_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_count.clamp_min(1.0)

        per_sample = -mean_log_prob
        valid = pos_count > 0
        if valid.sum() > 0:
            loss = per_sample[valid].mean()
        else:
            loss = features.sum() * 0.0

        return loss, per_sample
