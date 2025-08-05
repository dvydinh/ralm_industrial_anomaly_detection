"""Center loss for pulling normal features toward a learnable center.

Reference:
    Wen et al., "A Discriminative Feature Learning Approach for Deep Face
    Recognition", ECCV 2016.

    L_center = (1/2m) * sum_i ||x_i - c_{y_i}||^2

In anomaly detection we maintain a single center for the normal class and
update it with an exponential moving average over mini-batch means.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Penalize the squared distance of normal features to a running center.

    The center is updated via EMA:
        c <- (1 - alpha) * c + alpha * mean(x_normal)

    Args:
        feature_dim: Dimensionality of the feature vectors.
        center_momentum: EMA coefficient for center updates (alpha above).
    """

    def __init__(self, feature_dim, center_momentum=0.2):
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(feature_dim))

    def forward(self, features, normal_mask):
        """Compute center loss over normal samples.

        Args:
            features: (B, D) feature tensor.
            normal_mask: (B,) boolean mask where True = normal.

        Returns:
            loss: scalar mean squared distance for normal samples.
            per_sample: (B,) per-sample loss (zero for anomaly samples).
        """
        per_sample = torch.zeros(features.size(0), device=features.device)

        if normal_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device), per_sample

        batch_center = F.normalize(features[normal_mask].mean(dim=0), dim=0)
        with torch.no_grad():
            if self.center.square().sum() < 1e-12:
                self.center.copy_(batch_center)
            else:
                updated_center = (
                    (1 - self.center_momentum) * self.center
                    + self.center_momentum * batch_center
                )
                self.center.copy_(F.normalize(updated_center, dim=0))

        dist_sq = ((features - self.center.detach()) ** 2).sum(dim=1)
        per_sample = dist_sq * normal_mask.float()
        loss = per_sample.sum() / normal_mask.sum()

        return loss, per_sample
