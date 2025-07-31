"""Input-size-conditioned margin loss for synthetic anomaly supervision.

The conditioned margin is defined as:

    m = m_base + lambda_sigma * sigma + lambda_r * (1 - r / r_max)

where:
    m_base      -- base margin (standard hinge-style, cf. Weinberger & Saul,
                   "Distance Metric Learning for Large Margin Nearest Neighbor
                   Classification", JMLR 2009)
    sigma       -- running standard deviation of normal features (EMA)
    r           -- effective retained resolution for a sample
    r_max       -- original long-side resolution for that sample

The dispersion term is related to adaptive-margin metric learning, including:
    Li et al., "Boosting Few-Shot Learning With Adaptive Margin Loss",
    arXiv:2005.13826 / CVPR 2020.

The input-size term is a testable heuristic. It does not by itself establish
calibration or compensation for information loss; those interpretations require
matched ablations and controlled degradation experiments.

The loss itself is a hinge loss on the distance of anomaly features from the
center:
    L_margin = mean( max(0, m - ||x_anomaly - c||_2) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputSizeConditionedMarginLoss(nn.Module):
    """Hinge loss conditioned on feature spread and nominal input retention.

    Args:
        feature_dim: Feature dimensionality (must match CenterLoss).
        margin_base: Base margin value.
        lambda_sigma: Weight for the sigma-adaptive term.
        lambda_resolution: Weight for the resolution penalty term.
        original_resolution: Typical original image resolution.
        model_resolution: CLIP input resolution.
        sigma_momentum: EMA decay for tracking feature std.
    """

    def __init__(
        self,
        feature_dim,
        margin_base=0.5,
        lambda_sigma=0.3,
        lambda_resolution=0.3,
        original_resolution=900,
        model_resolution=224,
        sigma_momentum=0.1,
    ):
        super().__init__()
        self.margin_base = margin_base
        self.lambda_sigma = lambda_sigma
        self.lambda_resolution = lambda_resolution
        self.default_resolution_retention = min(
            1.0, model_resolution / original_resolution
        )
        self.sigma_momentum = sigma_momentum
        self.register_buffer("running_sigma", torch.tensor(0.0))

    def forward(
        self,
        features,
        anomaly_mask,
        normal_mask,
        center,
        resolution_retention=None,
    ):
        """Compute the conditioned margin loss for anomaly samples.

        Args:
            features: (B, D) feature tensor.
            anomaly_mask: (B,) boolean mask where True = anomaly.
            normal_mask: (B,) boolean mask where True = normal.
            center: (D,) current normal center from CenterLoss.
            resolution_retention: Optional (B,) tensor in [0, 1]. Each value
                is the ratio between the effective multi-scale resolution and
                the original image long side.

        Returns:
            loss: scalar margin loss.
            per_sample: (B,) per-sample loss.
            margin_value: Mean conditioned margin for diagnostics.
        """
        with torch.no_grad():
            if normal_mask.sum() > 0:
                normal_features = F.normalize(features[normal_mask], dim=1)
                normal_center = F.normalize(normal_features.mean(dim=0), dim=0)
                batch_sigma = torch.sqrt(
                    (normal_features - normal_center).square().sum(dim=1).mean()
                    + 1e-12
                )
                if self.running_sigma < 1e-12:
                    self.running_sigma.copy_(batch_sigma)
                else:
                    self.running_sigma.copy_(
                        (1 - self.sigma_momentum) * self.running_sigma
                        + self.sigma_momentum * batch_sigma
                    )

        if resolution_retention is None:
            resolution_retention = features.new_full(
                (features.size(0),), self.default_resolution_retention
            )
        else:
            resolution_retention = torch.as_tensor(
                resolution_retention, device=features.device, dtype=features.dtype
            )
            if resolution_retention.ndim == 0:
                resolution_retention = resolution_retention.expand(features.size(0))
            if resolution_retention.shape != (features.size(0),):
                raise ValueError("resolution_retention must have shape (batch,)")

        resolution_retention = resolution_retention.clamp(0.0, 1.0)
        margin = (
            self.margin_base
            + self.lambda_sigma * self.running_sigma.detach()
            + self.lambda_resolution * (1.0 - resolution_retention)
        )

        dist = torch.linalg.norm(features - center.detach(), ord=2, dim=1)
        per_sample = F.relu(margin - dist) * anomaly_mask.float()

        if anomaly_mask.sum() > 0:
            loss = per_sample.sum() / anomaly_mask.sum()
        else:
            loss = torch.tensor(0.0, device=features.device)

        return loss, per_sample, margin.mean().item()


AdaptiveMarginLoss = InputSizeConditionedMarginLoss
