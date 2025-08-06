"""Composite center, margin, and supervised contrastive objective.

Combines three objectives:
    L_total = alpha * L_center + beta * L_margin + gamma * L_contrastive

Each component is an established objective applied to synthetic anomaly
supervision. Their combination is an experimental design and is not presented
as novel until controlled ablations support that claim.
"""

import torch
import torch.nn as nn

from raml.losses.center_loss import CenterLoss
from raml.losses.margin_loss import InputSizeConditionedMarginLoss
from raml.losses.contrastive_loss import SupervisedContrastiveLoss
from raml.utils.difficulty_tracker import DifficultyTracker


class CompositeAnomalyLoss(nn.Module):
    """Combined loss with optional per-category difficulty re-weighting.

    Args:
        feature_dim: Feature dimensionality.
        margin_base: Base margin for hinge loss.
        lambda_sigma: Sigma-adaptive weight.
        lambda_resolution: Resolution penalty weight.
        original_resolution: Original image resolution.
        model_resolution: CLIP input resolution.
        temperature: Contrastive loss temperature.
        alpha: Center loss coefficient.
        beta: Margin loss coefficient.
        gamma: Contrastive loss coefficient.
        difficulty_cfg: Dict with DifficultyTracker kwargs, or None to disable.
    """

    def __init__(
        self,
        feature_dim=256,
        margin_base=0.5,
        lambda_sigma=0.3,
        lambda_resolution=0.3,
        original_resolution=900,
        model_resolution=224,
        temperature=0.07,
        alpha=1.0,
        beta=1.0,
        gamma=0.5,
        difficulty_cfg=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.center_loss = CenterLoss(feature_dim)
        self.margin_loss = InputSizeConditionedMarginLoss(
            feature_dim=feature_dim,
            margin_base=margin_base,
            lambda_sigma=lambda_sigma,
            lambda_resolution=lambda_resolution,
            original_resolution=original_resolution,
            model_resolution=model_resolution,
        )
        self.contrastive_loss = SupervisedContrastiveLoss(temperature)

        if difficulty_cfg is not None:
            self.tracker = DifficultyTracker(**difficulty_cfg)
        else:
            self.tracker = None

    @staticmethod
    def _weighted_mean(values, weights, mask):
        effective_weights = weights * mask.to(weights.dtype)
        denominator = effective_weights.sum()
        if denominator <= 0:
            return values.sum() * 0.0
        return (values * effective_weights).sum() / denominator

    def forward(
        self,
        features,
        labels,
        categories=None,
        resolution_retention=None,
    ):
        """Compute combined loss.

        Args:
            features: (B, D) feature tensor.
            labels: (B,) binary labels (0=normal, 1=anomaly).
            categories: Optional list of category names (length B).
            resolution_retention: Optional (B,) retained-resolution ratios.

        Returns:
            total_loss: scalar.
            info: dict with diagnostic values.
        """
        normal_mask = labels == 0
        anomaly_mask = labels == 1

        l_center, r_center = self.center_loss(features, normal_mask)
        l_margin, r_margin, margin_val = self.margin_loss(
            features,
            anomaly_mask,
            normal_mask,
            self.center_loss.center,
            resolution_retention=resolution_retention,
        )
        l_contrastive, r_contrastive = self.contrastive_loss(features, labels)

        per_sample = (
            self.alpha * r_center
            + self.beta * r_margin
            + self.gamma * r_contrastive
        )

        if categories is not None and self.tracker is not None:
            weights = torch.tensor(
                [self.tracker.get_weight(c) for c in categories],
                device=features.device,
                dtype=features.dtype,
            )
            for cat, val in zip(categories, per_sample.detach().cpu().tolist()):
                self.tracker.update(cat, val)

            same_label_count = (
                labels.unsqueeze(0).eq(labels.unsqueeze(1)).sum(dim=1) - 1
            )
            contrastive_mask = same_label_count > 0
            l_center = self._weighted_mean(r_center, weights, normal_mask)
            l_margin = self._weighted_mean(r_margin, weights, anomaly_mask)
            l_contrastive = self._weighted_mean(
                r_contrastive, weights, contrastive_mask
            )

        total_loss = (
            self.alpha * l_center
            + self.beta * l_margin
            + self.gamma * l_contrastive
        )

        info = {
            "center_loss": float(l_center.detach()),
            "margin_loss": float(l_margin.detach()),
            "contrastive_loss": float(l_contrastive.detach()),
            "adaptive_margin": margin_val,
            "running_sigma": self.margin_loss.running_sigma.item(),
            "active_hinge_fraction": (
                float((r_margin[anomaly_mask] > 0).float().mean().detach())
                if anomaly_mask.any()
                else 0.0
            ),
        }
        return total_loss, info


MACCLLoss = CompositeAnomalyLoss
