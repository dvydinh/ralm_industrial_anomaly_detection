import pytest


torch = pytest.importorskip("torch")

from raml.losses.contrastive_loss import SupervisedContrastiveLoss
from raml.losses.margin_loss import AdaptiveMarginLoss


def test_supervised_contrastive_loss_is_finite_and_differentiable():
    features = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss, per_sample = SupervisedContrastiveLoss(temperature=0.1)(features, labels)
    assert torch.isfinite(loss)
    assert torch.isfinite(per_sample).all()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_resolution_retention_changes_only_resolution_conditioned_margin():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    normal_mask = torch.tensor([True, True, False, False])
    anomaly_mask = ~normal_mask
    center = torch.tensor([1.0, 0.0])
    retention = torch.tensor([1.0, 1.0, 1.0, 0.25])

    conditioned = AdaptiveMarginLoss(
        feature_dim=2,
        margin_base=1.0,
        lambda_sigma=0.0,
        lambda_resolution=1.0,
    )
    _, conditioned_values, _ = conditioned(
        features, anomaly_mask, normal_mask, center, retention
    )
    assert conditioned_values[3] > conditioned_values[2]

    fixed = AdaptiveMarginLoss(
        feature_dim=2,
        margin_base=1.0,
        lambda_sigma=0.0,
        lambda_resolution=0.0,
    )
    _, fixed_values, _ = fixed(
        features, anomaly_mask, normal_mask, center, retention
    )
    assert torch.equal(fixed_values[2], fixed_values[3])
