import math

import pytest


pytest.importorskip("sklearn")

from raml.utils.metrics import (
    compute_auroc,
    compute_binary_metrics,
    select_f1_threshold,
)


def test_one_class_auroc_is_explicitly_undefined():
    assert math.isnan(compute_auroc([0, 0], [0.1, 0.2]))


def test_binary_metrics_are_finite_for_two_classes():
    threshold = select_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=threshold
    )
    assert all(math.isfinite(value) for value in metrics.values())


def test_f1_uses_the_supplied_validation_threshold():
    metrics = compute_binary_metrics(
        [0, 1], [0.4, 0.6], threshold=0.7
    )
    assert metrics["f1_at_validation_threshold"] == 0.0
