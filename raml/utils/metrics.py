"""Image-level anomaly classification metrics."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_arrays(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.size != scores.size:
        raise ValueError("labels and scores must be one-dimensional and equally sized")
    if labels.size == 0:
        raise ValueError("labels and scores cannot be empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    return labels, scores


def compute_auroc(labels, scores):
    """Compute AUROC, returning NaN when only one class is present."""
    labels, scores = _as_arrays(labels, scores)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _f1_at_threshold(labels, scores, threshold):
    predictions = scores >= threshold
    true_positive = np.logical_and(predictions, labels == 1).sum()
    false_positive = np.logical_and(predictions, labels == 0).sum()
    false_negative = np.logical_and(~predictions, labels == 1).sum()
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def select_f1_threshold(labels, scores):
    """Select an F1 threshold on validation data only."""
    labels, scores = _as_arrays(labels, scores)
    if np.unique(labels).size < 2:
        return float("nan")
    candidates = np.unique(scores)
    f1_values = np.asarray(
        [_f1_at_threshold(labels, scores, threshold) for threshold in candidates]
    )
    # Prefer the highest threshold when several candidates have identical F1.
    return float(candidates[np.flatnonzero(f1_values == f1_values.max())[-1]])


def compute_binary_metrics(labels, scores, threshold=None):
    """Compute image-level metrics without selecting on evaluation labels."""
    labels, scores = _as_arrays(labels, scores)
    positive_count = int((labels == 1).sum())
    metrics = {
        "sample_count": int(labels.size),
        "normal_count": int(labels.size - positive_count),
        "anomaly_count": positive_count,
    }
    if np.unique(labels).size < 2:
        metrics.update(
            {"auroc": float("nan"), "average_precision": float("nan")}
        )
    else:
        metrics.update(
            {
                "auroc": float(roc_auc_score(labels, scores)),
                "average_precision": float(
                    average_precision_score(labels, scores)
                ),
            }
        )
    if threshold is not None:
        metrics["f1_at_validation_threshold"] = (
            _f1_at_threshold(labels, scores, float(threshold))
            if np.isfinite(threshold)
            else float("nan")
        )
    return metrics


def select_per_category_thresholds(categories, labels, scores):
    """Select one threshold per category from a validation prediction set."""
    grouped = defaultdict(lambda: ([], []))
    for category, label, score in zip(categories, labels, scores):
        grouped[category][0].append(label)
        grouped[category][1].append(score)
    if not grouped:
        raise ValueError("categories cannot be empty")
    return {
        category: select_f1_threshold(category_labels, category_scores)
        for category, (category_labels, category_scores) in sorted(grouped.items())
    }


def compute_per_category_metrics(categories, labels, scores, thresholds=None):
    """Compute each metric per category and its category-macro average."""
    grouped = defaultdict(lambda: ([], []))
    for category, label, score in zip(categories, labels, scores):
        grouped[category][0].append(label)
        grouped[category][1].append(score)

    per_category = {
        category: compute_binary_metrics(
            category_labels,
            category_scores,
            threshold=None if thresholds is None else thresholds.get(category),
        )
        for category, (category_labels, category_scores) in sorted(grouped.items())
    }
    if not per_category:
        raise ValueError("categories cannot be empty")

    macro = {}
    metric_names = ["auroc", "average_precision"]
    if thresholds is not None:
        metric_names.append("f1_at_validation_threshold")
    for metric_name in metric_names:
        values = np.asarray(
            [metrics[metric_name] for metrics in per_category.values()],
            dtype=np.float64,
        )
        macro[metric_name] = (
            float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        )
    return per_category, macro


def compute_per_category_auroc(categories, labels, scores):
    """Compatibility wrapper returning per-category and macro AUROC."""
    per_category, macro = compute_per_category_metrics(categories, labels, scores)
    return (
        {category: metrics["auroc"] for category, metrics in per_category.items()},
        macro["auroc"],
    )
