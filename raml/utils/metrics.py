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
    labels, scores = _as_arrays(labels, scores)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))

def _f1_pr_at_threshold(labels, scores, threshold):
    predictions = scores >= threshold
    true_positive = np.logical_and(predictions, labels == 1).sum()
    false_positive = np.logical_and(predictions, labels == 0).sum()
    false_negative = np.logical_and(~predictions, labels == 1).sum()
    
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = float(2 * true_positive / denominator) if denominator else 0.0
    
    precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else 0.0
    recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0
    
    return f1, precision, recall

def compute_binary_metrics(labels, scores):
    labels, scores = _as_arrays(labels, scores)
    positive_count = int((labels == 1).sum())
    metrics = {
        "sample_count": int(labels.size),
        "normal_count": int(labels.size - positive_count),
        "anomaly_count": positive_count,
    }
    if np.unique(labels).size < 2:
        metrics.update({
            "auroc": float("nan"), 
            "ap": float("nan"),
            "f1_max": float("nan"),
            "precision": float("nan"),
            "recall": float("nan")
        })
    else:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["ap"] = float(average_precision_score(labels, scores))
        
        candidates = np.unique(scores)
        best_f1, best_p, best_r = 0.0, 0.0, 0.0
        best_t = candidates[0]
        
        for t in candidates:
            f1, p, r = _f1_pr_at_threshold(labels, scores, t)
            if f1 > best_f1 or (f1 == best_f1 and t > best_t):
                best_f1, best_p, best_r = f1, p, r
                best_t = t
                
        metrics["f1_max"] = best_f1
        metrics["precision"] = best_p
        metrics["recall"] = best_r

    return metrics

def compute_per_category_metrics(categories, labels, scores):
    grouped = defaultdict(lambda: ([], []))
    for category, label, score in zip(categories, labels, scores):
        grouped[category][0].append(label)
        grouped[category][1].append(score)

    per_category = {
        category: compute_binary_metrics(category_labels, category_scores)
        for category, (category_labels, category_scores) in sorted(grouped.items())
    }
    if not per_category:
        raise ValueError("categories cannot be empty")

    macro = {}
    metric_names = ["auroc", "ap", "f1_max", "precision", "recall"]
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
    # Kept for legacy compatibility if anything else uses it
    per_category, macro = compute_per_category_metrics(categories, labels, scores)
    return (
        {category: metrics["auroc"] for category, metrics in per_category.items()},
        macro["auroc"],
    )
