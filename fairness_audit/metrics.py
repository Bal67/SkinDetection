"""Core fairness-audit computations: per-skin-tone-group classification
metrics and cross-group disparity/significance checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from fairness_audit.grouping import SCHEMES, UNKNOWN_GROUP, assign_skin_tone_group

ALPHA = 0.05


def _compute_single_group_metrics(sub: pd.DataFrame) -> dict:
    y_true = sub["true_label"].astype(str).to_numpy()
    y_pred = sub["predicted_label"].astype(str).to_numpy()
    n = len(sub)

    accuracy = float((y_true == y_pred).mean()) if n else 0.0
    labels = sorted(set(y_true) | set(y_pred))

    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in labels if t != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "n": n,
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "labels": labels,
        "confusion_matrix": confusion,
    }


def compute_group_metrics(df: pd.DataFrame, group_col: str) -> dict:
    """Per-skin-tone-group classification metrics.

    Returns a dict keyed by group name (the "unknown" group, if present,
    is excluded). Each value has: n, accuracy, precision_macro,
    recall_macro, f1_macro, labels, confusion_matrix.
    """
    result = {}
    for group_name, sub in df.groupby(group_col):
        if group_name == UNKNOWN_GROUP:
            continue
        result[group_name] = _compute_single_group_metrics(sub)
    return result


def _two_proportion_z_test(x1: float, n1: int, x2: float, n2: int):
    """Two-proportion z-test. Returns (z_statistic, p_value), or (None, None)
    if undefined (an empty group)."""
    if n1 == 0 or n2 == 0:
        return None, None

    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        # No variance under the pooled proportion (e.g. both groups at 0% or 100%
        # accuracy): the groups are identical, so there is no detectable gap.
        return 0.0, 1.0

    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p_value)


def compute_disparity(group_metrics: dict) -> dict | None:
    """Cross-group accuracy disparity and a significance test comparing the
    best- and worst-performing groups.

    Returns None if fewer than two groups are present (no disparity is
    computable).
    """
    if len(group_metrics) < 2:
        return None

    accuracies = {g: m["accuracy"] for g, m in group_metrics.items()}
    best_group = max(accuracies, key=accuracies.get)
    worst_group = min(accuracies, key=accuracies.get)

    n_best = group_metrics[best_group]["n"]
    n_worst = group_metrics[worst_group]["n"]
    x_best = accuracies[best_group] * n_best
    x_worst = accuracies[worst_group] * n_worst

    z_stat, p_value = _two_proportion_z_test(x_best, n_best, x_worst, n_worst)
    significant = bool(p_value is not None and p_value < ALPHA)

    return {
        "best_group": best_group,
        "worst_group": worst_group,
        "best_accuracy": accuracies[best_group],
        "worst_accuracy": accuracies[worst_group],
        "accuracy_gap": accuracies[best_group] - accuracies[worst_group],
        "z_statistic": z_stat,
        "p_value": p_value,
        "significant": significant,
        "alpha": ALPHA,
    }


def generate_report(df: pd.DataFrame) -> dict:
    """Build the full, JSON-serializable fairness audit report: overall
    metrics, per-group metrics and disparity results for every supported
    grouping scheme, plus bookkeeping on excluded/unknown rows.
    """
    total_rows = len(df)
    excluded_unknown_count = int((df["fitzpatrick_scale"] == -1).sum())

    warnings = []
    if excluded_unknown_count > 0:
        warnings.append(
            f"{excluded_unknown_count} row(s) have unknown fitzpatrick_scale "
            "(-1 or blank) and were excluded from per-group fairness stats."
        )

    report = {
        "total_rows": total_rows,
        "excluded_unknown_count": excluded_unknown_count,
        "warnings": warnings,
        "overall_metrics": _compute_single_group_metrics(df),
        "schemes": {},
    }

    for scheme in SCHEMES:
        df_scheme = df.copy()
        df_scheme["skin_tone_group"] = df_scheme["fitzpatrick_scale"].apply(
            lambda v: assign_skin_tone_group(v, scheme=scheme)
        )
        group_metrics = compute_group_metrics(df_scheme, "skin_tone_group")
        report["schemes"][scheme] = {
            "group_metrics": group_metrics,
            "disparity": compute_disparity(group_metrics),
        }

    return report
