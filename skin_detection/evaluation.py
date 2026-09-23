"""Metrics: overall, per-class, calibration, and Fitzpatrick-stratified (skin-tone) performance.

Everything here is pure numpy/pandas/sklearn so it can be unit-tested without a model.
"""

import warnings
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from .data import tone_group

# Below this many test images a subgroup's metrics are flagged as unreliable. This is a reporting
# convention for this POC, not a statistical guarantee; see the bootstrap intervals as well.
MIN_RELIABLE_N = 100
MIN_RELIABLE_PER_CLASS = 5
MIN_SHARED_CLASS_N = 3


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([t in row for t, row in zip(y_true, topk)]))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc, avg_conf = correct[mask].mean(), conf[mask].mean()
            ece += mask.mean() * abs(acc - avg_conf)
            bins.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": int(mask.sum()),
                         "accuracy": float(acc), "mean_confidence": float(avg_conf)})
    return {"ece": float(ece), "mean_confidence": float(conf.mean()) if len(conf) else float("nan"),
            "bins": bins}


def bootstrap_ci(y_true, y_pred, metric, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over test images."""
    if len(y_true) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [metric(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return [float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))]


def _macro(y_true, y_pred):
    """Macro precision/recall/F1 over classes PRESENT in y_true (absent classes are undefined)."""
    present = np.unique(y_true)
    with warnings.catch_warnings():  # "y_pred contains classes not in y_true" is expected here
        warnings.simplefilter("ignore", UserWarning)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro",
                                                     zero_division=0)
    return float(p), float(r), float(f)


def _balanced_accuracy(y_true, y_pred):
    with warnings.catch_warnings():  # classes predicted but absent from the subgroup are ignored
        warnings.simplefilter("ignore", UserWarning)
        return float(balanced_accuracy_score(y_true, y_pred))


def subgroup_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, n_classes: int) -> Dict:
    n = int(len(y_true))
    out = {"n": n}
    if n == 0:
        return {**out, "reliable": False, "note": "no test images in this group"}
    present = np.unique(y_true)
    p, r, f = _macro(y_true, y_pred)
    per_class_counts = np.bincount(y_true, minlength=n_classes)[present]
    out.update({
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_95ci": bootstrap_ci(y_true, y_pred, accuracy_score),
        # balanced accuracy = mean recall over classes present in this group
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred) if len(present) > 1 else float("nan"),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
        "top3_accuracy": top_k_accuracy(y_true, probs, 3),
        "classes_present": int(len(present)),
        "classes_with_fewer_than_%d_images" % MIN_RELIABLE_PER_CLASS: int((per_class_counts < MIN_RELIABLE_PER_CLASS).sum()),
    })
    reasons = []
    if n < MIN_RELIABLE_N:
        reasons.append(f"only {n} test images (< {MIN_RELIABLE_N})")
    if len(present) < n_classes:
        reasons.append(f"only {len(present)}/{n_classes} classes present; macro metrics cover present classes only")
    if (per_class_counts < MIN_RELIABLE_PER_CLASS).any():
        reasons.append("some classes have < %d images; per-class recall is very noisy" % MIN_RELIABLE_PER_CLASS)
    out["reliable"] = n >= MIN_RELIABLE_N
    out["caveats"] = reasons
    return out


def evaluate(y_true: Sequence[int], probs: np.ndarray, class_names: Sequence[str],
             fitzpatrick: Sequence[int]) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    fitz = np.asarray(fitzpatrick, dtype=int)
    n_classes = len(class_names)
    if probs.shape != (len(y_true), n_classes):
        raise ValueError(f"probs shape {probs.shape} != ({len(y_true)}, {n_classes})")
    y_pred = probs.argmax(axis=1)
    labels = list(range(n_classes))

    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f,
                              "n_test": support})

    overall = subgroup_metrics(y_true, y_pred, probs, n_classes)
    overall["calibration"] = expected_calibration_error(y_true, probs)

    by_type = {}
    for t in [1, 2, 3, 4, 5, 6, -1]:
        name = "unknown" if t == -1 else f"fitzpatrick_{['I', 'II', 'III', 'IV', 'V', 'VI'][t - 1]}"
        m = fitz == t
        by_type[name] = subgroup_metrics(y_true[m], y_pred[m], probs[m], n_classes)

    groups = np.array([tone_group(t) for t in fitz])
    by_group = {g: subgroup_metrics(y_true[groups == g], y_pred[groups == g], probs[groups == g], n_classes)
                for g in ["lighter_I-III", "darker_IV-VI", "unknown"]}
    light, dark = by_group["lighter_I-III"], by_group["darker_IV-VI"]
    gaps = {}
    for key in ["accuracy", "balanced_accuracy", "macro_recall", "macro_f1", "top3_accuracy"]:
        if key in light and key in dark:
            gaps[f"{key}_gap"] = float(light[key] - dark[key])
    # Gap uncertainty: bootstrap the accuracy difference by resampling within each group.
    lm, dm = groups == "lighter_I-III", groups == "darker_IV-VI"
    if lm.sum() > 1 and dm.sum() > 1:
        rng = np.random.default_rng(0)
        lc, dc = (y_pred[lm] == y_true[lm]), (y_pred[dm] == y_true[dm])
        diffs = [rng.choice(lc, len(lc)).mean() - rng.choice(dc, len(dc)).mean() for _ in range(2000)]
        gaps["accuracy_gap_95ci"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
        gaps["note"] = ("Gap = lighter minus darker (positive = worse on darker skin). If the 95% CI "
                        "includes 0 the data cannot distinguish the groups; that is NOT evidence of fairness.")

    # Per-class recall by tone group: where does the model differ across skin tones?
    rows = []
    for ci, cname in enumerate(class_names):
        row = {"class": cname}
        for g in ["lighter_I-III", "darker_IV-VI"]:
            m = (groups == g) & (y_true == ci)
            row[f"n_{g}"] = int(m.sum())
            row[f"recall_{g}"] = float((y_pred[m] == ci).mean()) if m.any() else float("nan")
        rows.append(row)

    # Case-mix control: the two groups contain different conditions (and some conditions are
    # harder), so a raw accuracy gap mixes skin tone with condition mix. Compare mean per-class
    # recall over ONLY the classes with >= MIN_SHARED test images in BOTH groups.
    shared = [r for r in rows if min(r["n_lighter_I-III"], r["n_darker_IV-VI"]) >= MIN_SHARED_CLASS_N]
    if shared:
        lr = float(np.mean([r["recall_lighter_I-III"] for r in shared]))
        dr = float(np.mean([r["recall_darker_IV-VI"] for r in shared]))
        gaps["shared_class_macro_recall"] = {
            "n_shared_classes": len(shared), "min_images_per_group": MIN_SHARED_CLASS_N,
            "lighter": lr, "darker": dr, "gap": lr - dr,
            "classes": [r["class"] for r in shared],
        }

    return {
        "overall": overall,
        "per_class": per_class.to_dict(orient="records"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "by_fitzpatrick_type": by_type,
        "by_tone_group": by_group,
        "tone_gaps": gaps,
        "per_class_recall_by_tone_group": rows,
    }
