import numpy as np
import pandas as pd
import pytest

from skin_detection import config
from skin_detection import data as D
from skin_detection.evaluation import evaluate, top_k_accuracy


def _toy(n=300, n_labels=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)],
        "label": [f"cond{i % n_labels}" for i in range(n)],
        "fitzpatrick": rng.integers(1, 7, n),
        "source": "toy", "url": [f"http://x/{i}" for i in range(n)],
        "group_id": [f"img{i}" for i in range(n)],
    })


def test_split_assigns_each_image_to_exactly_one_split():
    out = D.split_by_group(_toy())
    assert set(out["split"]) == {"train", "val", "test"}
    assert out.groupby("image_id")["split"].nunique().max() == 1
    assert out["split"].value_counts()["test"] == pytest.approx(45, abs=6)


def test_split_keeps_groups_together():
    df = _toy()
    df["group_id"] = [f"g{i // 3}" for i in range(len(df))]  # e.g. near-duplicates / same patient
    df["label"] = [f"cond{(i // 3) % 6}" for i in range(len(df))]
    out = D.split_by_group(df)
    assert out.groupby("group_id")["split"].nunique().max() == 1


def test_check_no_leakage_detects_leak():
    out = D.split_by_group(_toy())
    leaked = pd.concat([out, out.iloc[[0]].assign(split="test" if out.iloc[0].split != "test" else "train")])
    with pytest.raises(AssertionError):
        D.check_no_leakage(leaked)


def test_committed_splits_have_no_md5_leakage():
    if not config.SPLITS_CSV.exists():
        pytest.skip("data/splits.csv not generated")
    splits = D.load_splits()  # raises on leakage
    assert splits["image_id"].is_unique  # one row per original image: no augmented copies in the file
    assert splits.groupby("image_id")["split"].nunique().max() == 1


def test_fitzpatrick_adapter_and_filter():
    df = D.filter_labels(D.load_fitzpatrick17k(), config.DEFAULT_LABELS)
    assert set(df["label"]) == set(config.DEFAULT_LABELS)
    assert not df["url"].str.contains("dermaamin.com").any()
    assert set(df["fitzpatrick"]).issubset({-1, 1, 2, 3, 4, 5, 6})
    assert set(D.COMMON_COLUMNS) <= set(df.columns)


def test_drop_exact_duplicates():
    df = _toy(10)
    df = pd.concat([df, df.iloc[[0]], df.iloc[[1]].assign(image_id="other")])
    assert len(D.drop_exact_duplicates(df)) == 10


def test_near_duplicate_grouping():
    hashes = np.array([[0] * 64, [0] * 63 + [1], [1] * 64], dtype=bool)
    groups = D.group_near_duplicates(["a", "b", "c"], hashes, max_distance=4)
    assert groups["a"] == groups["b"] != groups["c"]


def test_sample_weights_balance_classes_and_tones():
    df = pd.DataFrame({"label": ["a"] * 90 + ["b"] * 10, "fitzpatrick": [2] * 80 + [5] * 20})
    w = D.compute_sample_weights(df, ["a", "b"])
    assert w.mean() == pytest.approx(1.0, rel=1e-5)
    assert w[df.label == "b"].sum() == pytest.approx(w[df.label == "a"].sum(), rel=1e-4)
    assert w[df.fitzpatrick == 5].mean() > w[(df.fitzpatrick == 2) & (df.label == "a")].mean()


def test_evaluate_reports_tone_groups_and_gaps():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    fitz = rng.integers(1, 7, 200)
    probs = np.full((200, 3), 0.1)
    probs[np.arange(200), y] = 0.8  # perfect on everyone...
    dark = fitz >= 4
    probs[dark] = np.roll(probs[dark], 1, axis=1)  # ...except darker skin: always wrong
    m = evaluate(y, probs, ["a", "b", "c"], fitz)
    assert m["by_tone_group"]["lighter_I-III"]["accuracy"] == 1.0
    assert m["by_tone_group"]["darker_IV-VI"]["accuracy"] == 0.0
    assert m["tone_gaps"]["accuracy_gap"] == 1.0
    assert m["by_fitzpatrick_type"]["fitzpatrick_IV"]["n"] == int((fitz == 4).sum())
    assert not m["by_fitzpatrick_type"]["fitzpatrick_VI"]["reliable"]  # small group is flagged
    assert len(m["per_class"]) == 3
    shared = m["tone_gaps"]["shared_class_macro_recall"]
    assert shared["n_shared_classes"] == 3 and shared["gap"] == 1.0


def test_top_k_accuracy():
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    assert top_k_accuracy(np.array([2, 0]), probs, 1) == 0.0
    assert top_k_accuracy(np.array([2, 0]), probs, 3) == 1.0
