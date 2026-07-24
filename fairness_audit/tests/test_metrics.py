import io

from fairness_audit.metrics import compute_disparity, compute_group_metrics, generate_report
from fairness_audit.schema import load_predictions

SIMPLE_CSV = """image_id,true_label,predicted_label,fitzpatrick_scale
img_001,acne,acne,1
img_002,acne,eczema,2
img_003,eczema,eczema,3
img_004,eczema,eczema,4
img_005,melanoma,melanoma,5
img_006,melanoma,acne,6
img_007,acne,acne,-1
img_008,eczema,acne,
"""


def _make_two_group_csv(rows_light, correct_light, rows_dark, correct_dark):
    """Build a CSV with a "light" group (fitzpatrick=1) and a "dark" group
    (fitzpatrick=6), each with a controlled number of correct predictions,
    using the binary scheme's boundary (<=3 light, >3 dark)."""
    lines = ["image_id,true_label,predicted_label,fitzpatrick_scale"]
    idx = 0
    for i in range(rows_light):
        pred = "acne" if i < correct_light else "eczema"
        lines.append(f"img_{idx},acne,{pred},1")
        idx += 1
    for i in range(rows_dark):
        pred = "acne" if i < correct_dark else "eczema"
        lines.append(f"img_{idx},acne,{pred},6")
        idx += 1
    return "\n".join(lines) + "\n"


def test_generate_report_structure_and_unknown_exclusion():
    df = load_predictions(io.StringIO(SIMPLE_CSV))
    report = generate_report(df)

    assert report["total_rows"] == 8
    assert report["excluded_unknown_count"] == 2
    assert len(report["warnings"]) == 1
    assert set(report["schemes"].keys()) == {"light_mid_dark", "binary"}

    # unknown rows must not appear in any per-group metrics
    for scheme_result in report["schemes"].values():
        assert "unknown" not in scheme_result["group_metrics"]
        total_grouped = sum(m["n"] for m in scheme_result["group_metrics"].values())
        assert total_grouped == 6  # 8 rows minus 2 unknown

    overall = report["overall_metrics"]
    assert overall["n"] == 8
    assert 0.0 <= overall["accuracy"] <= 1.0


def test_compute_group_metrics_excludes_unknown_group():
    df = load_predictions(io.StringIO(SIMPLE_CSV))
    df["skin_tone_group"] = df["fitzpatrick_scale"].apply(
        lambda v: "unknown" if v == -1 else ("light" if v <= 3 else "dark")
    )
    metrics = compute_group_metrics(df, "skin_tone_group")
    assert "unknown" not in metrics
    assert set(metrics.keys()) <= {"light", "dark"}


def test_obvious_accuracy_gap_is_significant():
    csv = _make_two_group_csv(rows_light=50, correct_light=48, rows_dark=50, correct_dark=5)
    df = load_predictions(io.StringIO(csv))
    df["skin_tone_group"] = df["fitzpatrick_scale"].apply(lambda v: "light" if v <= 3 else "dark")
    group_metrics = compute_group_metrics(df, "skin_tone_group")
    disparity = compute_disparity(group_metrics)

    assert disparity is not None
    assert disparity["worst_group"] == "dark"
    assert disparity["best_group"] == "light"
    assert disparity["accuracy_gap"] > 0.5
    assert disparity["significant"] is True
    assert disparity["p_value"] < 0.05


def test_no_gap_is_not_falsely_significant():
    csv = _make_two_group_csv(rows_light=50, correct_light=25, rows_dark=50, correct_dark=25)
    df = load_predictions(io.StringIO(csv))
    df["skin_tone_group"] = df["fitzpatrick_scale"].apply(lambda v: "light" if v <= 3 else "dark")
    group_metrics = compute_group_metrics(df, "skin_tone_group")
    disparity = compute_disparity(group_metrics)

    assert disparity is not None
    assert disparity["accuracy_gap"] == 0.0
    assert disparity["significant"] is False


def test_single_group_edge_case_returns_none():
    csv = _make_two_group_csv(rows_light=20, correct_light=15, rows_dark=0, correct_dark=0)
    df = load_predictions(io.StringIO(csv))
    df["skin_tone_group"] = df["fitzpatrick_scale"].apply(lambda v: "light" if v <= 3 else "dark")
    group_metrics = compute_group_metrics(df, "skin_tone_group")

    assert list(group_metrics.keys()) == ["light"]
    assert compute_disparity(group_metrics) is None


def test_generate_report_is_json_serializable():
    import json

    df = load_predictions(io.StringIO(SIMPLE_CSV))
    report = generate_report(df)
    # Will raise TypeError if anything (e.g. numpy scalars) isn't serializable.
    json.dumps(report)
