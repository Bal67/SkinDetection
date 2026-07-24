"""Dermatology AI Fairness Auditor.

A Streamlit app that audits a *third-party* dermatology model's predictions
CSV for accuracy disparities across Fitzpatrick skin-tone groups. This app
does not diagnose anything itself and makes no medical claims -- it only
evaluates prediction/ground-truth pairs that the user supplies.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from fairness_audit.metrics import generate_report
from fairness_audit.schema import REQUIRED_COLUMNS, SchemaError, load_predictions

DEMO_CSV_PATH = "data/demo_predictions_synthetic.csv"
SCHEME_LABELS = {
    "light_mid_dark": "Light / Mid / Dark",
    "binary": "Light / Dark (binary)",
}

st.set_page_config(page_title="Dermatology AI Fairness Auditor", page_icon="🩺", layout="wide")


def sample_csv_bytes() -> bytes:
    sample = pd.DataFrame(
        {
            "image_id": ["img_001", "img_002", "img_003", "img_004"],
            "true_label": ["melanoma", "psoriasis", "melanoma", "vitiligo"],
            "predicted_label": ["melanoma", "psoriasis", "vitiligo", "vitiligo"],
            "confidence": [0.91, 0.77, 0.55, 0.88],
            "fitzpatrick_scale": [2, 5, 4, 6],
        }
    )
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def render_metrics_row(metrics: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("n", metrics["n"])
    cols[1].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    cols[2].metric("Precision (macro)", f"{metrics['precision_macro']:.1%}")
    cols[3].metric("Recall (macro)", f"{metrics['recall_macro']:.1%}")
    cols[4].metric("F1 (macro)", f"{metrics['f1_macro']:.1%}")


def render_disparity(disparity: dict | None) -> None:
    if disparity is None:
        st.info("Fewer than two skin-tone groups are present in this data, so no disparity comparison can be computed.")
        return

    gap_pct = disparity["accuracy_gap"] * 100
    p_value = disparity["p_value"]
    p_value_str = f"{p_value:.4f}" if p_value is not None else "n/a"
    message = (
        f"**Worst-performing group:** `{disparity['worst_group']}` "
        f"({disparity['worst_accuracy']:.1%} accuracy)  \n"
        f"**Best-performing group:** `{disparity['best_group']}` ({disparity['best_accuracy']:.1%} accuracy)  \n"
        f"**Accuracy gap:** {gap_pct:.1f} percentage points  \n"
        f"**Significance:** z = {disparity['z_statistic']:.3f}, p = {p_value_str} (alpha = {disparity['alpha']})"
    )

    flag_light_over_dark = disparity["worst_group"] == "dark" and disparity["best_group"] in ("light", "mid")
    if disparity["significant"] and flag_light_over_dark:
        st.error("Statistically significant accuracy disparity detected, disfavoring darker skin tones.\n\n" + message)
    elif disparity["significant"]:
        st.warning("Statistically significant accuracy disparity detected between groups.\n\n" + message)
    else:
        st.success("No statistically significant accuracy disparity detected between groups.\n\n" + message)


def render_report(report: dict, key_prefix: str) -> None:
    for w in report.get("warnings", []):
        st.info(w)

    st.subheader("Overall metrics (all rows, regardless of skin tone)")
    render_metrics_row(report["overall_metrics"])

    st.subheader("Fairness breakdown by skin-tone group")
    scheme = st.radio(
        "Grouping scheme",
        options=list(SCHEME_LABELS.keys()),
        format_func=lambda s: SCHEME_LABELS[s],
        horizontal=True,
        key=f"{key_prefix}_scheme",
    )

    scheme_report = report["schemes"][scheme]
    group_metrics = scheme_report["group_metrics"]

    if not group_metrics:
        st.info("No rows with a known Fitzpatrick skin-tone score are available for this grouping.")
    else:
        accuracy_by_group = pd.DataFrame(
            {"accuracy": {g: m["accuracy"] for g, m in group_metrics.items()}}
        )
        st.bar_chart(accuracy_by_group)

        table = pd.DataFrame(
            {
                g: {
                    "n": m["n"],
                    "accuracy": m["accuracy"],
                    "precision_macro": m["precision_macro"],
                    "recall_macro": m["recall_macro"],
                    "f1_macro": m["f1_macro"],
                }
                for g, m in group_metrics.items()
            }
        ).T
        st.dataframe(table)

    st.subheader("Disparity check")
    render_disparity(scheme_report["disparity"])

    st.download_button(
        "Download full report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="fairness_audit_report.json",
        mime="application/json",
        key=f"{key_prefix}_download",
    )


def run_audit_and_render(df: pd.DataFrame, key_prefix: str) -> None:
    report = generate_report(df)
    render_report(report, key_prefix)


st.title("Dermatology AI Fairness Auditor")
st.markdown(
    """
This tool audits the **fairness** of an existing dermatology AI model by checking whether its
prediction accuracy differs significantly across Fitzpatrick skin-tone groups. You supply a CSV of
a model's predictions alongside ground-truth labels; this tool computes accuracy/precision/recall/F1
overall and per skin-tone group, and flags statistically significant disparities.

**This is not a diagnostic tool.** It does not analyze images, does not produce medical predictions,
and makes no medical claims. It only evaluates prediction outputs that you provide from a model you
already have.
"""
)

with st.expander("Expected CSV format", expanded=False):
    st.markdown(
        f"""
Required columns: `{'`, `'.join(REQUIRED_COLUMNS)}`
Optional columns: `confidence` (0-1)

- `fitzpatrick_scale` should be an integer 1-6, or -1 (or blank) if unknown.
- `image_id` must be unique.
"""
    )
    st.dataframe(pd.read_csv(io.BytesIO(sample_csv_bytes())))
    st.download_button(
        "Download sample CSV template",
        data=sample_csv_bytes(),
        file_name="sample_predictions_template.csv",
        mime="text/csv",
    )

st.divider()

tab_upload, tab_demo = st.tabs(["Upload your own", "Try the demo (synthetic data)"])

with tab_upload:
    st.subheader("Upload a predictions CSV")
    uploaded = st.file_uploader("Predictions CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = load_predictions(uploaded)
        except SchemaError as e:
            st.error(str(e))
        else:
            st.success(f"Loaded {len(df)} rows.")
            run_audit_and_render(df, key_prefix="upload")

with tab_demo:
    st.caption(
        "SYNTHETIC DATA — this demo uses simulated model predictions (real Fitzpatrick-scale and "
        "label distributions from the underlying dataset, but simulated correctness). It is for "
        "illustration only and is NOT the output of a real deployed model."
    )
    if st.button("Load demo data"):
        try:
            demo_df = load_predictions(DEMO_CSV_PATH)
        except (SchemaError, FileNotFoundError) as e:
            st.error(f"Could not load demo data: {e}")
        else:
            st.info(f"Loaded {len(demo_df)} synthetic demo rows from `{DEMO_CSV_PATH}`.")
            run_audit_and_render(demo_df, key_prefix="demo")
