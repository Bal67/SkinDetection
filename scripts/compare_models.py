"""Build the model comparison table from the test-split reports written by scripts/evaluate.py.

    python scripts/compare_models.py        # -> reports/model_comparison.csv and .md

Numbers are read from reports/<model>/metrics.json only; nothing is typed in by hand. A model
without a test report is listed as "not evaluated".
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from skin_detection import config  # noqa: E402

MODELS = [
    ("MobileNetV2", "mobilenetv2"),
    ("EfficientNetV2-B0", "efficientnetv2b0"),
    ("PanDerm_Base Linear Probe", "panderm_base_linear_probe"),
    ("PanDerm_Base Partial FT", "panderm_base_partial_finetune"),
]


def row(name, report_dir):
    path = config.REPORTS_DIR / report_dir / "metrics.json"
    if not path.exists():
        return {"Model": name, "status": "not evaluated"}
    m = json.loads(path.read_text())
    if m["meta"]["split"] != "test":
        raise SystemExit(f"{path} is not a test-split report")
    o, light, dark, gaps = m["overall"], m["by_tone_group"]["lighter_I-III"], m["by_tone_group"]["darker_IV-VI"], m["tone_gaps"]
    return {
        "Model": name, "status": "test", "n_test": o["n"],
        "Accuracy": o["accuracy"], "Balanced Accuracy": o["balanced_accuracy"], "Macro F1": o["macro_f1"],
        "Top-3": o["top3_accuracy"],
        "Light I-III acc": light["accuracy"], "Light n": light["n"], "Light macro F1": light["macro_f1"],
        "Dark IV-VI acc": dark["accuracy"], "Dark n": dark["n"], "Dark macro F1": dark["macro_f1"],
        "Gap (acc, light-dark)": gaps["accuracy_gap"],
        "Gap 95% CI": gaps.get("accuracy_gap_95ci"),
        "Macro-F1 gap": gaps["macro_f1_gap"],
        "Case-mix controlled recall gap": gaps.get("shared_class_macro_recall", {}).get("gap"),
        "ECE": o["calibration"]["ece"],
    }


def main():
    df = pd.DataFrame([row(n, d) for n, d in MODELS])
    out = config.REPORTS_DIR / "model_comparison.csv"
    df.to_csv(out, index=False)
    fmt = df.copy()
    for c in fmt.columns:
        if fmt[c].dtype == float:
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    fmt["Gap 95% CI"] = fmt.get("Gap 95% CI", pd.Series(dtype=object)).map(
        lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]" if isinstance(v, list) else "")
    fmt = fmt.fillna("").astype(str)
    lines = ["| " + " | ".join(fmt.columns) + " |", "|" + "---|" * len(fmt.columns)]
    lines += ["| " + " | ".join(r) + " |" for r in fmt.itertuples(index=False)]
    (config.REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines) + "\n")
    print(fmt.fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
