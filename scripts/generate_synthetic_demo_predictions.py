"""
Generate a SYNTHETIC demo predictions dataset for the fairness-audit UI.

This is a fallback, not a substitute for the real thing: scripts/generate_demo_predictions.py
runs both saved baseline models on real downloaded images, but requires network egress to
atlasdermatologico.com.br, which this sandbox's allowlist proxy blocks (verified: every
non-dermaamin image URL in the dataset resolves to that one host, and it returns
403 Host not in allowlist). Until that script is run somewhere with normal internet access,
this synthetic file lets the demo/UI work end-to-end.

What is real vs. simulated here:
  - true_label, fitzpatrick_scale, image_id (md5hash): drawn from the REAL
    data/fitzpatrick17k.csv rows (real label set, real skin-tone distribution).
  - predicted_label, confidence: SIMULATED. Correctness per row is a coin flip whose
    probability depends on the row's skin-tone group, calibrated so the overall accuracy
    lands near the ~56% the README reports for the fine-tuned model, with a deliberate
    accuracy gap across skin-tone groups (light > mid > dark) to produce a demonstrative,
    non-trivial disparity report. These numbers are NOT measurements of the actual saved
    models and must never be presented as such -- every row is written to a file named
    *_synthetic.csv, and the UI must visibly label this dataset as illustrative.
"""

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "data", "fitzpatrick17k.csv")
OUT_PATH = os.path.join(REPO_ROOT, "data", "demo_predictions_synthetic.csv")

SEED = 42
SAMPLE_SIZE = 1500

# Same 26-class set determined (from the saved models' output layer + app.py cross-reference)
# in scripts/generate_demo_predictions.py -- kept identical so this file is a drop-in stand-in.
CLASS_LABELS = sorted([
    "allergic contact dermatitis", "basal cell carcinoma", "dariers disease",
    "ehlers danlos syndrome", "erythema multiforme", "folliculitis",
    "granuloma pyogenic", "granuloma annulare", "hailey hailey disease",
    "kaposi sarcoma", "keloid", "lichen planus", "lupus erythematosus",
    "melanoma", "mycosis fungoides", "myiasis", "nematode infection",
    "neutrophilic dermatoses", "photodermatoses", "pityriasis rosea",
    "psoriasis", "scabies", "scleroderma", "squamous cell carcinoma",
    "tungiasis", "vitiligo",
])

# Illustrative per-skin-tone accuracy targets (light/mid/dark, by fitzpatrick_scale
# <=2 / 3-4 / >=5), chosen to (a) average out near the README's reported ~56% overall
# fine-tuned accuracy and (b) show a clear, statistically visible disparity -- the point
# of the demo. Not derived from any real measurement.
GROUP_ACCURACY = {"light": 0.66, "mid": 0.56, "dark": 0.42}


def skin_tone_group(fitzpatrick_scale):
    if fitzpatrick_scale <= 0:
        return "unknown"
    if fitzpatrick_scale <= 2:
        return "light"
    if fitzpatrick_scale <= 4:
        return "mid"
    return "dark"


def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(CLASS_LABELS)]
    df["fitzpatrick_scale"] = df["fitzpatrick_scale"].fillna(-1).astype(int)

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED).reset_index(drop=True)

    rows = []
    for _, row in sample.iterrows():
        fitz = int(row["fitzpatrick_scale"])
        group = skin_tone_group(fitz)
        # unknown-skin-tone rows still need a plausible accuracy; use the overall mid rate.
        p_correct = GROUP_ACCURACY.get(group, GROUP_ACCURACY["mid"])
        correct = rng.random() < p_correct

        true_label = row["label"]
        if correct:
            predicted_label = true_label
            confidence = float(rng.beta(6, 2))  # skewed high
        else:
            wrong_choices = [c for c in CLASS_LABELS if c != true_label]
            predicted_label = rng.choice(wrong_choices)
            confidence = float(rng.beta(2, 3))  # skewed lower

        rows.append({
            "image_id": row["md5hash"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "fitzpatrick_scale": fitz,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    overall_acc = (out["true_label"] == out["predicted_label"]).mean()
    print(f"Wrote {len(out)} synthetic rows to {OUT_PATH}")
    print(f"Overall simulated accuracy: {overall_acc:.3f}")
    for group in ("light", "mid", "dark"):
        mask = out["fitzpatrick_scale"].apply(skin_tone_group) == group
        if mask.sum():
            acc = (out.loc[mask, "true_label"] == out.loc[mask, "predicted_label"]).mean()
            print(f"  {group}: n={mask.sum()}, simulated accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
