"""Evaluate a trained model, overall and stratified by Fitzpatrick skin type.

    python scripts/evaluate.py                          # default backbone, held-out TEST split
    python scripts/evaluate.py --backbone mobilenetv2   # another trained backbone
    python scripts/evaluate.py --split val              # use during development instead of test
    python scripts/evaluate.py --legacy                 # the original 2024 model (see caveat below)
    python scripts/evaluate.py --backbone panderm_base --mode linear_probe       # PyTorch env
    python scripts/evaluate.py --backbone panderm_base --mode partial_finetune

The test split must only be used for FINAL numbers. Do not tune anything (hyperparameters,
augmentation, thresholds, epochs) based on test results; use --split val for that.

Outputs (under --out-dir, default reports/<backbone>/, reports/panderm_base_<mode>/ or reports/legacy/):
    metrics.json                    everything, machine-readable
    per_class.csv                   precision/recall/F1/n per class
    fitzpatrick_metrics.csv         per Fitzpatrick type I..VI + lighter/darker/unknown groups
    per_class_recall_by_tone.csv    per-class recall for lighter vs darker skin
    confusion_matrix.csv / .png
    fitzpatrick_accuracy.png
    predictions.csv                 one row per evaluated image (top-3 outputs)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from skin_detection import config  # noqa: E402
from skin_detection import data as D  # noqa: E402
from skin_detection.evaluation import evaluate  # noqa: E402
from skin_detection.model import BACKBONES, load_trained_model  # noqa: E402
from skin_detection.preprocessing import normalize  # noqa: E402
from skin_detection.training import encode_labels, load_images  # noqa: E402

log = logging.getLogger("evaluate")

LEGACY_CAVEAT = (
    "The legacy model was trained on a random split of AUGMENTED rows made by an older pipeline; "
    "which original images it saw is unknown, so images in this split may have been in its "
    "training data. Its numbers here are likely optimistic and are NOT a valid held-out estimate."
)


def _json_safe(obj):
    """NaN/inf -> null so metrics.json is strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def plot(metrics, class_names, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, accent = "#1f2328", "#6e7781", "#2f6fb0"

    # Accuracy per Fitzpatrick type with bootstrap 95% CI and n.
    rows = [(k.replace("fitzpatrick_", ""), v) for k, v in metrics["by_fitzpatrick_type"].items()
            if k != "unknown" and v["n"] > 0]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs = np.arange(len(rows))
    acc = [v["accuracy"] for _, v in rows]
    lo = [v["accuracy"] - v["accuracy_95ci"][0] for _, v in rows]
    hi = [v["accuracy_95ci"][1] - v["accuracy"] for _, v in rows]
    ax.errorbar(xs, acc, yerr=[lo, hi], fmt="o", color=accent, ms=8, capsize=4, lw=2)
    overall = metrics["overall"]["accuracy"]
    ax.axhline(overall, color=muted, lw=1, ls="--")
    # label sits between the first two points, where there is never an error bar
    ax.text(0.5, overall + 0.01, f"overall {overall:.2f}", va="bottom", ha="center", color=muted, fontsize=9)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_xticks(xs, [f"{name}\nn={v['n']}" for name, v in rows], color=ink)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 accuracy (95% bootstrap CI)", color=ink)
    ax.set_title(f"{metrics['meta']['split'].capitalize()} accuracy by Fitzpatrick skin type ({metrics['meta']['model']})",
                 color=ink, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#d0d7de", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "fitzpatrick_accuracy.png", dpi=150)
    plt.close(fig)

    # Row-normalized confusion matrix (single-hue sequential).
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(class_names)), [f"{c} (n={int(n)})" for c, n in zip(class_names, cm.sum(1))], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized = recall)", loc="left")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--splits", type=Path, default=config.SPLITS_CSV)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--backbone", choices=sorted(BACKBONES) + ["panderm_base"], default=config.BACKBONE)
    ap.add_argument("--mode", choices=["linear_probe", "partial_finetune"], default=None,
                    help="PanDerm only")
    ap.add_argument("--device", default="auto", help="PanDerm only: auto | cuda | mps | cpu")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--legacy", action="store_true", help="evaluate models/finetuned_mobilenetv2.h5")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    panderm = args.backbone == "panderm_base"
    if panderm and args.mode is None:
        ap.error("--backbone panderm_base requires --mode linear_probe|partial_finetune")
    if args.legacy:
        model_path, meta_path = config.LEGACY_MODEL_PATH, config.LEGACY_MODEL_META_PATH
        out_dir = args.out_dir or config.REPORTS_DIR / "legacy"
    elif panderm:
        from skin_detection import panderm as PD

        model_path, meta_path = PD.mode_paths(args.mode)
        out_dir = args.out_dir or config.REPORTS_DIR / f"panderm_base_{args.mode}"
    else:
        default_model, default_meta = config.model_paths(args.backbone)
        model_path, meta_path = args.model or default_model, args.meta or default_meta
        out_dir = args.out_dir or config.REPORTS_DIR / args.backbone
    if args.split == "val":
        out_dir = out_dir / "val"
    out_dir.mkdir(parents=True, exist_ok=True)

    if panderm:
        device = PD.pick_device(args.device)
        model, meta = PD.load_classifier(args.mode, device)
    else:
        model, meta = load_trained_model(model_path, meta_path)
    class_names = meta["class_names"]

    splits = D.load_splits(args.splits)
    df = splits[splits.split == args.split].reset_index(drop=True)
    unsupported = set(df["label"]) - set(class_names)
    if unsupported:
        raise SystemExit(f"{args.split} split contains labels the model does not support: {sorted(unsupported)}")
    x, df = load_images(df, args.images_dir, meta["image_size"], meta["resize_mode"])
    y = encode_labels(df["label"], class_names)
    if panderm:
        probs = PD.predict_probs(model, normalize(x, meta["normalization"]), device, batch_size=32)
    else:
        probs = model.predict(normalize(x, meta["normalization"]), batch_size=args.batch_size, verbose=0)

    metrics = evaluate(y, probs, class_names, df["fitzpatrick"].to_numpy())
    metrics["meta"] = {
        "model": str(model_path.name) if not panderm else f"panderm_base {args.mode}",
        "backbone": meta.get("backbone", "mobilenetv2 (legacy)"),
        "split": args.split, "n_images": int(len(y)),
        "image_size": meta["image_size"], "resize_mode": meta["resize_mode"],
        "fitzpatrick_annotation": "fitzpatrick_scale (primary Fitzpatrick17k annotation)",
        "caveat": LEGACY_CAVEAT if meta.get("legacy_format") else None,
    }
    if args.split == "val":
        metrics["meta"]["note"] = "Validation split: used for model selection, so these numbers are optimistic."

    (out_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2, allow_nan=False))
    pd.DataFrame(metrics["per_class"]).to_csv(out_dir / "per_class.csv", index=False)
    rows = [{"group": k, **{m: v.get(m) for m in ["n", "accuracy", "balanced_accuracy", "macro_precision",
                                                   "macro_recall", "macro_f1", "top3_accuracy", "reliable"]},
             "accuracy_95ci": v.get("accuracy_95ci"), "caveats": "; ".join(v.get("caveats", []))}
            for k, v in {**metrics["by_fitzpatrick_type"], **metrics["by_tone_group"]}.items()]
    pd.DataFrame(rows).to_csv(out_dir / "fitzpatrick_metrics.csv", index=False)
    pd.DataFrame(metrics["per_class_recall_by_tone_group"]).to_csv(out_dir / "per_class_recall_by_tone.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix.csv")
    order = np.argsort(-probs, axis=1)[:, :3]
    pd.DataFrame({
        "image_id": df["image_id"], "label": df["label"], "fitzpatrick": df["fitzpatrick"],
        **{f"top{i + 1}": [class_names[j] for j in order[:, i]] for i in range(3)},
        **{f"top{i + 1}_prob": probs[np.arange(len(probs)), order[:, i]].round(4) for i in range(3)},
    }).to_csv(out_dir / "predictions.csv", index=False)
    plot(metrics, class_names, out_dir)

    o, g = metrics["overall"], metrics["tone_gaps"]
    log.info("\n[%s split, %d images] accuracy %.3f  balanced acc %.3f  macro F1 %.3f  top-3 %.3f  ECE %.3f",
             args.split, o["n"], o["accuracy"], o["balanced_accuracy"], o["macro_f1"], o["top3_accuracy"],
             o["calibration"]["ece"])
    for name, v in metrics["by_tone_group"].items():
        if v["n"]:
            log.info("  %-14s n=%-4d acc %.3f  macro recall %.3f  macro F1 %.3f%s", name, v["n"], v["accuracy"],
                     v["macro_recall"], v["macro_f1"], "" if v["reliable"] else "  (SMALL SAMPLE)")
    if "accuracy_gap" in g:
        log.info("  accuracy gap (lighter - darker) %.3f, 95%% CI %s", g["accuracy_gap"], g.get("accuracy_gap_95ci"))
    if "shared_class_macro_recall" in g:
        s = g["shared_class_macro_recall"]
        log.info("  case-mix controlled (%d classes with >=%d test images in both groups): macro recall "
                 "lighter %.3f vs darker %.3f, gap %.3f", s["n_shared_classes"], s["min_images_per_group"],
                 s["lighter"], s["darker"], s["gap"])
    if metrics["meta"]["caveat"]:
        log.warning("CAVEAT: %s", metrics["meta"]["caveat"])
    log.info("reports written to %s", out_dir)


if __name__ == "__main__":
    main()
