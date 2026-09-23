"""Train the image classifier with two-stage fine-tuning.

    python scripts/train.py                            # EfficientNetV2-B0 (default), 224px, pad resize
    python scripts/train.py --backbone mobilenetv2     # original-project baseline architecture
    python scripts/train.py --backbone convnext_tiny   # newest supported backbone (GPU advised)
    python scripts/train.py --finetune-epochs 0        # head-only (frozen backbone) ablation

PanDerm_Base (PyTorch, separate environment: requirements-panderm.txt; see docs/PANDERM.md):
    python scripts/train.py --backbone panderm_base --mode linear_probe
    python scripts/train.py --backbone panderm_base --mode partial_finetune

Uses ONLY the train and val rows of data/splits.csv. The test split is never loaded here.
Outputs:
    models/skin_<backbone>.keras               best checkpoint by validation balanced accuracy
    models/skin_<backbone>_class_names.json    class order + preprocessing settings (read by app/evaluation)
    reports/<backbone>/training_summary.json   config, data counts, history
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skin_detection import config  # noqa: E402
from skin_detection import data as D  # noqa: E402
from skin_detection.model import BACKBONES, build_model, load_trained_model, save_metadata  # noqa: E402
from skin_detection.training import (encode_labels, load_images, make_dataset,  # noqa: E402
                                     train_two_stage)

log = logging.getLogger("train")


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backbone", default=config.BACKBONE)
    if pre.parse_known_args(argv)[0].backbone == "panderm_base":
        # PyTorch model: hand off before TensorFlow is imported.
        from skin_detection import panderm_training
        return panderm_training.main(argv)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=config.SPLITS_CSV)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default=config.BACKBONE)
    ap.add_argument("--model-out", type=Path, default=None, help="default models/skin_<backbone>.keras")
    ap.add_argument("--meta-out", type=Path, default=None, help="default models/skin_<backbone>_class_names.json")
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    ap.add_argument("--resize-mode", choices=["pad", "stretch"], default="pad")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=15)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-epochs", type=int, default=20)
    ap.add_argument("--finetune-lr", type=float, default=1e-5)
    ap.add_argument("--fine-tune-from", default=None,
                    help="first backbone layer to unfreeze in stage B (default: per-backbone, see model.BACKBONES)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-tone-balance", action="store_true",
                    help="use class weights only (no lighter/darker skin-tone reweighting)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--deterministic", action="store_true",
                    help="enable TF op determinism (slower; full determinism is still not guaranteed on GPU)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = BACKBONES[args.backbone]
    default_model, default_meta = config.model_paths(args.backbone)
    args.model_out = args.model_out or default_model
    args.meta_out = args.meta_out or default_meta
    args.fine_tune_from = args.fine_tune_from or spec["fine_tune_from"]
    normalization = spec["normalization"]
    reports_dir = args.reports_dir / args.backbone

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    splits = D.load_splits(args.splits)
    train_df = splits[splits.split == "train"].reset_index(drop=True)
    val_df = splits[splits.split == "val"].reset_index(drop=True)
    # NOTE: splits[splits.split == "test"] is intentionally never used in this script.

    x_train, train_df = load_images(train_df, args.images_dir, args.image_size, args.resize_mode)
    x_val, val_df = load_images(val_df, args.images_dir, args.image_size, args.resize_mode)
    class_names = sorted(train_df["label"].unique())
    missing_in_val = set(class_names) - set(val_df["label"])
    if missing_in_val:
        log.warning("classes absent from validation: %s", sorted(missing_in_val))
    y_train = encode_labels(train_df["label"], class_names)
    y_val = encode_labels(val_df["label"], class_names)
    weights = D.compute_sample_weights(train_df, class_names, tone_balance=not args.no_tone_balance)
    log.info("train=%d val=%d classes=%d  sample-weight range %.2f..%.2f",
             len(y_train), len(y_val), len(class_names), weights.min(), weights.max())

    train_ds = make_dataset(x_train, y_train, weights, args.batch_size, training=True, seed=args.seed,
                            normalization=normalization)
    val_ds = make_dataset(x_val, y_val, batch_size=args.batch_size, normalization=normalization)

    model = build_model(len(class_names), args.image_size, backbone_name=args.backbone)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    # Write the class mapping BEFORE the first checkpoint lands, so the checkpoint on disk is never
    # without its matching class_names.json (it is rewritten with run details at the end).
    save_metadata(args.meta_out, class_names, args.image_size, args.resize_mode, normalization,
                  backbone=args.backbone, model_file=args.model_out.name, status="training in progress")
    history = train_two_stage(model, train_ds, val_ds, y_val, args.model_out, args.fine_tune_from,
                              head_epochs=args.head_epochs, head_lr=args.head_lr,
                              finetune_epochs=args.finetune_epochs, finetune_lr=args.finetune_lr,
                              patience=args.patience)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_metadata(args.meta_out, class_names, args.image_size, args.resize_mode, normalization,
                  backbone=args.backbone, model_file=args.model_out.name, trained_at=trained_at,
                  splits_file=str(args.splits.name), seed=args.seed,
                  selection_metric="val_balanced_accuracy")
    # Round-trip check: the saved checkpoint must load and agree with the saved class names.
    load_trained_model(args.model_out, args.meta_out)

    tone = lambda d: d["fitzpatrick"].map(D.tone_group).value_counts().to_dict()  # noqa: E731
    summary = {
        "trained_at": trained_at,
        "args": {k: str(v) for k, v in vars(args).items()},
        "tensorflow": tf.__version__, "keras": keras.__version__,
        "n_train": len(y_train), "n_val": len(y_val),
        "train_tone_groups": tone(train_df), "val_tone_groups": tone(val_df),
        "history": history,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("saved %s (best val balanced accuracy %.4f)", args.model_out, history["best_val_balanced_accuracy"])


if __name__ == "__main__":
    main()
