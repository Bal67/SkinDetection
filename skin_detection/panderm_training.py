"""PanDerm_Base training (PyTorch): linear probe and partial fine-tuning.

    python scripts/train.py --backbone panderm_base --mode linear_probe
    python scripts/train.py --backbone panderm_base --mode partial_finetune

Uses the SAME data/splits.csv as the TensorFlow baselines, and only its train and val rows.
Validation drives every decision (probe cost, early stopping, checkpoint selection); the test split
is never loaded here.

Class/skin-tone balancing uses data.compute_sample_weights (class-balanced, tone-balanced, clipped):
  * linear probe: full-batch L-BFGS, so the weights enter the loss directly (weighted cross-entropy);
  * partial fine-tuning: as in the official PanDerm recipe (`--weights`), a WeightedRandomSampler
    draws training images with these weights; the loss itself is unweighted (no double counting).

Training augmentation (partial fine-tuning only; the linear probe uses the official frozen-feature
protocol with no augmentation). The official recipe uses RandomResizedCrop(scale 0.75-1), horizontal +
vertical flips, RandomRotation(45) and ColorJitter(hue=0.2). Here we keep the crop and horizontal flip
but use rotation +/-14 degrees, no vertical flip and brightness/contrast +/-10% instead of hue jitter,
the same policy as the TensorFlow models: no hue/saturation change, no recoloring of skin, no color
inversion. Mixup/CutMix (official fine-tuning) are also not used.
"""

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from . import data as D
from . import panderm as P
from .preprocessing import load_image, normalize, resize_shorter_side

log = logging.getLogger("panderm")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=[P.NAME], default=P.NAME)
    ap.add_argument("--mode", choices=P.MODES, required=True)
    ap.add_argument("--splits", type=Path, default=config.SPLITS_CSV)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-tone-balance", action="store_true")
    ap.add_argument("--feature-batch-size", type=int, default=32, help="frozen-encoder inference batch")
    # linear probe
    ap.add_argument("--probe-cost-multipliers", default="0.01,0.1,1,10,100,1000",
                    help="multiples of the official cost C = 768 * n_classes / 100; best chosen on val "
                         "(grid widened after x10 was best on the first, narrower grid)")
    ap.add_argument("--probe-max-iter", type=int, default=1000)
    # partial fine-tuning
    ap.add_argument("--trainable-blocks", type=int, default=2, help="unfreeze the last N of 12 blocks")
    ap.add_argument("--batch-size", type=int, default=8, help="physical batch (use 4 if 8 does not fit)")
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=2e-5, help="LR of the top unfrozen block")
    ap.add_argument("--layer-decay", type=float, default=0.65, help="official PanDerm value")
    ap.add_argument("--weight-decay", type=float, default=0.05, help="official PanDerm value")
    ap.add_argument("--drop-path", type=float, default=0.2, help="official PanDerm value")
    ap.add_argument("--label-smoothing", type=float, default=0.1, help="official PanDerm value")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    # smoke tests only
    ap.add_argument("--max-train-samples", type=int, default=None, help="debug: subsample train")
    ap.add_argument("--max-val-samples", type=int, default=None, help="debug: subsample val")
    ap.add_argument("--output-suffix", default="", help="debug: write *_<suffix> files instead")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- data


def load_split_frames(args):
    splits = D.load_splits(args.splits)
    train = splits[splits.split == "train"].reset_index(drop=True)
    val = splits[splits.split == "val"].reset_index(drop=True)
    # NOTE: splits[splits.split == "test"] is intentionally never used here.
    if args.max_train_samples:
        per_class = max(1, args.max_train_samples // train["label"].nunique())
        train = train.groupby("label").head(per_class).reset_index(drop=True)  # keeps every class
    if args.max_val_samples:
        val = val.sample(n=min(args.max_val_samples, len(val)), random_state=args.seed).reset_index(drop=True)
    return train, val


def eval_arrays(df, images_dir):
    """Canonical PanDerm eval preprocessing -> normalized (n, 224, 224, 3) float32 + kept rows."""
    from .training import load_images

    x, kept = load_images(df, images_dir, P.IMAGE_SIZE, P.RESIZE_MODE)
    return normalize(x, P.NORMALIZATION), kept


def extract_features(model, x_norm: np.ndarray, device, batch_size: int, amp: bool) -> np.ndarray:
    import torch

    dtype = P.autocast_dtype(device, amp)
    feats = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(x_norm), batch_size):
            xb = P.to_tensor_batch(x_norm[i:i + batch_size]).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                feats.append(model.features(xb).float().cpu().numpy())
    return np.concatenate(feats)


def cached_features(model, df, split_name, args, device):
    """Frozen CLS features for a split, cached under data/features/ (gitignored: derived from the
    NC-ND weights). The cache key includes the checkpoint hash and preprocessing."""
    cache_dir = config.DATA_DIR / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{P.NAME}_cls_{P.CHECKPOINT_SHA256[:12]}_{P.RESIZE_MODE}_{split_name}{args.output_suffix}"
    path = cache_dir / f"{key}.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if list(cached["image_id"]) == list(df["image_id"]):
            return cached["features"], df
    x, kept = eval_arrays(df, args.images_dir)
    t = time.time()
    feats = extract_features(model, x, device, args.feature_batch_size, not args.no_amp)
    log.info("extracted %s features %s in %.0fs", split_name, feats.shape, time.time() - t)
    np.savez(path, features=feats, image_id=kept["image_id"].to_numpy(dtype=str))
    return feats, kept


# --------------------------------------------------------------------------- linear probe


def run_linear_probe(args, device):
    import torch
    from sklearn.metrics import balanced_accuracy_score, f1_score

    from .training import encode_labels

    train_df, val_df = load_split_frames(args)
    model = P.make_classifier(train_df["label"].nunique(), pooling="cls", n_trainable_blocks=0)
    model.to(device)
    f_train, train_df = cached_features(model, train_df, "train", args, device)
    f_val, val_df = cached_features(model, val_df, "val", args, device)

    class_names = sorted(train_df["label"].unique())
    y_train = encode_labels(train_df["label"], class_names)
    y_val = encode_labels(val_df["label"], class_names)
    weights = D.compute_sample_weights(train_df, class_names, tone_balance=not args.no_tone_balance)

    official_cost = P.EMBED_DIM * len(class_names) / 100
    grid = []
    for mult in [float(m) for m in args.probe_cost_multipliers.split(",")]:
        C = official_cost * mult
        layer = P.fit_logistic_regression(f_train, y_train, weights, len(class_names), C,
                                          max_iter=args.probe_max_iter, seed=args.seed)
        with torch.no_grad():
            pred = layer(torch.as_tensor(f_val)).argmax(1).numpy()
        row = {"cost_multiplier": mult, "C": C,
               "val_balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
               "val_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
               "val_accuracy": float((pred == y_val).mean())}
        log.info("C=%.3g (x%g official): val balanced acc %.4f  macro F1 %.4f", C, mult,
                 row["val_balanced_accuracy"], row["val_macro_f1"])
        grid.append((row, layer))
    best_row, best_layer = max(grid, key=lambda rl: (rl[0]["val_balanced_accuracy"], rl[0]["val_macro_f1"]))
    model.head.load_state_dict(best_layer.state_dict())

    return finish(args, model, class_names, {
        "pooling": "cls", "trainable_blocks": 0,
        "selection": "probe cost C chosen by validation balanced accuracy (ties: macro F1)",
        "cost_grid": [r for r, _ in grid], "best": best_row,
        "official_cost": official_cost,
        "note": "Full-batch L-BFGS on frozen features (official protocol); there are no epochs, so early "
                "stopping does not apply. Validation selects C.",
        "n_train": len(y_train), "n_val": len(y_val),
    }, best_row["val_balanced_accuracy"])


# --------------------------------------------------------------------------- partial fine-tuning


def make_train_dataset(df, class_names, images_dir, seed):
    import torch
    from torchvision import transforms

    from .training import encode_labels

    # Pre-resize once to the official Resize(256) size; random crops are taken from that.
    base = round(P.IMAGE_SIZE * 256 / 224)
    images, keep = [], []
    for i, image_id in enumerate(df["image_id"]):
        try:
            images.append(resize_shorter_side(load_image(D.local_image_path(image_id, images_dir)), base))
            keep.append(i)
        except (FileNotFoundError, ValueError):
            pass
    if len(keep) < 0.95 * len(df):
        raise RuntimeError(f"Only {len(keep)}/{len(df)} training images found in {images_dir}")
    df = df.iloc[keep].reset_index(drop=True)
    labels = encode_labels(df["label"], class_names)
    augment = transforms.Compose([
        transforms.RandomResizedCrop(P.IMAGE_SIZE, scale=(0.75, 1.0)),  # official
        transforms.RandomHorizontalFlip(),                                # official
        transforms.RandomRotation(14),                                    # official 45; conservative here
        transforms.ColorJitter(brightness=0.1, contrast=0.1),             # replaces official hue=0.2
    ])

    class TrainSet(torch.utils.data.Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, i):
            x = normalize(np.asarray(augment(images[i]), dtype=np.uint8), P.NORMALIZATION)
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), int(labels[i])

    return TrainSet(), df, labels


def param_groups(model, head_lr, backbone_lr, layer_decay, weight_decay):
    """Separate AdamW groups: head at head_lr; unfrozen blocks at backbone_lr * layer_decay^depth
    (depth 0 = top block, as in the official layer-wise decay); final norm at backbone_lr.
    1-D tensors (biases, norms, layer-scale gammas) get no weight decay, as in the official code."""
    groups = {}
    first = model.first_trainable
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            lr = head_lr
        elif name.startswith("encoder.blocks."):
            depth_from_top = P.DEPTH - 1 - int(name.split(".")[2])
            lr = backbone_lr * layer_decay ** depth_from_top
            assert int(name.split(".")[2]) >= first
        else:
            lr = backbone_lr
        wd = 0.0 if p.ndim <= 1 else weight_decay
        groups.setdefault((lr, wd), []).append(p)
    return [{"params": ps, "lr": lr, "initial_lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


def run_partial_finetune(args, device):
    import torch
    from sklearn.metrics import balanced_accuracy_score, f1_score

    from .training import encode_labels

    torch.manual_seed(args.seed)
    train_df, val_df = load_split_frames(args)
    class_names = sorted(train_df["label"].unique())
    train_set, train_df, y_train = make_train_dataset(train_df, class_names, args.images_dir, args.seed)
    x_val, val_df = eval_arrays(val_df, args.images_dir)
    y_val = encode_labels(val_df["label"], class_names)

    weights = D.compute_sample_weights(train_df, class_names, tone_balance=not args.no_tone_balance)
    gen = torch.Generator().manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                                     num_samples=len(train_set), replacement=True, generator=gen)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                                         num_workers=args.num_workers, drop_last=True,
                                         pin_memory=device.type == "cuda")

    model = P.make_classifier(len(class_names), pooling="mean", n_trainable_blocks=args.trainable_blocks,
                              drop_path_rate=args.drop_path, grad_checkpointing=args.grad_checkpointing)
    model.to(device)
    opt = torch.optim.AdamW(param_groups(model, args.head_lr, args.backbone_lr, args.layer_decay,
                                         args.weight_decay))
    dtype = P.autocast_dtype(device, not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps, warmup_steps = steps_per_epoch * args.epochs, int(steps_per_epoch * args.warmup_epochs)

    def lr_scale(step):  # linear warmup then cosine to 1% of the initial LR
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    weights_path, _ = P.mode_paths(args.mode)
    weights_path = weights_path.with_name(weights_path.stem + args.output_suffix + weights_path.suffix)
    best, best_epoch, history, step = -1.0, -1, [], 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.epochs):
        model.train()
        t0, running, n_batches = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                loss = loss_fn(model(xb), yb) / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            n_batches += 1
            if (i + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * lr_scale(step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
        probs = P.predict_probs(model, x_val, device, args.feature_batch_size, not args.no_amp)
        pred = probs.argmax(1)
        row = {"epoch": epoch + 1, "train_loss": running / max(1, n_batches),
               "val_balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
               "val_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
               "val_accuracy": float((pred == y_val).mean()), "seconds": round(time.time() - t0, 1),
               "peak_memory_mb": P.peak_memory_mb(device)}
        history.append(row)
        improved = row["val_balanced_accuracy"] > best
        if improved:
            best, best_epoch = row["val_balanced_accuracy"], epoch + 1
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.trainable_state_dict(), weights_path)
        log.info("epoch %d: loss %.4f  val balanced acc %.4f  macro F1 %.4f  (%.0fs)%s", epoch + 1,
                 row["train_loss"], row["val_balanced_accuracy"], row["val_macro_f1"], row["seconds"],
                 "  * saved" if improved else "")
        if epoch + 1 - best_epoch >= args.patience:
            log.info("early stopping: no val improvement for %d epochs", args.patience)
            break

    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
    return finish(args, model, class_names, {
        "pooling": "mean", "trainable_blocks": args.trainable_blocks,
        "selection": "best epoch by validation balanced accuracy; early stopping on the same",
        "best_epoch": best_epoch, "history": history,
        "physical_batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "precision": str(dtype) if dtype is not None else "float32",
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "sampler": "WeightedRandomSampler(compute_sample_weights, replacement=True) as in official --weights",
        "n_train": len(y_train), "n_val": len(y_val),
    }, best, save_weights=False)


# --------------------------------------------------------------------------- shared


def finish(args, model, class_names, info, best_val, save_weights=True):
    import torch

    P.save_class_names(class_names)
    weights_path, meta_path = P.mode_paths(args.mode)
    if args.output_suffix:
        weights_path = weights_path.with_name(weights_path.stem + args.output_suffix + weights_path.suffix)
        meta_path = meta_path.with_name(meta_path.stem + args.output_suffix + meta_path.suffix)
    if save_weights:
        torch.save(model.trainable_state_dict(), weights_path)
    import torchvision

    meta = {"mode": args.mode, "backbone": P.NAME, "best_val_balanced_accuracy": best_val,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "splits_file": args.splits.name, "seed": args.seed, "tone_balance": not args.no_tone_balance,
            "device": str(next(model.parameters()).device), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "args": {k: str(v) for k, v in vars(args).items()},
            **info}
    meta_path.write_text(json.dumps(meta, indent=2))
    report_dir = args.reports_dir / f"{P.NAME}_{args.mode}{args.output_suffix}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_summary.json").write_text(json.dumps(meta, indent=2))
    log.info("saved %s (best val balanced accuracy %.4f)", weights_path, best_val)
    return meta


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = P.pick_device(args.device)
    info = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    log.info("device: %s%s", device, f" ({info.name}, {info.total_memory / 2**30:.1f} GiB)" if info else "")
    if args.mode == "linear_probe":
        return run_linear_probe(args, device)
    return run_partial_finetune(args, device)
