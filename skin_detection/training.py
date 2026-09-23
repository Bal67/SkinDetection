"""Image loading into memory, training-time augmentation, and two-stage training.

Data discipline (enforced by the callers in scripts/):
  * The split (data/splits.csv) is made on original images BEFORE augmentation.
  * Augmentation is applied on the fly to TRAINING batches only; no augmented files are written,
    so an augmented copy can never land in another partition.
  * Model selection, early stopping, LR schedule and any threshold decisions use VALIDATION data.
  * The test split is never touched during training; scripts/evaluate.py is the only consumer.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data as data_mod
from .preprocessing import load_image, normalize, to_model_size

log = logging.getLogger(__name__)


def load_images(df: pd.DataFrame, images_dir, image_size: int, resize_mode: str,
                max_missing_frac: float = 0.05) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load + geometry-transform images for df rows -> (uint8 array (n, S, S, 3), kept rows).

    Missing/unreadable images are dropped and reported; if more than max_missing_frac are
    missing the run fails rather than silently training on a different dataset.
    """
    arrays, keep = [], []
    missing = []
    for i, image_id in enumerate(df["image_id"]):
        path = data_mod.local_image_path(image_id, images_dir)
        try:
            arrays.append(to_model_size(load_image(path), image_size, resize_mode))
            keep.append(i)
        except (FileNotFoundError, ValueError):
            missing.append(image_id)
    if missing:
        log.warning("%d/%d images missing or unreadable (e.g. %s)", len(missing), len(df), missing[:3])
    if len(df) and len(missing) / len(df) > max_missing_frac:
        raise RuntimeError(
            f"{len(missing)}/{len(df)} images missing from {images_dir}. "
            "Run scripts/prepare_dataset.py --download first (or point SKIN_IMAGES_DIR at your images)."
        )
    if not arrays:
        return np.zeros((0, image_size, image_size, 3), np.uint8), df.iloc[[]]
    return np.stack(arrays), df.iloc[keep].reset_index(drop=True)


def build_augmenter(seed: int):
    """Conservative, label-preserving augmentation on [0, 255] images.

    Deliberately excluded: color inversion (not a plausible skin image), hue/saturation jitter
    (skin and lesion color are diagnostically relevant, and recoloring must not be used to
    "simulate" darker skin), vertical flips (clinical photos have an anatomical orientation).
    """
    import keras

    geometric = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.04, fill_mode="constant", fill_value=0.0, seed=seed + 1),  # about +/-14 degrees
        keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0, seed=seed + 2),
        keras.layers.RandomZoom((-0.1, 0.1), fill_mode="constant", fill_value=0.0, seed=seed + 3),
    ], name="augment")

    def augment(x):
        import tensorflow as tf

        x = geometric(tf.cast(x, tf.float32), training=True)
        x = tf.image.random_brightness(x, max_delta=0.1 * 255.0, seed=seed + 4)
        x = tf.image.random_contrast(x, 0.9, 1.1, seed=seed + 5)
        return tf.clip_by_value(x, 0.0, 255.0)

    return augment


def make_dataset(x_uint8: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
                 batch_size: int = 32, training: bool = False, seed: int = 42):
    """tf.data pipeline: [augment (train only)] -> canonical normalize -> batch."""
    import tensorflow as tf

    tensors = (x_uint8, y) if sample_weight is None else (x_uint8, y, sample_weight)
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        ds = ds.shuffle(len(x_uint8), seed=seed, reshuffle_each_iteration=True)
        augment = build_augmenter(seed)
        ds = ds.map(lambda x, *rest: (augment(x), *rest), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, *rest: (normalize(x), *rest), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def val_balanced_accuracy_callback(val_ds, y_val: np.ndarray):
    """Adds logs['val_balanced_accuracy'] each epoch (mean per-class recall on validation).

    Used for checkpointing/early stopping instead of plain accuracy, which would reward
    predicting the majority classes. Must be placed BEFORE callbacks that read it.
    """
    import keras
    from sklearn.metrics import balanced_accuracy_score

    class _Cb(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(val_ds, verbose=0).argmax(axis=1)
            score = balanced_accuracy_score(y_val, preds)
            if logs is not None:
                logs["val_balanced_accuracy"] = score
            print(f" - val_balanced_accuracy: {score:.4f}")

    return _Cb()


def train_two_stage(model, train_ds, val_ds, y_val, checkpoint_path: Path,
                    head_epochs: int = 15, head_lr: float = 1e-3,
                    finetune_epochs: int = 20, finetune_lr: float = 1e-5,
                    fine_tune_from: str = "block_13_expand", patience: int = 5):
    """Stage A: frozen backbone, train head. Stage B: unfreeze top of backbone (BN stays frozen),
    recompile with a much smaller LR, continue. One ModelCheckpoint instance spans both stages so
    the file on disk is the best validation checkpoint across the whole run."""
    import keras

    from .model import unfreeze_top_of_backbone

    monitor, mode = "val_balanced_accuracy", "max"
    checkpoint = keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor=monitor, mode=mode,
                                                 save_best_only=True, verbose=1)
    bal_acc = val_balanced_accuracy_callback(val_ds, y_val)

    def callbacks():
        return [
            bal_acc,
            checkpoint,
            keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        ]

    history = {}
    model.compile(optimizer=keras.optimizers.Adam(head_lr), loss="sparse_categorical_crossentropy",
                  weighted_metrics=["accuracy"])
    h = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=callbacks())
    history["stage_a_head"] = {k: [float(v) for v in vals] for k, vals in h.history.items()}

    n_unfrozen = 0
    if finetune_epochs > 0:
        n_unfrozen = unfreeze_top_of_backbone(model, fine_tune_from)
        model.compile(optimizer=keras.optimizers.Adam(finetune_lr), loss="sparse_categorical_crossentropy",
                      weighted_metrics=["accuracy"])
        h = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs, callbacks=callbacks())
        history["stage_b_finetune"] = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    history["unfrozen_backbone_layers_with_weights"] = n_unfrozen
    history["best_val_balanced_accuracy"] = float(checkpoint.best)
    return history


def encode_labels(labels: Sequence[str], class_names: Sequence[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(class_names)}
    unknown = set(labels) - set(index)
    if unknown:
        raise ValueError(f"Labels not in class_names: {sorted(unknown)}")
    return np.array([index[label] for label in labels], dtype=np.int32)
