"""Model construction, two-stage fine-tuning helpers, and validated save/load.

A trained model is always paired with a metadata JSON (models/skin_<backbone>_class_names.json)
that stores the exact class order, backbone and preprocessing settings used in training. Loading validates that:
  * the metadata exists and is well-formed,
  * the model's output dimension equals len(class_names),
  * the model's input size equals the metadata image_size.
Any failure raises ModelLoadError; there is no fallback to an untrained network.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DEFAULT_BACKBONE
from .preprocessing import NORMALIZATIONS, RESIZE_MODES

# Supported ImageNet backbones (all CNNs from keras.applications; no extra dependencies).
#   constructor: keras.applications attribute
#   normalization: what the backbone expects as input (see preprocessing.NORMALIZATIONS)
#   fine_tune_from: first layer unfrozen in stage B (roughly the last quarter to third of the network)
BACKBONES = {
    # 2018. Baseline kept for comparability with the original project.
    "mobilenetv2": {"constructor": "MobileNetV2", "normalization": "mobilenet_v2",
                    "fine_tune_from": "block_13_expand"},
    # 2021. Default: newer, more accurate than MobileNetV2 and still cheap enough to train on a CPU.
    "efficientnetv2b0": {"constructor": "EfficientNetV2B0", "normalization": "raw_0_255",
                         "fine_tune_from": "block6a_expand_conv"},
    # 2022. The newest CNN family in keras.applications; ~6x the compute of EfficientNetV2-B0 (GPU advised).
    "convnext_tiny": {"constructor": "ConvNeXtTiny", "normalization": "raw_0_255",
                      "fine_tune_from": "convnext_tiny_downsampling_block_2"},
}


class ModelLoadError(RuntimeError):
    pass


# --------------------------------------------------------------------------- metadata


def save_metadata(path, class_names, image_size, resize_mode, normalization="mobilenet_v2", **extra):
    meta = {
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "image_size": int(image_size),
        "resize_mode": resize_mode,
        "normalization": normalization,
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2))
    return meta


def load_metadata(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"Class-name/metadata file not found: {path}")
    try:
        meta = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"Metadata file {path} is not valid JSON: {exc}") from exc
    names = meta.get("class_names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ModelLoadError(f"{path}: 'class_names' must be a non-empty list of strings")
    if len(set(names)) != len(names):
        raise ModelLoadError(f"{path}: duplicate entries in 'class_names'")
    if "num_classes" in meta and meta["num_classes"] != len(names):
        raise ModelLoadError(f"{path}: num_classes={meta['num_classes']} but {len(names)} class names")
    if meta.get("resize_mode") not in RESIZE_MODES:
        raise ModelLoadError(f"{path}: resize_mode must be one of {RESIZE_MODES}")
    if not isinstance(meta.get("image_size"), int):
        raise ModelLoadError(f"{path}: image_size must be an integer")
    if meta.get("normalization") not in NORMALIZATIONS:
        raise ModelLoadError(f"{path}: normalization must be one of {NORMALIZATIONS}")
    return meta


def validate_model_against_metadata(model, meta: dict) -> None:
    out_dim = model.output_shape[-1]
    if out_dim != len(meta["class_names"]):
        raise ModelLoadError(
            f"Model outputs {out_dim} classes but metadata lists {len(meta['class_names'])}. "
            "The model and class_names.json do not belong together."
        )
    in_shape = tuple(model.input_shape[1:])
    expected = (meta["image_size"], meta["image_size"], 3)
    if in_shape != expected:
        raise ModelLoadError(f"Model input shape {in_shape} != metadata image size {expected}")


# --------------------------------------------------------------------------- architecture


def build_model(num_classes: int, image_size: int, backbone_name: str = DEFAULT_BACKBONE,
                dense_units: int = 256, dropout: float = 0.5, l2: float = 1e-4,
                weights: Optional[str] = "imagenet"):
    """ImageNet backbone -> GAP -> Dense(relu) -> Dropout -> Dense(softmax).

    The backbone starts frozen (stage A). It is always called with training=False so its
    BatchNormalization layers stay in inference mode even after layers are unfrozen in stage B;
    this is the conservative BN handling recommended for fine-tuning on small datasets.
    """
    import keras

    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone_name!r}; choose from {sorted(BACKBONES)}")
    constructor = getattr(keras.applications, BACKBONES[backbone_name]["constructor"])
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    backbone = constructor(weights=weights, include_top=False, input_shape=(image_size, image_size, 3))
    backbone.trainable = False
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(dense_units, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(l2), name="head_dense")(x)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"skin_{backbone_name}")


def get_backbone(model):
    """The nested pretrained network (the only sub-model inside our classifier)."""
    import keras

    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No backbone sub-model found in model")


def unfreeze_top_of_backbone(model, fine_tune_from: str) -> int:
    """Stage B: unfreeze backbone layers from `fine_tune_from` onward (see BACKBONES for the
    per-backbone default), keeping every BatchNormalization layer frozen. Returns the number of
    trainable backbone layers that have weights."""
    import keras

    backbone = get_backbone(model)
    names = [layer.name for layer in backbone.layers]
    if fine_tune_from not in names:
        raise ValueError(f"Layer {fine_tune_from!r} not in backbone")
    start = names.index(fine_tune_from)
    backbone.trainable = True
    n_trainable = 0
    for i, layer in enumerate(backbone.layers):
        trainable = i >= start and not isinstance(layer, keras.layers.BatchNormalization)
        layer.trainable = trainable
        n_trainable += int(trainable and bool(layer.weights))
    return n_trainable


# --------------------------------------------------------------------------- loading


def _build_legacy_architecture(num_classes: int, image_size: int):
    """The exact architecture of the 2024 models (Sequential, frozen backbone)."""
    import keras

    backbone = keras.applications.MobileNetV2(
        weights=None, include_top=False, input_shape=(image_size, image_size, 3)
    )
    return keras.Sequential([
        keras.Input((image_size, image_size, 3)),
        backbone,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])


def _load_legacy_h5_weights(model, h5_path) -> None:
    """Load weights from the 2024 H5 file by *layer name*.

    keras.models.load_model cannot deserialize that file (Keras 3.4 wrote a nested-Sequential
    config that current Keras rejects) and Keras' positional load_weights assigns weights to
    the wrong layers. Every weight is matched by name and shape, and every model weight must
    be found, otherwise ModelLoadError is raised.
    """
    import h5py

    backbone = model.layers[0]
    dense_layers = [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]
    with h5py.File(h5_path, "r") as f:
        mw = f["model_weights"]
        saved_dense = [n for n in mw.attrs["layer_names"] if str(n).startswith("dense")]
        saved_dense = [n.decode() if isinstance(n, bytes) else str(n) for n in saved_dense]
        if len(saved_dense) != len(dense_layers):
            raise ModelLoadError("Legacy H5 head does not match the expected architecture")

        def read_group(group_name):
            g = mw[group_name]
            out = {}
            for wname in g.attrs["weight_names"]:
                wname = wname.decode() if isinstance(wname, bytes) else str(wname)
                # key by "<layer>/<variable>" (the last two path components)
                out["/".join(wname.split("/")[-2:])] = np.asarray(g[wname])
            return out

        backbone_weights = read_group(backbone.name)
        loaded = 0
        for layer in backbone.layers:
            for var in layer.weights:
                key = f"{layer.name}/{var.name}"
                if key not in backbone_weights:
                    raise ModelLoadError(f"Legacy H5 missing backbone weight {key}")
                value = backbone_weights[key]
                if tuple(value.shape) != tuple(var.shape):
                    raise ModelLoadError(f"Shape mismatch for {key}: {value.shape} vs {var.shape}")
                var.assign(value)
                loaded += 1
        for layer, saved_name in zip(dense_layers, saved_dense):
            saved = read_group(saved_name)
            for var in layer.weights:
                key = f"{saved_name}/{var.name}"
                if key not in saved or tuple(saved[key].shape) != tuple(var.shape):
                    raise ModelLoadError(f"Legacy H5 head weight {key} missing or wrong shape")
                var.assign(saved[key])
                loaded += 1
    if loaded != len(model.weights):
        raise ModelLoadError(f"Only {loaded}/{len(model.weights)} legacy weights were loaded")


def load_trained_model(model_path, meta_path):
    """Load a trained model plus its metadata, validating that they agree.

    Returns (model, meta). Raises ModelLoadError on any problem.
    """
    model_path, meta_path = Path(model_path), Path(meta_path)
    meta = load_metadata(meta_path)
    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")

    import keras

    try:
        if meta.get("legacy_format") == "keras3.4_sequential_h5":
            model = _build_legacy_architecture(len(meta["class_names"]), meta["image_size"])
            _load_legacy_h5_weights(model, model_path)
        else:
            model = keras.models.load_model(model_path, compile=False)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model from {model_path}: {exc}") from exc

    validate_model_against_metadata(model, meta)
    return model, meta
