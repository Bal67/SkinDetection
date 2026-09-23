"""Model/class-mapping agreement and loading. Requires TensorFlow; skipped otherwise."""

import json

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from skin_detection import config  # noqa: E402
from skin_detection.model import (ModelLoadError, build_model, get_backbone, load_trained_model,  # noqa: E402
                                  save_metadata, unfreeze_top_of_backbone)


@pytest.fixture(scope="module")
def tiny_model():
    return build_model(num_classes=3, image_size=32, weights=None)


def test_saved_model_roundtrip_and_validation(tiny_model, tmp_path):
    model_path, meta_path = tmp_path / "m.keras", tmp_path / "class_names.json"
    tiny_model.save(model_path)
    save_metadata(meta_path, ["a", "b", "c"], 32, "pad")
    model, meta = load_trained_model(model_path, meta_path)
    assert model.output_shape[-1] == len(meta["class_names"])

    save_metadata(meta_path, ["a", "b"], 32, "pad")  # wrong number of classes
    with pytest.raises(ModelLoadError, match="outputs 3 classes"):
        load_trained_model(model_path, meta_path)

    save_metadata(meta_path, ["a", "b", "c"], 64, "pad")  # wrong image size
    with pytest.raises(ModelLoadError, match="input shape"):
        load_trained_model(model_path, meta_path)


def test_missing_files_raise(tmp_path):
    with pytest.raises(ModelLoadError):
        load_trained_model(tmp_path / "nope.keras", tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text(json.dumps({"class_names": ["a", "a"], "image_size": 8, "resize_mode": "pad"}))
    with pytest.raises(ModelLoadError, match="duplicate"):
        load_trained_model(tmp_path / "nope.keras", tmp_path / "bad.json")


def test_unfreeze_keeps_batchnorm_frozen(tiny_model):
    backbone = get_backbone(tiny_model)
    assert not backbone.trainable
    n = unfreeze_top_of_backbone(tiny_model, "block_13_expand")
    assert n > 0
    names = [layer.name for layer in backbone.layers]
    start = names.index("block_13_expand")
    for i, layer in enumerate(backbone.layers):
        if isinstance(layer, keras.layers.BatchNormalization):
            assert not layer.trainable
        elif i < start:
            assert not layer.trainable


def test_legacy_model_loads_with_matching_class_names():
    if not config.LEGACY_MODEL_PATH.exists():
        pytest.skip("legacy model not present")
    model, meta = load_trained_model(config.LEGACY_MODEL_PATH, config.LEGACY_MODEL_META_PATH)
    assert model.output_shape[-1] == len(meta["class_names"]) == 26
    assert meta["class_names"] == sorted(meta["class_names"])  # sklearn LabelEncoder order
    probs = model.predict(np.zeros((1, 128, 128, 3), np.float32), verbose=0)
    assert probs.shape == (1, 26)
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)


def test_trained_model_if_present():
    if not config.MODEL_PATH.exists():
        pytest.skip("retrained model not present")
    model, meta = load_trained_model(config.MODEL_PATH, config.MODEL_META_PATH)
    assert model.output_shape[-1] == len(meta["class_names"])
