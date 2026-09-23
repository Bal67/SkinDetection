"""Model/class-mapping agreement and loading. Requires TensorFlow; skipped otherwise."""

import json

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from skin_detection import config  # noqa: E402
from skin_detection.model import (BACKBONES, ModelLoadError, build_model, get_backbone,  # noqa: E402
                                  load_trained_model, save_metadata, unfreeze_top_of_backbone)


@pytest.fixture(scope="module")
def tiny_model():
    return build_model(num_classes=3, image_size=32, backbone_name="mobilenetv2", weights=None)


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

    save_metadata(meta_path, ["a", "b", "c"], 32, "pad", normalization="imagenet_caffe")  # unknown
    with pytest.raises(ModelLoadError, match="normalization"):
        load_trained_model(model_path, meta_path)


def test_missing_files_raise(tmp_path):
    with pytest.raises(ModelLoadError):
        load_trained_model(tmp_path / "nope.keras", tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text(json.dumps({"class_names": ["a", "a"], "image_size": 8, "resize_mode": "pad"}))
    with pytest.raises(ModelLoadError, match="duplicate"):
        load_trained_model(tmp_path / "nope.keras", tmp_path / "bad.json")


@pytest.mark.parametrize("backbone_name", sorted(BACKBONES))
def test_unfreeze_keeps_batchnorm_frozen(backbone_name):
    model = build_model(num_classes=3, image_size=32, backbone_name=backbone_name, weights=None)
    assert model.output_shape == (None, 3)
    backbone = get_backbone(model)
    assert not backbone.trainable
    fine_tune_from = BACKBONES[backbone_name]["fine_tune_from"]
    n = unfreeze_top_of_backbone(model, fine_tune_from)
    assert n > 0
    names = [layer.name for layer in backbone.layers]
    start = names.index(fine_tune_from)
    assert start > len(names) // 2  # only the top part of the network is unfrozen
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


@pytest.mark.parametrize("backbone_name", sorted(BACKBONES))
def test_trained_model_if_present(backbone_name):
    model_path, meta_path = config.model_paths(backbone_name)
    if not model_path.exists():
        pytest.skip(f"{backbone_name} model not trained")
    model, meta = load_trained_model(model_path, meta_path)
    assert model.output_shape[-1] == len(meta["class_names"])
    assert meta["backbone"] == backbone_name
    assert meta["normalization"] == BACKBONES[backbone_name]["normalization"]
