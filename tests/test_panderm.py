"""PanDerm_Base integration tests. Run in the PyTorch environment (requirements-panderm.txt);
skipped where torch, the official code clone or the pretrained checkpoint is unavailable."""

import json
from argparse import Namespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
pytest.importorskip("timm")

from skin_detection import config  # noqa: E402
from skin_detection import panderm as P  # noqa: E402
from skin_detection.preprocessing import normalize, preprocess, to_model_size  # noqa: E402

needs_repo = pytest.mark.skipif(
    not (P.REPO_DIR / "classification" / "models" / "modeling_finetune.py").exists(),
    reason="official PanDerm code not cloned (third_party/PanDerm)")
needs_ckpt = pytest.mark.skipif(not P.CHECKPOINT_PATH.exists(), reason="PanDerm_Base checkpoint not downloaded")


@pytest.mark.parametrize("size", [(300, 200), (200, 300), (640, 480), (256, 256), (1000, 333)])
def test_preprocessing_matches_official_eval_transform(size):
    """Our single PIL implementation must equal PanDerm's torchvision Resize(256)+CenterCrop(224)+Normalize."""
    from torchvision import transforms

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    official = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))])
    ours = preprocess(img, P.IMAGE_SIZE, P.RESIZE_MODE, P.NORMALIZATION)
    assert ours.shape == (224, 224, 3) and ours.dtype == np.float32
    np.testing.assert_allclose(ours, official(img).numpy().transpose(1, 2, 0), atol=1e-5)


def test_panderm_normalization_constants():
    x = np.array([[[0.485 * 255, 0.456 * 255, 0.406 * 255]]], dtype=np.float32)
    np.testing.assert_allclose(normalize(x, "panderm_imagenet"), 0, atol=1e-5)
    assert to_model_size(Image.new("RGB", (50, 80)), 224, "resize_center_crop").shape == (224, 224, 3)


@needs_repo
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_features_match_official_forward(pooling):
    model = P.make_classifier(5, pooling, n_trainable_blocks=2, checkpoint=None).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model.features(x), model.encoder.forward_features(x, is_train=False), atol=1e-5)
        assert model(x).shape == (2, 5)


@needs_repo
def test_linear_probe_trains_only_the_head():
    model = P.make_classifier(26, "cls", n_trainable_blocks=0, checkpoint=None)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["head.weight", "head.bias"]


@needs_repo
def test_partial_finetune_unfreezes_only_top_blocks_and_uses_lower_backbone_lr():
    from skin_detection.panderm_training import param_groups

    model = P.make_classifier(26, "mean", n_trainable_blocks=2, checkpoint=None)
    trainable = {n.split(".")[2] for n, p in model.named_parameters() if p.requires_grad and ".blocks." in n}
    assert trainable == {"10", "11"}
    assert not any(p.requires_grad for n, p in model.named_parameters() if "patch_embed" in n or "cls_token" in n)
    groups = param_groups(model, head_lr=1e-3, backbone_lr=2e-5, layer_decay=0.65, weight_decay=0.05)
    head_lrs = {g["lr"] for g in groups if any(p is model.head.weight for p in g["params"])}
    block_lrs = {g["lr"] for g in groups if g["lr"] not in head_lrs}
    assert head_lrs == {1e-3}
    assert max(block_lrs) == pytest.approx(2e-5) and min(block_lrs) == pytest.approx(2e-5 * 0.65)
    model.train()
    assert not model.encoder.blocks[0].training and model.encoder.blocks[11].training


def test_logistic_regression_head_fits_separable_data():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, (3, 16))
    y = np.repeat(np.arange(3), 30)
    x = centers[y] + rng.normal(0, 0.5, (90, 16))
    layer = P.fit_logistic_regression(x, y, np.ones(90), 3, C=10.0, max_iter=200)
    with torch.no_grad():
        assert (layer(torch.as_tensor(x, dtype=torch.float32)).argmax(1).numpy() == y).mean() > 0.95


def test_training_never_loads_test_rows():
    from skin_detection.panderm_training import load_split_frames

    if not config.SPLITS_CSV.exists():
        pytest.skip("no splits file")
    args = Namespace(splits=config.SPLITS_CSV, max_train_samples=None, max_val_samples=None, seed=0)
    train, val = load_split_frames(args)
    assert set(train["split"]) == {"train"} and set(val["split"]) == {"val"}
    assert not set(train["image_id"]) & set(val["image_id"])


@needs_ckpt
def test_checkpoint_hash_is_verified(tmp_path):
    bad = tmp_path / "fake.pth"
    torch.save({"x": torch.zeros(1)}, bad)
    with pytest.raises(P.PanDermSetupError, match="sha256"):
        P.load_pretrained_state_dict(bad)


@needs_repo
@needs_ckpt
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_official_checkpoint_loads_strictly(pooling):
    enc = P.build_encoder(pooling, verify_hash=False)  # raises on any missing/unexpected key
    assert sum(p.numel() for p in enc.parameters()) > 85_000_000  # ViT-B/16


@needs_repo
@needs_ckpt
def test_class_mapping_must_match_head(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PANDERM_DIR", tmp_path)
    monkeypatch.setattr(P, "CLASS_NAMES_PATH", tmp_path / "class_names.json")
    model = P.make_classifier(3, "cls", n_trainable_blocks=0, verify_hash=False)
    weights, meta = P.mode_paths("linear_probe")
    torch.save(model.trainable_state_dict(), weights)
    meta.write_text(json.dumps({"pooling": "cls", "trainable_blocks": 0}))
    P.save_class_names(["a", "b", "c"])
    loaded, info = P.load_classifier("linear_probe", verify_hash=False)
    assert loaded.head.out_features == len(info["class_names"]) == 3

    from skin_detection.model import ModelLoadError

    (tmp_path / "class_names.json").write_text(json.dumps({**json.loads((tmp_path / "class_names.json").read_text()),
                                                            "class_names": ["a", "b"], "num_classes": 2}))
    with pytest.raises(ModelLoadError, match="head has 3 outputs"):
        P.load_classifier("linear_probe", verify_hash=False)


@pytest.mark.parametrize("mode", P.MODES)
def test_trained_panderm_if_present(mode):
    weights, _ = P.mode_paths(mode)
    if not weights.exists() or not P.CHECKPOINT_PATH.exists():
        pytest.skip(f"PanDerm {mode} not trained")
    model, meta = P.load_classifier(mode, verify_hash=False)
    assert model.head.out_features == len(meta["class_names"]) == 26
    probs = P.predict_probs(model, np.zeros((2, 224, 224, 3), np.float32), torch.device("cpu"))
    assert probs.shape == (2, 26) and np.allclose(probs.sum(1), 1, atol=1e-4)
