from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from skin_detection.inference import is_uncertain, normalized_entropy, top_k
from skin_detection.preprocessing import load_image, normalize, preprocess, to_model_size


def _png_bytes(img):
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.parametrize("mode", ["pad", "stretch"])
@pytest.mark.parametrize("size", [(300, 200), (50, 400), (224, 224)])
def test_preprocess_shape_dtype_range(mode, size):
    img = Image.new("RGB", size, (255, 0, 0))
    out = preprocess(img, 224, mode)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_pad_preserves_aspect_ratio():
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    px = to_model_size(img, 100, "pad")
    assert px.dtype == np.uint8
    assert (px[0] == 0).all() and (px[-1] == 0).all()  # padded top/bottom rows
    assert (px[50] == 255).all()  # content row spans the full width


def test_normalize_matches_mobilenet_v2():
    x = np.array([0, 127.5, 255], dtype=np.float32)
    np.testing.assert_allclose(normalize(x, "mobilenet_v2"), [-1, 0, 1])


def test_normalize_raw_passthrough_and_unknown_mode():
    x = np.array([0, 255], dtype=np.uint8)
    out = normalize(x, "raw_0_255")  # EfficientNetV2/ConvNeXt normalize inside the network
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0, 255])
    with pytest.raises(ValueError):
        normalize(x, "caffe")


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "CMYK"])
def test_load_image_converts_to_rgb(mode):
    img = Image.new(mode, (10, 10))
    buf = BytesIO()
    img.save(buf, "JPEG" if mode == "CMYK" else "PNG")
    assert load_image(buf.getvalue()).mode == "RGB"


def test_load_image_applies_exif_orientation():
    img = Image.new("RGB", (40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    buf = BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert load_image(buf.getvalue()).size == (20, 40)


def test_load_image_rejects_garbage():
    with pytest.raises(ValueError):
        load_image(b"not an image")


def test_top_k_orders_and_limits():
    names = ["a", "b", "c", "d"]
    result = top_k([0.1, 0.5, 0.15, 0.25], names, k=3)
    assert [n for n, _ in result] == ["b", "d", "c"]
    assert result[0][1] == pytest.approx(0.5)


def test_top_k_rejects_size_mismatch():
    with pytest.raises(ValueError):
        top_k([0.5, 0.5], ["a", "b", "c"])


def test_uncertainty():
    assert normalized_entropy([0.25] * 4) == pytest.approx(1.0)
    assert is_uncertain([0.25] * 4)
    assert not is_uncertain([0.97, 0.01, 0.01, 0.01])
