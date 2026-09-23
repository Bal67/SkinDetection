"""The single canonical image preprocessing used by training, evaluation and the Streamlit app.

Pipeline:  bytes/file -> PIL (EXIF-transposed, RGB) -> geometry (to a square of image_size)
           -> float32 -> MobileNetV2 normalization ([-1, 1]).

Training inserts augmentation between `to_model_size` (uint8 geometry) and `normalize`;
nothing else differs between training and inference.

Geometry ("resize_mode"):
  * "pad"     (default for new models): scale the longer side to image_size keeping aspect
              ratio, then pad the shorter side symmetrically with black. No distortion and no
              part of the image (e.g. a lesion near the edge) is cropped away.
  * "stretch" (legacy model only): plain PIL resize to image_size x image_size, which is what
              the 2024 model was trained with. It distorts rectangular images and exists only so
              the legacy model is fed the input it was trained on.
Interpolation is bilinear in both modes.
"""

import os
from io import BytesIO
from typing import Union

import numpy as np
from PIL import Image, ImageOps

RESIZE_MODES = ("pad", "stretch")
_RESAMPLE = Image.BILINEAR


def load_image(source: Union[str, bytes, BytesIO, os.PathLike]) -> Image.Image:
    """Open an image file/bytes, apply EXIF orientation and convert to RGB.

    Raises ValueError if the data cannot be decoded as an image.
    """
    if isinstance(source, bytes):
        source = BytesIO(source)
    try:
        img = Image.open(source)
        img.load()
    except Exception as exc:  # PIL raises several exception types for bad input
        raise ValueError(f"Could not read image: {exc}") from exc
    img = ImageOps.exif_transpose(img)
    return to_rgb(img)


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL mode (RGBA, P, L, CMYK, ...) to RGB. Transparency is composited on white."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def to_model_size(img: Image.Image, image_size: int, resize_mode: str = "pad") -> np.ndarray:
    """Geometry step: returns a uint8 array of shape (image_size, image_size, 3)."""
    if resize_mode not in RESIZE_MODES:
        raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {resize_mode!r}")
    img = to_rgb(img)
    if resize_mode == "stretch":
        out = img.resize((image_size, image_size), _RESAMPLE)
    else:
        w, h = img.size
        scale = image_size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), _RESAMPLE)
        out = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        out.paste(resized, ((image_size - new_w) // 2, (image_size - new_h) // 2))
    return np.asarray(out, dtype=np.uint8)


def normalize(pixels) -> np.ndarray:
    """MobileNetV2 normalization: [0, 255] -> [-1, 1] float32.

    Identical to keras.applications.mobilenet_v2.preprocess_input (x / 127.5 - 1), written out
    so the app does not need to import it and so it works on both numpy arrays and tf tensors.
    """
    if isinstance(pixels, np.ndarray):
        return pixels.astype(np.float32) / 127.5 - 1.0
    import tensorflow as tf

    return tf.cast(pixels, tf.float32) / 127.5 - 1.0


def preprocess(img: Image.Image, image_size: int, resize_mode: str = "pad") -> np.ndarray:
    """Full inference preprocessing for one PIL image -> float32 (image_size, image_size, 3)."""
    return normalize(to_model_size(img, image_size, resize_mode))
