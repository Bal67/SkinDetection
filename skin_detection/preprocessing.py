"""The single canonical image preprocessing used by training, evaluation and the Streamlit app.

Pipeline:  bytes/file -> PIL (EXIF-transposed, RGB) -> geometry (to a square of image_size)
           -> float32 -> backbone-specific normalization (recorded in the model's metadata).

Training inserts augmentation between `to_model_size` (uint8 geometry) and `normalize`;
nothing else differs between training and inference.

Geometry ("resize_mode"):
  * "pad"     (default for new models): scale the longer side to image_size keeping aspect
              ratio, then pad the shorter side symmetrically with black. No distortion and no
              part of the image (e.g. a lesion near the edge) is cropped away.
  * "stretch" (legacy model only): plain PIL resize to image_size x image_size, which is what
              the 2024 model was trained with. It distorts rectangular images and exists only so
              the legacy model is fed the input it was trained on.
  * "resize_center_crop" (PanDerm): PanDerm's official eval transform, torchvision
              Resize(256) + CenterCrop(224): scale the SHORTER side to image_size * 256 / 224, then
              center-crop image_size x image_size. Unlike "pad", this can cut off image borders; it is
              used because it is what PanDerm was pretrained/evaluated with.
Interpolation is bilinear in all modes.
"""

import os
from io import BytesIO
from typing import Union

import numpy as np
from PIL import Image, ImageOps

RESIZE_MODES = ("pad", "stretch", "resize_center_crop")
# "mobilenet_v2":     x / 127.5 - 1 (MobileNetV2 expects [-1, 1]).
# "raw_0_255":        float32 pixels in [0, 255]; EfficientNetV2 and ConvNeXt normalize inside the network.
# "panderm_imagenet": (x / 255 - mean) / std with PanDerm's official constants. NOTE: the official
#                     code uses std 0.228 for the red channel (standard ImageNet is 0.229); we copy
#                     the official value because that is what the released model was trained with.
NORMALIZATIONS = ("mobilenet_v2", "raw_0_255", "panderm_imagenet")
PANDERM_MEAN = (0.485, 0.456, 0.406)
PANDERM_STD = (0.228, 0.224, 0.225)
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
    elif resize_mode == "resize_center_crop":
        out = center_crop(resize_shorter_side(img, round(image_size * 256 / 224)), image_size)
    else:
        w, h = img.size
        scale = image_size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), _RESAMPLE)
        out = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        out.paste(resized, ((image_size - new_w) // 2, (image_size - new_h) // 2))
    return np.asarray(out, dtype=np.uint8)


def resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    """Same arithmetic as torchvision.transforms.Resize(size) on a PIL image (longer side truncated)."""
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    return img.resize((new_w, new_h), _RESAMPLE)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Same arithmetic as torchvision.transforms.CenterCrop(size) for images at least `size` wide/high."""
    w, h = img.size
    top, left = int(round((h - size) / 2.0)), int(round((w - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def normalize(pixels, normalization: str = "mobilenet_v2"):
    """[0, 255] pixels -> the float32 input the backbone expects. Works on numpy arrays and tf tensors.

    "mobilenet_v2" is identical to keras.applications.mobilenet_v2.preprocess_input (x / 127.5 - 1),
    written out so the app does not need to import it.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {normalization!r}")
    if isinstance(pixels, np.ndarray):
        x = pixels.astype(np.float32)
    else:
        import tensorflow as tf

        x = tf.cast(pixels, tf.float32)
    if normalization == "mobilenet_v2":
        return x / 127.5 - 1.0
    if normalization == "panderm_imagenet":
        mean = np.asarray(PANDERM_MEAN, np.float32) * 255.0
        std = np.asarray(PANDERM_STD, np.float32) * 255.0
        return (x - mean) / std
    return x


def preprocess(img: Image.Image, image_size: int, resize_mode: str = "pad",
               normalization: str = "mobilenet_v2") -> np.ndarray:
    """Full inference preprocessing for one PIL image -> float32 (image_size, image_size, 3)."""
    return normalize(to_model_size(img, image_size, resize_mode), normalization)
