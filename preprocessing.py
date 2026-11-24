"""Image loading and preprocessing for the plant disease TFLite model."""

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

ColorOrder = Literal["RGB", "BGR"]
INPUT_SIZE = (224, 224)


def preprocess_image(
    image: str | Path | np.ndarray,
    *,
    color_order: ColorOrder = "RGB",
    target_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Return a normalized float32 image in NHWC shape (1, height, width, 3).

    File paths are read as RGB. NumPy images default to RGB; pass
    ``color_order="BGR"`` for frames from OpenCV.
    """
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be 'RGB' or 'BGR'")
    if len(target_size) != 2 or any(size <= 0 for size in target_size):
        raise ValueError("target_size must contain two positive dimensions")

    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            rgb_image = source.convert("RGB")
            resized = rgb_image.resize(
                (target_size[1], target_size[0]), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(resized, dtype=np.float32)
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NumPy image must have shape (height, width, 3)")
        if image.size == 0:
            raise ValueError("image must not be empty")
        if not np.issubdtype(image.dtype, np.number):
            raise TypeError("NumPy image must contain numeric pixel values")
        pixels = image.astype(np.float32, copy=False)
        if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 255:
            raise ValueError("pixel values must be finite and in the range [0, 255]")
        if np.issubdtype(image.dtype, np.floating) and pixels.max() <= 1.0:
            pixels = pixels * 255.0
        if color_order == "BGR":
            pixels = pixels[..., ::-1]
        resized = Image.fromarray(pixels.astype(np.uint8)).resize(
            (target_size[1], target_size[0]), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype=np.float32)
    else:
        raise TypeError("image must be a file path or a NumPy array")

    return np.expand_dims(pixels / 255.0, axis=0).astype(np.float32)
