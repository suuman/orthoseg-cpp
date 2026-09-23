import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)


class InvalidImage(ValueError):
    pass


class ImageTooLarge(InvalidImage):
    pass


def decode_png(data, limits, mask=False):
    if len(data) > limits["max_file_bytes"]:
        raise ImageTooLarge("PNG exceeds maximum file size")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != "PNG" or getattr(im, "n_frames", 1) != 1:
                raise InvalidImage("Expected a single-frame PNG")
            if im.width * im.height > limits["max_pixels"]:
                raise ImageTooLarge("PNG exceeds maximum pixel count")
            if mask:
                if im.mode != "L":
                    raise InvalidImage("Mask must be single-channel 8-bit grayscale PNG")
            elif im.mode in ("RGB", "RGBA"):
                log.info("Converting RGB X-ray to grayscale for preprocessing")
                im = im.convert("L")
            elif im.mode not in ("L", "I", "I;16", "I;16B", "I;16L"):
                raise InvalidImage(f"Unsupported X-ray mode: {im.mode}")
            im.load()
            arr = np.array(im)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImage("Invalid or damaged PNG") from exc
    if arr.size == 0:
        raise InvalidImage("Empty image")
    if mask:
        validate_mask(arr)
    return arr


def validate_mask(arr):
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise InvalidImage("Mask must be single-channel uint8")
    values = np.unique(arr)
    if not np.isin(values, [0, 1, 2]).all():
        raise InvalidImage(f"Mask contains invalid labels: {values.tolist()}; expected only 0, 1, 2")


def encode_mask(arr):
    validate_mask(arr)
    stream = io.BytesIO()
    Image.fromarray(arr).save(stream, format="PNG")
    return stream.getvalue()


@dataclass(frozen=True)
class Geometry:
    original: tuple
    resized: tuple
    padded: tuple


def prepare(arr, config):
    p = config["preprocessing"]
    x = arr.astype(np.float32)
    low, high = np.percentile(x, [p["lower_percentile"], p["upper_percentile"]])
    x = np.clip((x - low) / (high - low), 0, 1) if high > low else np.zeros_like(x)
    h, w = x.shape
    scale = min(1., p["max_size"] / max(h, w))
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    stride = int(np.prod(config["model"]["strides"]))
    # At least two bottleneck pixels per axis for instance normalization.
    ph, pw = (max(2 * stride, ((n + stride - 1) // stride) * stride) for n in (rh, rw))
    x = np.asarray(Image.fromarray(x.astype(np.float32)).resize((rw, rh), Image.Resampling.BILINEAR))
    x = np.pad(x, ((0, ph-rh), (0, pw-rw)))
    return x[None].astype(np.float32), Geometry((h, w), (rh, rw), (ph, pw))


def prepare_mask(mask, geometry):
    rh, rw = geometry.resized
    ph, pw = geometry.padded
    resized = np.asarray(Image.fromarray(mask).resize((rw, rh), Image.Resampling.NEAREST))
    return np.pad(resized, ((0, ph-rh), (0, pw-rw)))[None].astype(np.int64)


def restore_mask(mask, geometry):
    validate_mask(mask)
    if mask.shape != geometry.padded:
        raise ValueError("Prediction shape does not match preprocessing geometry")
    rh, rw = geometry.resized
    h, w = geometry.original
    result = np.asarray(Image.fromarray(mask[:rh, :rw]).resize((w, h), Image.Resampling.NEAREST))
    validate_mask(result)
    return result
