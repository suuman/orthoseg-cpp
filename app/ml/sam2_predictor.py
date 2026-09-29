"""Prompted MedSAM2 inference from the locally supplied SAM2.1 checkpoint."""
import threading
import time

import numpy as np
import torch
from PIL import Image

from app.ml.model import get_device
from app.ml.predictor import ModelUnavailable
from app.ml.preprocessing import InvalidImage


def parse_box(value, shape):
    if not isinstance(value, list) or len(value) != 4 or any(type(x) is not int for x in value):
        raise InvalidImage("Each SAM2 box must be four integer pixel coordinates [x0,y0,x1,y1]")
    x0, y0, x1, y1 = value
    h, w = shape
    if not (0 <= x0 < x1 < w and 0 <= y0 < y1 < h):
        raise InvalidImage("SAM2 box must have positive area and lie within the original image")
    return np.asarray(value, dtype=np.float32)


def parse_boxes(value, shape):
    """Accept one legacy box or up to two boxes for bilateral anatomy."""
    if isinstance(value, list) and len(value) == 4 and all(type(x) is int for x in value):
        return [parse_box(value, shape)]
    if not isinstance(value, list) or not 1 <= len(value) <= 2:
        raise InvalidImage("Provide one or two SAM2 boxes per bone")
    return [parse_box(box, shape) for box in value]


def sam2_rgb(arr, config):
    # Match medsam2_infer_custom.py: retain uint8, min/max normalize higher depth.
    if arr.dtype == np.uint8:
        x = arr
    else:
        x = arr.astype(np.float32)
        low, high = float(x.min()), float(x.max())
        x = ((x - low) / (high - low + 1e-8) * 255).astype(np.uint8)
    return np.repeat(x[:, :, None], 3, axis=2)


def sam2_input(arr, config):
    height, width = arr.shape
    target_width = max(1, round(width * 1024 / height))
    # Bound the square allocation as well as the input image.
    if max(1024, target_width) ** 2 > config["limits"]["max_pixels"]:
        raise InvalidImage("MedSAM2 square canvas exceeds the pixel limit")
    rgb = sam2_rgb(arr, config)
    resized = np.asarray(Image.fromarray(rgb).resize((target_width, 1024), Image.Resampling.BILINEAR)).copy()
    return resized, target_width / width, 1024 / height


def square_input(arr, config):
    rgb, sx, sy = sam2_input(arr, config)
    h, w = rgb.shape[:2]
    side = max(h, w)
    y, x = (side - h) // 2, (side - w) // 2
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[y:y+h, x:x+w] = rgb
    return canvas, (sx, sy, x, y, w, h)


def prompt_canvas(binary, geometry, side):
    _, _, x, y, w, h = geometry
    resized = np.asarray(Image.fromarray(binary.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST))
    canvas = np.zeros((side, side), np.uint8)
    canvas[y:y+h, x:x+w] = resized
    return canvas


def mask_logits(binary):
    t = torch.as_tensor(binary.copy(), dtype=torch.float32)[None, None]
    small = torch.nn.functional.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return ((small[0] > 0.5).float() * 20 - 10).numpy()


def mask_box(binary):
    ys, xs = np.nonzero(binary)
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], np.float32) if xs.size else None


class Sam2Predictor:
    def __init__(self, config):
        self.config = config
        self.device = get_device(config)
        self._lock = threading.RLock()
        self._info_lock = threading.Lock()
        self._predictor = None
        self._version = None
        self._metadata = {}

    def load(self):
        from app.ml.sam2_model import parent_path, load_active
        if not parent_path(self.config).is_file():
            return
        with self._lock:
            model, meta, _ = load_active(self.config, self.device)
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor = SAM2ImagePredictor(model.eval())
            with self._info_lock:
                self._predictor = predictor
                self._version, self._metadata = meta["version"], meta

    def info(self):
        with self._info_lock:
            return {"model_loaded": self._predictor is not None, "model_version": self._version,
                    "architecture": "SAM2.1 Hiera Tiny", "requires_prompts": True,
                    "supported_prompts": ["femur_box", "tibia_box", "mask"],
                    "mask_encoding": "rgb_discrete", "labels": {"femur": 1, "tibia": 2},
                    "checkpoint": self._metadata.get("checkpoint_path"),
                    "model_status": self._metadata.get("status")}

    def predict(self, arr, boxes, prompt=None):
        from app.ml.preprocessing import validate_mask, class_mask
        validated = {label: parse_boxes(box, arr.shape) for label, box in boxes.items()}
        if any(label not in (1, 2) for label in validated):
            raise InvalidImage("SAM2 supports only femur and tibia")
        if prompt is not None:
            validate_mask(prompt)
            if prompt.shape[:2] != arr.shape:
                raise InvalidImage("Prompt mask dimensions must match the original image")
        if not validated and (prompt is None or not prompt.any()):
            raise InvalidImage("Provide a femur/tibia box or a non-empty labeled mask")
        start = time.perf_counter()
        with self._lock:
            if self._predictor is None:
                raise ModelUnavailable("Prompted MedSAM2 unavailable; install local assets and restart backend")
            output = np.zeros((*arr.shape, 3), dtype=np.uint8)
            with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                    dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                rgb, geometry = square_input(arr, self.config)
                sx, sy, x, y, w, h = geometry
                self._predictor.set_image(rgb)
                for label in (1, 2):
                    prompts = []
                    for box in validated.get(label, []):
                        scaled = box * np.array([sx, sy, sx, sy], np.float32) + [x, y, x, y]
                        prompts.append({"box": scaled.astype(np.float32)})
                    if prompt is not None and class_mask(prompt, label).any():
                        binary = prompt_canvas(class_mask(prompt, label), geometry, rgb.shape[0])
                        prompts.append({"mask_input": mask_logits(binary), "box": mask_box(binary)})
                    for kwargs in prompts:
                        masks, scores, _ = self._predictor.predict(**kwargs, multimask_output=False, return_logits=True)
                        if masks.shape != (1, *rgb.shape[:2]) or not np.isfinite(scores).all() or not np.isfinite(masks).all():
                            raise RuntimeError("MedSAM2 produced an invalid mask or score")
                        binary = (masks[0, y:y+h, x:x+w] > 0).astype(np.uint8)
                        restored = np.asarray(Image.fromarray(binary).resize(
                            (arr.shape[1], arr.shape[0]), Image.Resampling.NEAREST), dtype=bool)
                        output[..., label - 1][restored] = label
            return output, self._version, (time.perf_counter() - start) * 1000
