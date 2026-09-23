"""Prompted MedSAM2 inference from the locally supplied SAM2.1 checkpoint."""
import hashlib
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.ml.model import get_device
from app.ml.predictor import ModelUnavailable
from app.ml.preprocessing import InvalidImage


def parse_box(value, shape):
    if value is None:
        return None
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
    """Turn an 8/16-bit X-ray into the RGB uint8 input expected by SAM2."""
    p = config["preprocessing"]
    x = arr.astype(np.float32)
    low, high = np.percentile(x, [p["lower_percentile"], p["upper_percentile"]])
    if high > low:
        x = np.clip((x - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    else:
        x = np.zeros(arr.shape, dtype=np.uint8)
    return np.repeat(x[:, :, None], 3, axis=2)


def sam2_input(arr, config):
    """Resize to 1024 pixels high, preserving aspect ratio and source geometry."""
    height, width = arr.shape
    target_width = max(1, round(width * 1024 / height))
    rgb = sam2_rgb(arr, config)
    resized = np.asarray(Image.fromarray(rgb).resize((target_width, 1024), Image.Resampling.BILINEAR)).copy()
    return resized, target_width / width, 1024 / height


class Sam2Predictor:
    def __init__(self, config):
        self.config = config
        self.device = get_device(config)
        self.root = Path(config["paths"]["models"]) / "pretrained" / "medsam2"
        self._lock = threading.RLock()
        self._predictor = None
        self._version = None

    def load(self):
        checkpoint, model_config = self.root / "checkpoint.pt", self.root / "sam2.1_hiera_t.yaml"
        if not checkpoint.is_file() or not model_config.is_file():
            return
        # The supplied SAM2 package lives alongside its local weights. No network lookup.
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from sam2.build import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(str(model_config), str(checkpoint), device=str(self.device)).eval()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        version = "medsam2_hiera_t_" + digest.hexdigest()[:12]
        with self._lock:
            self._predictor = SAM2ImagePredictor(model)
            self._version = version

    def info(self):
        with self._lock:
            return {"model_loaded": self._predictor is not None, "model_version": self._version,
                    "architecture": "SAM2.1 Hiera Tiny", "requires_prompts": True,
                    "supported_prompts": ["femur_box", "tibia_box"]}

    def predict(self, arr, boxes):
        if not boxes:
            raise InvalidImage("Provide at least one femur_box or tibia_box")
        validated = {label: parse_boxes(box, arr.shape) for label, box in boxes.items()}
        if any(label not in (1, 2) for label in validated):
            raise InvalidImage("SAM2 supports only femur and tibia boxes")
        start = time.perf_counter()
        with self._lock:
            if self._predictor is None:
                raise ModelUnavailable("Prompted SAM2 model unavailable; install local SAM2 dependencies and restart backend")
            output = np.zeros(arr.shape, dtype=np.uint8)
            confidence = np.full(arr.shape, -np.inf, dtype=np.float32)
            with torch.inference_mode():
                rgb, scale_x, scale_y = sam2_input(arr, self.config)
                self._predictor.set_image(rgb)
                for label, bone_boxes in validated.items():
                    for box in bone_boxes:
                        scaled = box * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
                        masks, scores, _ = self._predictor.predict(box=scaled, multimask_output=False)
                        if masks.shape != (1, *rgb.shape[:2]) or not np.isfinite(scores).all():
                            raise RuntimeError("SAM2 produced an invalid mask or score")
                        original_mask = np.asarray(Image.fromarray(np.asarray(masks[0], dtype=np.uint8))
                                                   .resize((arr.shape[1], arr.shape[0]), Image.Resampling.NEAREST), dtype=bool)
                        winner = original_mask & (float(scores[0]) > confidence)
                        output[winner] = label
                        confidence[winner] = float(scores[0])
            return output, self._version, (time.perf_counter() - start) * 1000
