"""Independent nnUNet v2 inference for the locally supplied 2D X-ray model."""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None

from app.ml.model import get_device
from app.ml.predictor import ModelUnavailable
from app.ml.preprocessing import ImageTooLarge, validate_mask


def prepare_nnunet_image(arr, max_pixels):
    """Match the supplied nnUNet inference recipe: uint8 and 2048-high 2D input."""
    height, width = arr.shape
    target_width = max(1, round(width * 2048 / height))
    if 2048 * target_width > max_pixels:
        raise ImageTooLarge("nnUNet resized image exceeds maximum pixel count")
    gray = (arr / 256).astype(np.uint8) if arr.dtype == np.uint16 else arr.astype(np.uint8)
    resized = (cv2.resize(gray, (target_width, 2048), interpolation=cv2.INTER_LINEAR)
               if cv2 is not None else
               np.asarray(Image.fromarray(gray).resize((target_width, 2048), Image.Resampling.BILINEAR)))
    return resized.astype(np.float32)[None, None], (height, width)


class NnUnetPredictor:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["paths"]["models"]) / "pretrained" / "nnunet2"
        self.device = get_device(config)
        self._lock = threading.RLock()
        self._predictor = None
        self._version = None

    def load(self):
        checkpoint = self.root / "fold_0" / "checkpoint_best.pth"
        plans, dataset = self.root / "plans.json", self.root / "dataset.json"
        if not all(path.is_file() for path in (checkpoint, plans, dataset)):
            return
        labels = json.loads(dataset.read_text())["labels"]
        if labels != {"background": 0, "femur": 1, "tibia": 2}:
            raise ValueError("nnUNet labels do not match canonical background/femur/tibia IDs")
        plan = json.loads(plans.read_text())
        architecture = plan["configurations"]["2d"]["architecture"]["network_class_name"]
        if architecture != "dynamic_network_architectures.architectures.unet.PlainConvUNet":
            raise ValueError("Unexpected nnUNet checkpoint architecture")
        for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            os.environ.setdefault(name, "/tmp")
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                    use_mirroring=False,
                                    perform_everything_on_device=self.device.type == "cuda",
                                    device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(
            str(self.root), use_folds=(0,), checkpoint_name="checkpoint_best.pth")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with self._lock:
            self._predictor = predictor
            self._version = "nnunet2_xray_best_" + digest.hexdigest()[:12]

    def info(self):
        with self._lock:
            return {"model_loaded": self._predictor is not None,
                    "model_version": self._version,
                    "architecture": "nnUNet v2 PlainConvUNet 2D",
                    "input_height": 2048, "labels": {"background": 0, "femur": 1, "tibia": 2},
                    "cuda_active": self.device.type == "cuda"}

    def predict(self, arr):
        start = time.perf_counter()
        data, (height, width) = prepare_nnunet_image(arr, self.config["limits"]["max_pixels"])
        props = {"spacing": (999.0, 1.0, 1.0),
                 "original_size_of_raw_data": (1, 2048, data.shape[-1])}
        with self._lock:
            if self._predictor is None:
                raise ModelUnavailable("nnUNet v2 model unavailable; install its dependencies and restart backend")
            with torch.inference_mode():
                prediction = self._predictor.predict_single_npy_array(
                    data, props, None, None, save_or_return_probabilities=False)
            mask = np.asarray(prediction)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.shape != data.shape[-2:]:
                raise RuntimeError("nnUNet prediction dimensions do not match inference image")
            mask = mask.astype(np.uint8)
            validate_mask(mask)
            mask = (cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    if cv2 is not None else
                    np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST)))
            validate_mask(mask)
            return mask, self._version, (time.perf_counter() - start) * 1000
