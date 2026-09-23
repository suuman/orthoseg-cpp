import threading
import time

import numpy as np
import torch

from app.core.config import LABELS
from app.ml.checkpoint import load_checkpoint, production_path
from app.ml.model import get_device
from app.ml.preprocessing import prepare, restore_mask


class ModelUnavailable(RuntimeError):
    pass


class Predictor:
    def __init__(self, config):
        self.config = config
        self.device = get_device(config)
        self._lock = threading.RLock()
        self._model = None
        self._metadata = None

    def load(self):
        path = production_path(self.config)
        if not path.exists():
            return
        # Loading and publication are serialized with inference, including GPU allocation.
        with self._lock:
            model, metadata, _ = load_checkpoint(path)
            model.to(self.device).eval()
            self._model, self._metadata = model, metadata

    def info(self):
        with self._lock:
            m = self._metadata or {}
            return {"model_loaded": self._model is not None, "model_version": m.get("version"),
                    "architecture": m.get("config", self.config)["model"],
                    "checkpoint_identifier": m.get("version"),
                    "input_configuration": m.get("config", self.config)["preprocessing"],
                    "labels": LABELS, "cuda_active": self.device.type == "cuda",
                    "created_at": m.get("created_at")}

    def predict(self, arr):
        start = time.perf_counter()
        with self._lock:
            if self._model is None:
                raise ModelUnavailable("No production model installed; train and promote a candidate first")
            x, geometry = prepare(arr, self._metadata["config"])
            with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                    enabled=self.device.type == "cuda" and self.config["inference"]["mixed_precision"]):
                logits = self._model(torch.from_numpy(x[None]).to(self.device))
                if logits.shape[1] != 3 or not torch.isfinite(logits).all():
                    raise RuntimeError("Model produced invalid logits")
                mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            mask = restore_mask(mask, geometry)
            version = self._metadata["version"]
        return mask, version, (time.perf_counter() - start) * 1000
