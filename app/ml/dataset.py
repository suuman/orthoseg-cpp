import hashlib
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Compose, RandAffined, RandGaussianNoised, RandScaleIntensityd
from torch.utils.data import Dataset

from app.ml.preprocessing import decode_png, prepare, prepare_mask


def validate_pair(image, mask, config):
    raw = image.read_bytes()
    x = decode_png(raw, config["limits"])
    y = decode_png(mask.read_bytes(), config["limits"], mask=True)
    if x.shape != y.shape:
        raise ValueError(f"Image/mask dimensions differ for {image.name}")
    return hashlib.sha256(raw).hexdigest()


def snapshot(store, config):
    records = store.records()
    training = []
    hashes = set()
    for r in records:
        image, mask = store.root / r["image"], store.root / r["mask"]
        digest = validate_pair(image, mask, config)
        if digest != r["image_sha256"] or hashlib.sha256(mask.read_bytes()).hexdigest() != r["mask_sha256"]:
            raise ValueError(f"Case integrity check failed: {r['case_id']}")
        hashes.add(digest)
        training.append((image, mask))
    root = Path(config["paths"]["data"]) / "validation"
    images = {p.name: p for p in (root / "images").glob("*.png")}
    labels = {p.name: p for p in (root / "labels").glob("*.png")}
    if images.keys() != labels.keys():
        raise ValueError("Validation images and labels must have matching filenames")
    validation = []
    for name in sorted(images):
        if validate_pair(images[name], labels[name], config) in hashes:
            raise ValueError("Validation image also exists in training pool")
        validation.append((images[name], labels[name]))
    if not training or not validation:
        raise ValueError("At least one approved training case and one fixed validation case are required")
    return records, training, validation


class XrayDataset(Dataset):
    def __init__(self, pairs, config, augment=False):
        self.pairs, self.config = pairs, config
        self.transform = None
        if augment:
            a = config["training"]["augmentation"]
            self.transform = Compose([
                RandAffined(keys=["image", "label"], prob=a["probability"],
                            rotate_range=a["rotation_radians"], scale_range=(a["scale"], a["scale"]),
                            translate_range=(a["translation_pixels"], a["translation_pixels"]),
                            mode=("bilinear", "nearest"), padding_mode="zeros"),
                RandScaleIntensityd(keys="image", factors=a["intensity"], prob=a["probability"]),
                RandGaussianNoised(keys="image", std=a["noise_std"], prob=a["probability"]),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, mask = self.pairs[index]
        x = decode_png(image.read_bytes(), self.config["limits"])
        y = decode_png(mask.read_bytes(), self.config["limits"], mask=True)
        x, geom = prepare(x, self.config)
        sample = {"image": x, "label": prepare_mask(y, geom)}
        if self.transform:
            sample = self.transform(sample)
        return {"image": torch.as_tensor(np.asarray(sample["image"]), dtype=torch.float32),
                "label": torch.as_tensor(np.asarray(sample["label"]), dtype=torch.long)}


def pad_batch(samples):
    h = max(s["image"].shape[-2] for s in samples)
    w = max(s["image"].shape[-1] for s in samples)
    return {key: torch.stack([torch.nn.functional.pad(s[key],
            (0, w-s[key].shape[-1], 0, h-s[key].shape[-2])) for s in samples]) for key in ("image", "label")}
