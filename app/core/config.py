import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LABELS = {0: "background", 1: "femur", 2: "tibia"}


def load_config(path=None):
    with (ROOT / "configs/default.yaml").open() as f:
        config = yaml.safe_load(f)
    selected = path or os.environ.get("XRAY_CONFIG")
    if selected:
        with Path(selected).open() as f:
            override = yaml.safe_load(f) or {}
        def merge(dst, src):
            for key, value in src.items():
                if key not in dst:
                    raise ValueError(f"Unknown configuration key: {key}")
                if isinstance(value, dict) and isinstance(dst[key], dict):
                    merge(dst[key], value)
                else:
                    dst[key] = value
        merge(config, override)
    config = copy.deepcopy(config)
    for key in config["paths"]:
        config["paths"][key] = str((ROOT / config["paths"][key]).resolve())
    p, m, t = config["preprocessing"], config["model"], config["training"]
    if not 0 <= p["lower_percentile"] < p["upper_percentile"] <= 100:
        raise ValueError("Invalid normalization percentiles")
    if m["architecture"] != "unet" or len(m["channels"]) != len(m["strides"]) + 1:
        raise ValueError("Invalid UNet architecture")
    if any(x < 1 for x in m["channels"] + m["strides"]):
        raise ValueError("Model dimensions must be positive")
    if p["max_size"] < 16 or any(t[k] < 1 for k in ("epochs", "batch_size", "new_case_repeat_factor")):
        raise ValueError("Invalid image size or training configuration")
    if t["learning_rate"] <= 0 or t["num_workers"] < 0:
        raise ValueError("Invalid training configuration")
    if any(config["limits"][k] < 1 for k in ("max_file_bytes", "max_pixels")):
        raise ValueError("Upload limits must be positive")
    a = t["augmentation"]
    if not 0 <= a["probability"] <= 1 or any(a[k] < 0 for k in a if k != "probability"):
        raise ValueError("Invalid augmentation configuration")
    if not 1 <= config["server"]["port"] <= 65535:
        raise ValueError("Invalid server port")
    if not isinstance(config["management"]["enabled"], bool):
        raise ValueError("management.enabled must be boolean")
    threads = config["management"]["cpu_threads"]
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("management.cpu_threads must be between 1 and 32")
    return config
