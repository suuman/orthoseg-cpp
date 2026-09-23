import torch
from monai.networks.nets import UNet


def build_model(config):
    m = config["model"]
    if m["architecture"] != "unet":
        raise ValueError(f"Unsupported architecture: {m['architecture']}")
    return UNet(spatial_dims=2, in_channels=1, out_channels=3,
                channels=m["channels"], strides=m["strides"], num_res_units=m["num_res_units"])


def get_device(config):
    value = config["inference"]["device"]
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(value)
