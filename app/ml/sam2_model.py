"""Shared model construction and reference/active checkpoint resolution."""
import copy
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def sam2_config(config):
    selected = copy.deepcopy(config)
    selected["training"]["model"] = "medsam2"
    return selected


def asset_root(config):
    return Path(config["paths"]["models"]) / config["sam2"]["asset_directory"]


def build_sam2_model(config, checkpoint=None, device="cpu"):
    root = asset_root(config)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sam2.build import build_sam2
    model = build_sam2(str(root / "sam2.1_hiera_t.yaml"),
                       str(checkpoint) if checkpoint else None,
                       device=str(device), apply_postprocessing=False)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("MedSAM2 checkpoint contains non-finite parameters")
    return model


def reference_metadata(config):
    path = asset_root(config) / "checkpoint.pt"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"version": "medsam2_hiera_t_" + digest[:12], "architecture": "medsam2",
            "checkpoint_path": str(path), "checkpoint_sha256": digest,
            "status": "reference", "parent_model_version": None,
            "created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "mask_encoding": "rgb_discrete", "training_case_count": None}


def parent_path(config):
    from app.ml.checkpoint import production_path
    active = production_path(sam2_config(config))
    return active if active.is_file() else asset_root(config) / "checkpoint.pt"


def load_active(config, device):
    from app.ml.checkpoint import load_checkpoint, production_path
    path = production_path(sam2_config(config))
    if path.is_file():
        model, meta, payload = load_checkpoint(path)
        return model.to(device), {**meta, "checkpoint_path": str(path.resolve()), "status": "active"}, payload
    path = asset_root(config) / "checkpoint.pt"
    return build_sam2_model(config, path, device), reference_metadata(config), None
