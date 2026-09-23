import argparse
import io
import json
import os
import uuid
from pathlib import Path

import torch

from app.core.config import LABELS, load_config
from app.core.paths import atomic_write, file_lock, sync_dir
from app.ml.model import build_model
from app.ml.store import safe_id


def save_checkpoint(path, model, metadata, optimizer=None, scheduler=None):
    buffer = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "metadata": metadata,
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None}, buffer)
    atomic_write(path, buffer.getvalue())


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    safe_id(meta["version"])
    if meta["labels"] != LABELS:
        raise ValueError("Checkpoint label mapping mismatch")
    config = meta["config"]
    model = build_model(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model parameters")
    model.eval()
    return model, meta, payload


def production_path(config):
    return Path(config["paths"]["models"]) / "production" / "model.pt"


def validate_candidate(config, version):
    safe_id(version)
    root = Path(config["paths"]["models"])
    candidate = root / "candidates" / version
    _, meta, _ = load_checkpoint(candidate / "best.pt")
    external = json.loads((candidate / "metadata.json").read_text())
    # JSON stringifies numeric dictionary keys.
    if json.loads(json.dumps(meta)) != external or meta["version"] != version:
        raise ValueError("Candidate metadata does not match checkpoint")
    metrics = json.loads((candidate / "metrics.json").read_text())
    if metrics != meta["validation_metrics"]:
        raise ValueError("Candidate metrics do not match checkpoint")
    return meta


def promote(config, version):
    # Share the trainer lock so an in-progress candidate can never be installed.
    with file_lock(Path(config["paths"]["data"]) / "training" / ".training.lock", blocking=False):
        return _promote(config, version)


def _promote(config, version):
    validate_candidate(config, version)
    root = Path(config["paths"]["models"])
    candidate = root / "candidates" / version
    with file_lock(root / ".promotion.lock"):
        release = root / "archive" / f"{version}-{uuid.uuid4().hex}"
        release.mkdir(parents=True)
        atomic_write(release / "model.pt", (candidate / "best.pt").read_bytes())
        atomic_write(release / "metadata.json", (candidate / "metadata.json").read_bytes())
        production = root / "production"
        if production.exists() and not production.is_symlink():
            raise ValueError("Production must be an atomic release symlink; move legacy production directory into archive first")
        temporary = root / f".production-{uuid.uuid4().hex}"
        temporary.symlink_to(release.relative_to(root), target_is_directory=True)
        os.replace(temporary, production)
        sync_dir(root)
    print(f"Promoted {version}. Previous releases retained in models/archive. Restart the server.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--config")
    args = parser.parse_args()
    promote(load_config(args.config), args.version)


if __name__ == "__main__":
    main()
