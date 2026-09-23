import argparse
from contextlib import nullcontext
import os
import hashlib
import json
import logging
import platform
import uuid
from pathlib import Path

import monai
import numpy as np
import torch
import yaml
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from app.core.config import LABELS, load_config
from app.core.logging import configure_logging
from app.core.paths import atomic_write, file_lock, write_json
from app.ml.checkpoint import load_checkpoint, production_path, save_checkpoint
from app.ml.dataset import XrayDataset, pad_batch, snapshot
from app.ml.metrics import segmentation_metrics
from app.ml.model import build_model, get_device
from app.ml.preprocessing import decode_png, prepare, restore_mask
from app.ml.store import CaseStore, now

log = logging.getLogger(__name__)


def environment():
    return {"python": platform.python_version(), "pytorch": str(torch.__version__),
            "monai": str(monai.__version__), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def evaluate(model, pairs, config, device):
    model.eval()
    def predictions():
        with torch.inference_mode():
            for image, label in pairs:
                x, geom = prepare(decode_png(image.read_bytes(), config["limits"]), config)
                logits = model(torch.from_numpy(x[None]).to(device))
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite validation logits")
                pred = restore_mask(logits.argmax(1)[0].cpu().numpy().astype(np.uint8), geom)
                yield pred, decode_png(label.read_bytes(), config["limits"], mask=True)
    return segmentation_metrics(predictions())


def train(config, source="production", checkpoint=None, dry_run=False, progress=None, _lock_fd=None):
    store = CaseStore(config)
    lock_path = store.root / ".training.lock"
    if _lock_fd is not None:
        # Administrative subprocess inherits the already-held lock; no race with CLI training.
        if os.fstat(_lock_fd).st_ino != lock_path.stat().st_ino or os.fstat(_lock_fd).st_dev != lock_path.stat().st_dev:
            raise ValueError("Invalid inherited training lock")
    with (nullcontext() if _lock_fd is not None else file_lock(lock_path, blocking=False)):
        return _train(config, store, source, checkpoint, dry_run, progress or (lambda **_: None))


def _train(config, store, source, checkpoint, dry_run, progress):
    progress(status="preparing")
    print("Femur/Tibia X-ray MONAI Fine-tuning", flush=True)
    print(json.dumps(environment(), indent=2), flush=True)
    device = get_device(config)
    set_determinism(seed=config["training"]["seed"])
    parent, payload = None, None
    if source == "scratch":
        model = build_model(config)
    else:
        path = Path(checkpoint) if source == "checkpoint" else production_path(config)
        if not path.exists():
            raise ValueError("No production checkpoint. Supply --from-scratch explicitly or --from-checkpoint PATH")
        model, parent, payload = load_checkpoint(path)
        if parent["config"]["model"] != config["model"] or parent["config"]["preprocessing"] != config["preprocessing"]:
            raise ValueError("Model/preprocessing configuration must match the parent checkpoint")
    with file_lock(store.root / ".store.lock"):
        records, training, validation = snapshot(store, config)
        state = store.training_state()
    new_indices = [i for i, r in enumerate(records) if state["revisions"].get(r["case_id"]) != r["revision"]]
    print(f"Device: {device}\nParent: {parent['version'] if parent else 'scratch'}\n"
          f"Approved training cases: {len(training)}\nNew/revised cases: {len(new_indices)}\n"
          f"Validation cases: {len(validation)}", flush=True)
    if dry_run:
        print("Dry run passed: environment, checkpoint and dataset integrity verified.")
        return None
    t = config["training"]
    replay = training + [training[i] for i in new_indices for _ in range(t["new_case_repeat_factor"] - 1)]
    loader = DataLoader(XrayDataset(replay, config, augment=True), batch_size=t["batch_size"],
                        shuffle=True, num_workers=t["num_workers"], collate_fn=pad_batch,
                        generator=torch.Generator().manual_seed(t["seed"]))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    if payload and t["resume_optimizer"]:
        if payload.get("optimizer") is None or payload.get("scheduler") is None:
            raise ValueError("Checkpoint lacks optimizer/scheduler state")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
    loss_fn = DiceCELoss(include_background=t["include_background"], to_onehot_y=True, softmax=True)
    amp = device.type == "cuda" and t["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    version = "femur_tibia_2d_" + uuid.uuid4().hex
    candidate = Path(config["paths"]["models"]) / "candidates" / version
    candidate.mkdir(parents=True, exist_ok=False)
    progress(candidate_version=version, parent_model_version=parent["version"] if parent else None)
    atomic_write(candidate / "training_config.yaml", yaml.safe_dump(config).encode())
    write_json(candidate / "dataset_snapshot.json", {"training": records,
        "validation": [{"filename": x.name, "image_sha256": hashlib.sha256(x.read_bytes()).hexdigest(),
                         "mask_sha256": hashlib.sha256(y.read_bytes()).hexdigest()} for x, y in validation]})
    best = -1.
    history = []
    for epoch in range(t["epochs"]):
        progress(status="training", epoch=epoch + 1, epochs=t["epochs"])
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(batch["image"].to(device)), batch["label"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        progress(status="validating")
        metrics = evaluate(model, validation, config, device)
        score = metrics["mean_foreground_dice"]
        if score is None:
            raise ValueError("Validation metrics undefined: no foreground in predictions or reference labels")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics})
        write_json(candidate / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            metadata = {"version": version, "parent_model_version": parent["version"] if parent else None,
                        "created_at": now(), "architecture": config["model"]["architecture"],
                        "training_case_count": len(training), "new_cases_incorporated": len(new_indices),
                        "validation_metrics": metrics, "best_epoch": epoch + 1, "labels": LABELS,
                        "config": config, "environment": environment(),
                        "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
            save_checkpoint(candidate / "best.pt", model, metadata, optimizer, scheduler)
            write_json(candidate / "metadata.json", metadata)
            write_json(candidate / "metrics.json", metrics)
        progress(best_validation_dice=best, metrics=metrics)
    write_json(candidate / "completed.json", {"version": version, "completed_at": now()})
    with file_lock(store.root / ".store.lock"):
        write_json(store.root / "training_state.json", {"revisions": {r["case_id"]: r["revision"] for r in records},
                   "last_training_time": now(), "latest_candidate_version": version})
    print(f"Best validation results: {json.loads((candidate / 'metrics.json').read_text())}\n"
          f"Candidate saved: {candidate / 'best.pt'}\nProduction model has NOT been replaced.", flush=True)
    return version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-production", action="store_true")
    group.add_argument("--from-checkpoint")
    group.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    configure_logging()
    try:
        config = load_config(args.config)
        if args.epochs is not None:
            if args.epochs < 1:
                raise ValueError("epochs must be positive")
            config["training"]["epochs"] = args.epochs
        train(config, "scratch" if args.from_scratch else "checkpoint" if args.from_checkpoint else "production",
              args.from_checkpoint, args.dry_run)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Fine-tuning failed: {exc}\n")


if __name__ == "__main__":
    main()
