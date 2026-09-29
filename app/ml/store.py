"""Immutable revision files; one atomic manifest is the commit point for each case."""
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.paths import atomic_write, file_lock, write_json
from app.ml.preprocessing import InvalidImage, decode_png, mask_statistics


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise InvalidImage("case/model identifier must be 1-128 ASCII letters, digits, dots, underscores or hyphens")
    return value


class CaseStore:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["paths"]["data"]) / "training"
        for directory in ("images", "labels", "metadata", "revisions"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def records(self):
        return [json.loads(p.read_text()) for p in sorted((self.root / "metadata").glob("*.json"))]

    @property
    def state_path(self):
        suffix = "_medsam2" if self.config["training"].get("model") == "medsam2" else ""
        return self.root / f"training_state{suffix}.json"

    def training_state(self):
        path = self.state_path
        return json.loads(path.read_text()) if path.exists() else {"revisions": {}, "last_training_time": None, "latest_candidate_version": None}

    def status(self):
        state = self.training_state()
        records = self.records()
        return {"total_approved_cases": len(records), "new_cases_since_last_training": sum(
            state["revisions"].get(r["case_id"]) != r["revision"] for r in records),
            "last_training_time": state["last_training_time"], "latest_candidate_version": state["latest_candidate_version"]}

    def case_page(self, offset=0, limit=50):
        with file_lock(self.root / ".store.lock"):
            records = self.records()
            state = self.training_state()
            items = []
            for record in records[offset:offset + limit]:
                stats = record
                if "femur_pixels" not in stats or "tibia_pixels" not in stats:
                    stats = mask_statistics(decode_png((self.root / record["mask"]).read_bytes(),
                                                      self.config["limits"], mask=True))
                labels = [name for name, key in (("Femur", "femur_pixels"), ("Tibia", "tibia_pixels"))
                          if stats.get(key, 0) > 0]
                items.append({"case_id": record["case_id"],
                    "filename": record.get("original_filename") or record["case_id"],
                    "labels": labels or ["Background only"],
                    "overlap_pixels": stats.get("overlap_pixels", 0),
                    "status": "incorporated" if state["revisions"].get(record["case_id"]) == record["revision"] else "pending"})
            return {"items": items, "total": len(records), "offset": offset, "limit": limit}

    def submit(self, image, mask, case_id=None, **fields):
        x = decode_png(image, self.config["limits"])
        y = decode_png(mask, self.config["limits"], mask=True)
        if x.shape != y.shape[:2]:
            raise InvalidImage(f"Image dimensions {x.shape} do not match mask {y.shape}")
        digest = hashlib.sha256(image).hexdigest()
        requested = safe_id(case_id) if case_id else digest
        with file_lock(self.root / ".store.lock"):
            records = self.records()
            by_id = next((r for r in records if r["case_id"] == requested), None)
            if by_id and by_id["image_sha256"] != digest:
                raise InvalidImage("case_id already belongs to a different original image")
            previous = by_id or next((r for r in records if r["image_sha256"] == digest), None)
            identifier = previous["case_id"] if previous else requested
            revision = uuid.uuid4().hex
            base = Path("revisions") / identifier / revision
            record = {"case_id": identifier, "revision": revision, "image_sha256": digest,
                      "mask_sha256": hashlib.sha256(mask).hexdigest(), "updated_at": now(),
                      "image": str(base / "image.png"), "mask": str(base / "mask.png"),
                      "previous_revision": previous["revision"] if previous else None, **fields,
                      "mask_encoding": "rgb_discrete" if y.ndim == 3 else "class_index",
                      **mask_statistics(y)}
            # Orphan files after interruption are harmless: only manifests define active cases.
            atomic_write(self.root / record["image"], image)
            atomic_write(self.root / record["mask"], mask)
            write_json(self.root / base / "metadata.json", record)
            write_json(self.root / "metadata" / f"{identifier}.json", record)
            status = self.status()
        return {"status": "accepted", "case_id": identifier, "training_status": "queued",
                "training_case_count": status["total_approved_cases"],
                "new_cases_since_last_training": status["new_cases_since_last_training"],
                "mask_encoding": record["mask_encoding"], **mask_statistics(y)}
