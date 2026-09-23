"""Immutable revision files; one atomic manifest is the commit point for each case."""
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.paths import atomic_write, file_lock, write_json
from app.ml.preprocessing import InvalidImage, decode_png


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

    def training_state(self):
        path = self.root / "training_state.json"
        return json.loads(path.read_text()) if path.exists() else {"revisions": {}, "last_training_time": None, "latest_candidate_version": None}

    def status(self):
        state = self.training_state()
        records = self.records()
        return {"total_approved_cases": len(records), "new_cases_since_last_training": sum(
            state["revisions"].get(r["case_id"]) != r["revision"] for r in records),
            "last_training_time": state["last_training_time"], "latest_candidate_version": state["latest_candidate_version"]}

    def submit(self, image, mask, case_id=None, **fields):
        x = decode_png(image, self.config["limits"])
        y = decode_png(mask, self.config["limits"], mask=True)
        if x.shape != y.shape:
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
                      "previous_revision": previous["revision"] if previous else None, **fields}
            # Orphan files after interruption are harmless: only manifests define active cases.
            atomic_write(self.root / record["image"], image)
            atomic_write(self.root / record["mask"], mask)
            write_json(self.root / base / "metadata.json", record)
            write_json(self.root / "metadata" / f"{identifier}.json", record)
            status = self.status()
        return {"status": "accepted", "case_id": identifier, "training_status": "queued",
                "training_case_count": status["total_approved_cases"],
                "new_cases_since_last_training": status["new_cases_since_last_training"]}
