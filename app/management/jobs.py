"""Local job metadata around the existing trainer and promotion implementation."""
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

import yaml

from app.core.config import ROOT
from app.core.paths import atomic_write, file_lock, write_json
from app.ml.checkpoint import production_path, promote, validate_candidate
from app.ml.store import CaseStore, now, safe_id

ACTIVE = {"queued", "preparing", "training", "validating"}


def process_identity(pid):
    try:
        # State and start ticks distinguish PID reuse, including after server restart.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except (OSError, IndexError):
        return None


def public_metadata(meta):
    return {key: meta.get(key) for key in ("version", "architecture", "created_at",
        "training_case_count", "parent_model_version", "validation_metrics")}


class Management:
    def __init__(self, config, predictor):
        self.config, self.predictor = config, predictor
        self.store = CaseStore(config)
        self.root = self.store.root / "jobs"
        self.control_lock = self.store.root / ".management.lock"
        self._candidate_cache = None

    def _job_path(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid job identifier")
        return self.root / job_id

    def _read_job(self, job_id):
        root = self._job_path(job_id)
        record = json.loads((root / "status.json").read_text())
        if record["status"] in ACTIVE:
            process = json.loads((root / "process.json").read_text()) if (root / "process.json").exists() else {}
            if not process.get("identity") or process_identity(process.get("pid")) != process.get("identity"):
                # Re-read after checking liveness: the worker may have just finished.
                record = json.loads((root / "status.json").read_text())
                if record["status"] in ACTIVE:
                    record.update(status="failed", finished_at=now(), error="Training process exited without completing. See recent log.")
                    write_json(root / "status.json", record)
        log = root / "training.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 16000))
                record["recent_log"] = stream.read(16000).decode(errors="replace").splitlines()[-80:]
        else:
            record["recent_log"] = []
        return record

    def job(self, job_id):
        with file_lock(self.control_lock):
            return self._read_job(job_id)

    def latest_job(self):
        pointer = self.root / "latest.json"
        return self._read_job(json.loads(pointer.read_text())["job_id"]) if pointer.exists() else None

    def _training_busy(self):
        try:
            with file_lock(self.store.root / ".training.lock", blocking=False):
                return False
        except BlockingIOError:
            return True

    def candidate(self):
        version = self.store.training_state().get("latest_candidate_version")
        if not version:
            return None
        try:
            safe_id(version)
            path = Path(self.config["paths"]["models"]) / "candidates" / version
            signature = tuple((f.stat().st_mtime_ns, f.stat().st_size) for f in
                              (path / "best.pt", path / "metadata.json", path / "metrics.json"))
            cache_key = (version, signature)
            if self._candidate_cache and self._candidate_cache[0] == cache_key:
                return self._candidate_cache[1]
            meta = validate_candidate(self.config, version)
            result = {**public_metadata(meta), "status": "completed", "valid": True}
            self._candidate_cache = (cache_key, result)
            return result
        except Exception:
            return {"version": version, "status": "invalid", "valid": False}

    def summary(self):
        with file_lock(self.control_lock):
            job = self.latest_job()
            loaded = self.predictor.info()
            production = None
            try:
                # Resolve the release once so metadata and checkpoint refer to one version.
                release = production_path(self.config).parent.resolve()
                if (release / "model.pt").is_file():
                    production = public_metadata(json.loads((release / "metadata.json").read_text()))
            except (OSError, ValueError, TypeError):
                pass
            validation = Path(self.config["paths"]["data"]) / "validation"
            images = {p.name for p in (validation / "images").glob("*.png")}
            labels = {p.name for p in (validation / "labels").glob("*.png")}
            busy = self._training_busy()
            import torch
            return {"backend": {"status": "ok", "device": self.predictor.device.type,
                                "gpu": torch.cuda.get_device_name() if self.predictor.device.type == "cuda" else None,
                                "model_loaded": loaded["model_loaded"]},
                    "production": production, "loaded_model_version": loaded["model_version"],
                    "restart_required": bool(production and production["version"] != loaded["model_version"]),
                    "dataset": {**self.store.status(), "validation_case_count": len(images & labels),
                                "validation_pairs_match": images == labels},
                    "candidate": self.candidate(), "job": job, "training_active": busy,
                    "can_start": not busy and production_path(self.config).is_file(),
                    "training_policy": {"device": "cpu", "cpu_threads": self.config["management"]["cpu_threads"],
                                        "epochs": self.config["training"]["epochs"], "inference_available": loaded["model_loaded"]}}

    def start(self):
        with file_lock(self.control_lock):
            existing = self.latest_job()
            if existing and existing["status"] in ACTIVE:
                raise BlockingIOError("A training job is already active")
            if not production_path(self.config).is_file():
                raise ValueError("No production checkpoint. Bootstrap using finetune.sh --from-scratch first.")
            training_lock = (self.store.root / ".training.lock").open("a")
            try:
                fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                job_id = uuid.uuid4().hex
                directory = self._job_path(job_id)
                directory.mkdir(parents=True)
                config = copy.deepcopy(self.config)
                config["inference"]["device"] = "cpu"
                # Preserve server configuration/hyperparameters in an immutable job snapshot.
                atomic_write(directory / "config.yaml", yaml.safe_dump(config).encode())
                record = {"id": job_id, "status": "queued", "started_at": now(), "finished_at": None,
                          "candidate_version": None, "epoch": None, "epochs": config["training"]["epochs"],
                          "best_validation_dice": None, "error": None}
                write_json(directory / "status.json", record)
                env = {**os.environ, "PYTHONPATH": str(ROOT),
                       "OMP_NUM_THREADS": str(config["management"]["cpu_threads"]),
                       "MKL_NUM_THREADS": str(config["management"]["cpu_threads"])}
                with (directory / "training.log").open("ab") as log:
                    try:
                        process = subprocess.Popen([sys.executable, "-m", "app.management.worker",
                            str(directory), str(training_lock.fileno())], cwd=ROOT, env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            pass_fds=(training_lock.fileno(),), start_new_session=True)
                    except OSError:
                        record.update(status="failed", finished_at=now(), error="Unable to launch training process")
                        write_json(directory / "status.json", record)
                        raise
                write_json(directory / "process.json", {"pid": process.pid, "identity": process_identity(process.pid)})
                write_json(self.root / "latest.json", {"job_id": job_id})
                threading.Thread(target=process.wait, daemon=True).start()
                return record
            finally:
                # close(), not LOCK_UN: the child inherits this same locked open description.
                training_lock.close()

    def promote(self, version):
        safe_id(version)
        with file_lock(self.control_lock):
            candidate = self.candidate()
            if not candidate or not candidate["valid"] or candidate["version"] != version:
                raise ValueError("Select the valid completed candidate shown by the backend")
            promote(self.config, version)
        return {"status": "promoted", "production_model_version": version,
                "loaded_model_version": self.predictor.info()["model_version"],
                "restart_required": self.predictor.info()["model_version"] != version}
