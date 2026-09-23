"""Detached administrative job process. Calls the same trainer used by finetune.sh."""
import json
import os
from pathlib import Path
import sys
import traceback

from app.core.config import load_config
from app.core.paths import write_json
from app.ml.store import now


def run(directory, lock_fd):
    path = directory / "status.json"
    record = json.loads(path.read_text())
    def progress(**changes):
        record.update(changes, updated_at=now())
        write_json(path, record)
    try:
        import torch
        from app.ml.trainer import train
        config = load_config(directory / "config.yaml")
        torch.set_num_threads(config["management"]["cpu_threads"])
        version = train(config, progress=progress, _lock_fd=lock_fd)
        progress(status="completed", candidate_version=version, finished_at=now())
    except BaseException as exc:
        traceback.print_exc()
        progress(status="failed", error=str(exc)[:2000] or type(exc).__name__, finished_at=now())
        return 1
    finally:
        os.close(lock_fd)
    return 0


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]), int(sys.argv[2])))
