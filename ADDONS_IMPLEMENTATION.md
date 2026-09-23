# Administrative model-management implementation

## Existing behavior inspected

The backend provided GET `/health`, `/model/info`, `/training/status`, POST
`/segment` and `/training/cases`. Training and promotion were available through
`finetune.sh` and `promote_candidate.sh` only. Promotion archived releases and
required restart; there was no job API or hot reload. OrthoSeg was Qt/C++, with
an asynchronous MONAI client and an independent annotation/export integration.

## Added backend API

- GET `/management/status`: aggregate backend, disk-production, loaded-production,
  dataset, validated completed-candidate and latest-job metadata.
- POST `/training/start`, JSON `{}`: reserve the shared training lock and launch a
  detached worker that calls the existing trainer from the production checkpoint.
- GET `/training/jobs/{id}`: persisted state/progress/error and bounded log tail.
- POST `/models/{version}/promote`, JSON `{}`: reuse validated existing promotion.

These thin wrappers were necessary because the UI previously had no HTTP way to
start training, inspect job progress, or promote a candidate. They are opt-in
(`management.enabled`) and restricted to direct localhost clients, local Host
headers and no browser Origin. Unknown request-body fields are rejected. No
filesystem paths, commands or arbitrary training overrides are accepted.

The backend remains authoritative. Training uses historical/new cases, fixed
validation, configured replay/loss/augmentation/epochs and existing checkpoint
logic. No training code is duplicated in C++. Administrative jobs use CPU and
bounded threads so the loaded inference GPU model remains available. Existing
CLI device selection is unchanged. All trainers and promotion share a filesystem
lock. Workers retain that lock across backend/UI closure; status survives restart,
and process identity detects interruption. No automatic retry/cancel is added.

## UI and deliberate actions

`ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1` exposes **Tools → AI Model Management**.
Without it, normal annotators see no new management controls. The dialog is
modeless, loads on open/manual Refresh, and polls every five seconds only while
training is active and visible. It uses backend defaults and requires separate
confirmation for fine-tuning and promotion. Buttons prevent duplicate starts and
promotion of invalid/missing candidates. Missing values display N/A.

Production/candidate mean foreground, femur and tibia Dice are shown side by side.
Training shows state, epoch, best Dice, start time, error and up to 80 recent log
lines. Production-on-disk and currently-loaded versions are shown separately.
Promotion preserves old releases, refreshes metadata and explicitly requests a
backend restart. No hot reload, dataset editor, automatic model-quality decision,
or new rollback mechanism was introduced.

## Files

Backend modified: `README.md`, `configs/default.yaml`, `app/core/config.py`,
`app/main.py`, `app/ml/trainer.py`, `app/ml/checkpoint.py`.

Backend added: `configs/management.yaml`, `app/api/management.py`,
`app/management/__init__.py`, `app/management/jobs.py`, `app/management/worker.py`,
`tests/test_management.py`, `scripts/management_smoke.py`, this document.

Trainer changes add optional progress reporting and inherited-lock support; the
training algorithm is unchanged. Checkpoint changes expose its existing validation
for reuse and prevent promotion while training is writing candidates.

OrthoSeg modified: `CMakeLists.txt`, `README.md`, `include/MonaiClient.h`,
`src/MonaiClient.cpp`, `include/MainWindow.h`, `src/MainWindow.cpp`.

OrthoSeg added: `include/ModelManagementDialog.h`, `src/ModelManagementDialog.cpp`,
`tests/test_management.cpp`, `MODEL_MANAGEMENT.md`.

Document, canvas, label definitions, existing Export/Add to AI Training hooks,
and all MedSAM2/AI Fill sources are unchanged by this add-on. Prior integration
changes in the OrthoSeg working tree and the unrelated `dd.txt` were preserved.
The original design/integration/add-ons specification files were not edited.

## Verification

- Backend: 34 pytest tests pass, including real asynchronous training, continued
  inference, duplicate/API/CLI locking, invalid candidates, failed training,
  localhost restrictions, manual promotion and loaded-version behavior on restart.
- Qt: all six CTest suites pass (management, MONAI integration, AI Fill,
  segmentation, document and UI).
- Live Qt-to-backend smoke passes with isolated synthetic images/weights: original
  16-bit PNG/current corrected-mask upload does not start training; separate
  confirmed training completes; confirmed promotion retains the old archive;
  inference continues on its old loaded version until restart.
- Python compile checks passed. No synthetic data/weights were installed in the
  real production directory. GPU inference/training was not exercised by these
  CPU smoke tests; MedSAM2 source remains untouched.

## Run

Backend with optional administration:

```bash
cd /run/media/suman/Data/monai
./scripts/run_server.sh --config configs/management.yaml
```

OrthoSeg:

```bash
cd /home/suman/agentic_coding/orthoseg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1 ./build/orthoseg
```

Both repositories retain the existing `MONAI_BACKEND_URL` default
`http://127.0.0.1:8000`. Use the normal backend startup command/UI launch without
admin flags to keep management disabled/hidden. A production checkpoint is still
required before fine-tuning; bootstrap explicitly using the existing CLI if absent.

**Save/Export/Add to AI Training never start fine-tuning. No automatic training or
promotion occurs on startup, refresh, case submission, or case-count thresholds.**
