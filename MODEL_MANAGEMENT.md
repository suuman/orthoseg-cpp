# Optional administrative model management

The admin menu is opt-in: `ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1 ./build/orthoseg`.
It adds **Tools → AI Model Management**, a modeless dialog separate from all
annotation controls. Without this environment variable the normal UI is unchanged.
This flag controls visibility; it is not an authentication credential.

Use the existing `MONAI_BACKEND_URL` (default `http://127.0.0.1:8000`). Backend
management must separately be enabled using `management.enabled: true` or the
provided `configs/management.yaml`. Administrative APIs are limited to direct
localhost requests; browser Origin headers and nonlocal Host headers are denied.
The backend assumes a trusted workstation, not remote/multi-user administration.

## API and behavior

Existing annotation APIs (`/health`, `/model/info`, `/segment`, `/training/cases`,
`/training/status`) remain unchanged. New operations:

- `GET /management/status`: aggregate status from backend-owned metadata/counters.
- `POST /training/start` with `{}`: launch a background worker using existing trainer defaults.
- `GET /training/jobs/{id}`: persistent job progress and bounded recent logs.
- `POST /models/{version}/promote` with `{}`: invoke existing promotion logic.

No commands, arbitrary filesystem paths, model-file uploads, or training
hyperparameter inputs are accepted. No training implementation exists in OrthoSeg.
The client owns the HTTP requests; the dialog only renders status and asks for
confirmation. The existing backend URL is reused, without filesystem coupling.

Training is from production; bootstrap weights must first be created explicitly
using the existing backend CLI. Administrative training is CPU-only with a bounded
thread count so the loaded production GPU model is unaffected. CLI GPU training
retains its existing behavior. Training/CLI/promotion share a filesystem lock;
conflicts return 409. Candidates remain separate until explicitly promoted.

The panel displays backend readiness/device/GPU, production version/architecture/
creation time/case count/parent, dataset counts, candidate status/version/parent,
and production/candidate foreground, femur and tibia Dice. Missing fields display
N/A. No judgment of model quality or automatic acceptance threshold is invented.
The latest job shows state, epoch/total epochs, best Dice, start time and useful
failure messages. Only the latest 80 log lines are displayed.

Refresh occurs on open, manually, and every five seconds while a job is active
and the panel is visible. Networking is asynchronous and duplicate actions are
disabled. Closing the panel/UI does not stop a backend worker. Training failures
leave production inference unchanged. Only a valid completed candidate can be
promoted, after confirmation; invalid/missing candidates disable the button.
Promotion refreshes metadata and tells the operator to restart the backend.
`production.version` is the disk version; `loaded_model_version` is the model still
serving inference. They can differ until restart. No hot-reloading was added.

No rollback/history editor or training-case deletion UI was added. Existing
backend archive/CLI rollback remains available.

## Files changed in OrthoSeg

- `include/MonaiClient.h`, `src/MonaiClient.cpp`: explicit management requests using
  the existing networking/error/timeout handling; annotation methods retained.
- `include/MainWindow.h`, `src/MainWindow.cpp`: opt-in Tools action/dialog lifetime.
- `CMakeLists.txt`: isolated management widget library and focused test target.
- `README.md`: setup instructions.
- Added `include/ModelManagementDialog.h`, `src/ModelManagementDialog.cpp`,
  `tests/test_management.cpp`, and this document.

Document, canvas, labels, mask storage, Save/Export hooks and all MedSAM2 code are
unchanged by this add-on. The existing integration test still verifies that
annotation submission never starts training.

## Validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The focused management test covers opt-in visibility, asynchronous status and
metadata/metric rendering, no training on opening, training confirmation/cancel,
duplicate prevention, responsiveness, job errors, invalid/missing candidates,
promotion confirmation/cancel/version selection/failure/success, loaded/disk
version separation and offline handling. Existing AI Fill, editor, document,
MONAI annotation and UI tests run alongside it.

The backend supplies `scripts/management_smoke.py` to drive the real OrthoSeg
annotation and management dialogs against the actual backend with temporary
synthetic weights/data. It verifies current corrected training data, explicit
training and promotion, archive preservation and continued inference from the
loaded old model until restart. GPU MedSAM2 execution still needs a working
NVIDIA driver; its source is untouched and existing non-GPU regressions remain.
