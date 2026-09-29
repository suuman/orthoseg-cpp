# Optional administrative model management

The AI Fill panel contains a **Model Management** button. It is enabled only
after the selected local backend accepts `GET /management/status`, and opens a
modeless dialog with a UNet / MedSAM2 selector. The optional Tools menu shortcut remains available with
`ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1 ./build/orthoseg`; it follows the same
backend permission check. The flag is not an authentication credential.

Use the URL in AI Fill, initially taken from `MONAI_BACKEND_URL` (default
`http://127.0.0.1:8000`). Backend
management must separately be enabled using `management.enabled: true` or the
provided `configs/management.yaml`. Administrative APIs are limited to direct
localhost requests; browser Origin headers and nonlocal Host headers are denied.
The backend assumes a trusted workstation, not remote/multi-user administration.

## API and behavior

Existing annotation APIs (`/health`, `/model/info`, `/segment`, `/training/cases`,
`/training/status`) remain unchanged. New operations:

- `GET /management/status` (optional `?model=medsam2`): aggregate status from backend-owned metadata/counters.
- `POST /training/start` with `{}` or `{"model":"medsam2"}`: launch a background worker using existing trainer defaults.
- `GET /training/jobs/{id}`: persistent job progress and bounded recent logs.
- `POST /models/{version}/promote` with `{}` or `{"model":"medsam2"}`: invoke existing promotion logic.

No commands, arbitrary filesystem paths, model-file uploads, or training
hyperparameter inputs are accepted. No training implementation exists in OrthoSeg.
The client owns the HTTP requests; the dialog only renders status and asks for
confirmation. The existing backend URL is reused, without filesystem coupling.

UNet training starts from production; bootstrap weights are created using the backend CLI. MedSAM2 starts from the active checkpoint or the installed updated reference when no candidate has been promoted. It consumes independent Femur/Tibia channels and preserves overlap. Administrative training is CPU-only with a bounded
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

The updated MedSAM2 integration also extends document, canvas, mask-transfer and Save/Export handling to preserve overlapping channels. The existing integration test still verifies that
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
loaded old model until restart. The backend also supplies `scripts/medsam2_e2e.py`, which uses the actual updated MedSAM2 checkpoint and drives OrthoSeg editing, export/upload, fine-tuning and promotion with isolated synthetic data, then restarts and checks the new checkpoint.

## Corrected-case reminder

In Model Management, select UNet or MedSAM2 and set **Remind me to fine-tune after
corrected cases**. The threshold is saved separately for each family; 0 means
Off. The most recently configured family receives reminders. A case counts only
after a successful **Add to AI Training** upload. Counts come from that model's
backend training state, so repeated edits to one case do not become multiple
cases. Brush strokes and Save Only do not count.

At the threshold the app offers to open Model Management. It never starts a job
from a reminder or upload. Use **Fine-tune New Candidate** and confirm to start;
inspect the resulting candidate and promote separately. Declining suppresses
repeat reminders for the current milestone. A completed training run resets the
pending-case count. Reminders are suppressed while training is active or the
backend cannot start training.

UI training requires backend `management.enabled: true`, a parent model, and a
fixed validation dataset. The toolbar button appears when access is available;
`ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1` additionally exposes the Tools menu entry.
If UI management is unavailable, the workstation script is
`/run/media/suman/Data/monai/scripts/finetune.sh`. Pass `--model unet` or
`--model medsam2`, `--dry-run` to validate, and `--epochs N` for a training run.
For MedSAM2 on this workstation set `XRAY_PYTHON=/home/suman/deepnet/bin/python`.
UNet initial training requires `--from-scratch`. Candidate promotion is separate.

## Uploaded files

The **Files added to fine-tuning** table lists the backend's active approved cases,
including original filename, Femur/Tibia labels (or Background only), overlap
pixel count, and incorporation status for the selected model. It displays 50
cases per page and supports Previous files/Next files. Refresh updates the list;
switching model resets pagination and shows that model's training state. Older
records without a filename use their case ID. The list contains submitted cases,
not every mask exported from the editor.

**Export list (CSV)** saves all pages for the selected model, including filenames,
labels, overlap pixels, training status, model and case IDs. A failed or cancelled
export leaves any existing destination file unchanged.
