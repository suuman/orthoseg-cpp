# Updated MedSAM2 implementation

Implemented from `medsamupate.md` (the supplied filename). The backend remains in
`/run/media/suman/Data/monai`; the Qt/C++ application remains in
`/home/suman/agentic_coding/orthoseg`. No second backend or training service was added.

## Run the applications

Backend, using the existing model-enabled Python environment:

```bash
cd /run/media/suman/Data/monai
./scripts/run_server_models.sh --config configs/management.yaml
```

Frontend:

```bash
cd /home/suman/agentic_coding/orthoseg
cmake --build build --parallel
MONAI_BACKEND_URL=http://127.0.0.1:8000 ./build/orthoseg
```

The existing combined launcher also works:

```bash
/home/suman/agentic_coding/orthoseg/scripts/launch_with_monai.sh
```

Restart an already-running backend to load this update. In AI Fill select
**MONAI MedSAM2 (boxes / masks)**. Its health dot checks the prompted model's own
readiness. Use Bounding Box (including the second box for bilateral cases),
Paint Mask, Load Mask or Normal Fill Mask. Mask prompts target the selected
Femur or Tibia; the unprompted bone's annotation is preserved. Loaded RGB label
prompts extract the selected channel, and grayscale binary prompts remain supported.
Use AI Result as Next Prompt applies the current result and prepares the selected
bone for refinement. Background/Fibula are not MedSAM2 refinement targets.

Review the RGB preview and apply it. Brush, selected-label eraser, fill, Draw & Fill,
Lasso, undo and isolated view continue to use independent document channels.
Export Mask and batch Save Mask & Next offer **Add to AI Training** or **Save Only**.
The source image and corrected mask are submitted before a batch advances.
Saving or uploading never starts training automatically.

Open **Model Management**, select **MedSAM2**, inspect corrected sample counts,
then choose **Fine-tune New Candidate**. The existing status/log/epoch/Dice display
tracks the job. Choose **Promote Candidate** after reviewing it, then restart the
backend to activate the new checkpoint. The UNet selection and its existing
training/promotion controls remain available. Management remains disabled unless
the localhost backend grants the existing management privilege.

## Installed model and preprocessing

The new reference assets are copied to `models/pretrained/medsam2_updated/`:

- `checkpoint.pt`: source `/run/media/suman/Data/medsam2/checkpoint.pt`.
- `sam2.1_hiera_t.yaml` and `sam2/`: matching architecture/configuration/package.
- `medsam2_infer_custom.py`: the supplied behavioral reference.
- `manifest.json`: source, checksum, encoding and installation provenance.

The initial reference version is `medsam2_hiera_t_8113ea7212e9`. Neither the
source checkpoint nor the previously installed `models/pretrained/medsam2/`
checkpoint is overwritten. Large model assets remain excluded from Git by the
existing repository policy.

Input height remains 1024, with aspect ratio preserved. The resized image is
centered on a square canvas; boxes and masks use that same transform. SAM2 then
applies its model transform. Padding is removed before nearest-neighbor restoration
to the original image size. Uint8 intensity is preserved; higher-depth grayscale
images use min/max normalization, matching the reference implementation. The
backend retains its existing RGB-X-ray-to-grayscale conversion.

Masks are predicted independently with logit threshold zero. Bilateral predictions
are combined within each bone's channel. Mask prompts use a 256×256 thresholded
bilinear logit map at −10/+10, plus a bounding box derived from that mask.
There is no Femur-versus-Tibia argmax or confidence competition.

## Mask and training-pool formats

The prompted inference response and corrected training labels use lossless
8-bit RGB PNGs:

| Region | RGB bytes |
| --- | --- |
| Background | `(0,0,0)` |
| Femur | `(1,0,0)` |
| Tibia | `(0,2,0)` |
| Femur + Tibia | `(1,2,0)` |

The UI uses OpenCV's BGR ordering internally, so an overlapping pixel is stored
as `Vec3b(0,2,1)` there. Display intensity is separate: overlap renders yellow.
The full exported annotation can additionally contain Fibula in B=3; the
Femur/Tibia training upload deliberately includes only the two supported channels.
That does not remove Fibula from the document or exported annotation.

The existing pool layout is retained:

```text
data/training/
  metadata/<case_id>.json
  revisions/<case_id>/<revision>/image.png
  revisions/<case_id>/<revision>/mask.png
  revisions/<case_id>/<revision>/metadata.json
  training_state.json                 # UNet consumed revisions
  training_state_medsam2.json          # MedSAM2 consumed revisions
  jobs/<job_id>/config.yaml
  jobs/<job_id>/status.json
  jobs/<job_id>/training.log
```

Original image and mask bytes are stored unchanged. Case identity remains the
source PNG SHA-256. Revisions, file hashes, source model version, original filename,
annotator and notes retain their existing provenance behavior. New records include
`mask_encoding`, `femur_pixels`, `tibia_pixels` and `overlap_pixels`. R must be 0/1,
G must be 0/2 and B must be zero. Malformed annotations are rejected before writing
a revision. Binary training targets are derived independently from these channels.

Validation pairs remain in `data/validation/images` and `data/validation/labels`,
with matching filenames and the existing checks for integrity and train/validation
image duplication. At least one approved training case and one validation case are
required; the application does not generate validation data for real training.

## Fine-tuning and checkpoint lifecycle

MedSAM2 is an adapter inside the existing trainer. It freezes image/memory encoders
and trains the prompt encoder and mask decoder with independent binary
cross-entropy, Dice and IoU-quality losses. Training alternates box prompts and
coarse-mask prompts. Batch size is implemented through gradient accumulation.
The existing epoch count, learning rate, replay factor, seed, optimizer/scheduler,
AMP handling, dataset snapshots, progress callbacks and checkpoint bookkeeping are
reused. The UNet-specific affine/intensity/noise augmentation settings are not
applied by the MedSAM2 adapter. Validation uses reference-derived boxes; its Dice
measures prompted segmentation, not automatic bone localization.

Administrative jobs continue to run on CPU with the configured thread limit,
while the loaded inference model stays available. CLI jobs retain configured
device selection. All families share the existing training/promotion lock.

The model lifecycle is:

```text
models/pretrained/medsam2_updated/checkpoint.pt          reference, preserved
models/medsam2/candidates/medsam2_<uuid>/best.pt          new candidate
models/medsam2/candidates/medsam2_<uuid>/metadata.json
models/medsam2/candidates/medsam2_<uuid>/dataset_snapshot.json
models/medsam2/candidates/medsam2_<uuid>/history.json
models/medsam2/archive/<release>/model.pt                promoted immutable release
models/medsam2/production -> archive/<release>           active disk selection
```

Before the first promotion, inference and fine-tuning start from the reference.
Subsequent jobs start from the selected production checkpoint. Metadata includes
creation time, parent version/path, checkpoint path, training case count, dataset
snapshot/hash, configuration/hash, validation metrics, encoding and environment.
Management reports reference, disk-selected and loaded versions separately,
including active flags. Promotion copies a candidate to an archived release and
atomically switches the production symlink. Restart loads it; old releases remain.
UNet retains its existing `models/candidates`, `models/archive`, and
`models/production` locations.

CLI alternatives, from the backend directory:

```bash
XRAY_PYTHON=/home/suman/deepnet/bin/python ./scripts/finetune.sh \
  --model medsam2 --config configs/management.yaml --dry-run
XRAY_PYTHON=/home/suman/deepnet/bin/python ./scripts/finetune.sh \
  --model medsam2 --config configs/management.yaml --epochs 1
XRAY_PYTHON=/home/suman/deepnet/bin/python ./scripts/promote_candidate.sh \
  medsam2_YOUR_CANDIDATE_ID --model medsam2 --config configs/management.yaml
```

## API examples

Box inference (replace example coordinates with valid coordinates for the image):

```bash
curl --fail-with-body -D response.headers \
  -F image=@xray.png \
  -F 'femur_box=[20,30,220,330]' \
  -F 'tibia_box=[150,280,430,580]' \
  http://127.0.0.1:8000/segment/prompted -o prediction.png
```

Mask refinement:

```bash
curl --fail-with-body -F image=@xray.png -F mask=@corrected-femur-tibia.png \
  http://127.0.0.1:8000/segment/prompted -o refined.png
```

The existing PNG response convention is retained; channels now encode independent
membership. Example headers from the actual synthetic integration test:

```http
HTTP/1.1 200 OK
Content-Type: image/png
X-Model-Version: medsam2_hiera_t_8113ea7212e9
X-Mask-Encoding: rgb_discrete
X-Image-Width: 64
X-Image-Height: 128
X-Femur-Pixels: 287
X-Tibia-Pixels: 287
X-Overlap-Pixels: 287
```

`GET /model/info` includes `prompted_model.checkpoint`, `model_version`,
`mask_encoding`, label definitions and supported prompts.

Submitting an approved correction:

```bash
curl --fail-with-body -F image=@xray.png -F mask=@corrected-femur-tibia.png \
  -F model_version=medsam2_hiera_t_8113ea7212e9 \
  -F original_filename=xray.png http://127.0.0.1:8000/training/cases
```

Example response (the existing `queued` field means awaiting manual training):

```json
{"status":"accepted","case_id":"<image-sha256>","training_status":"queued",
 "training_case_count":1,"new_cases_since_last_training":1,
 "mask_encoding":"rgb_discrete","femur_pixels":453,"tibia_pixels":457,"overlap_pixels":452}
```

Fine-tuning, progress, and promotion through the existing management endpoints:

```bash
curl --fail-with-body 'http://127.0.0.1:8000/management/status?model=medsam2'
curl --fail-with-body -H 'Content-Type: application/json' \
  -d '{"model":"medsam2"}' http://127.0.0.1:8000/training/start
curl --fail-with-body http://127.0.0.1:8000/training/jobs/JOB_ID
curl --fail-with-body -H 'Content-Type: application/json' \
  -d '{"model":"medsam2"}' http://127.0.0.1:8000/models/CANDIDATE_VERSION/promote
```

Start returns HTTP 202 with a job record, for example:

```json
{"id":"<32-character-job-id>","model":"medsam2","status":"queued",
 "candidate_version":null,"epoch":null,"epochs":50,"error":null}
```

Later job records include `status` (`preparing`, `training`, `validating`,
`completed`, or `failed`), `candidate_version`, epoch progress, best Dice,
metrics and bounded recent logs. Promotion returns the selected version,
loaded version and `restart_required`. Omitting `model` still selects UNet.

## Tests and observed results

Inference test with the actual installed weights, isolated from normal data:

```bash
cd /run/media/suman/Data/monai
PYTHONPATH=. /home/suman/deepnet/bin/python scripts/medsam2_e2e.py --inference-only
```

Full fine-tuning test through the real HTTP backend and offscreen Qt controls:

```bash
PYTHONPATH=. /home/suman/deepnet/bin/python scripts/medsam2_e2e.py \
  --orthoseg-build /home/suman/agentic_coding/orthoseg/build
```

This creates an isolated directory under `data/medsam2-e2e/`, uses the actual
supplied checkpoint and a synthetic X-ray phantom, drives inference and canvas
brush edits, exports and uploads the correction, starts training and promotes
through Model Management, restarts the backend, and runs box and mask inference
with the new checkpoint. It verifies the original checkpoint checksum is unchanged.
No test candidate is installed into the normal production directory.

The completed run is in
`data/medsam2-e2e/842b4bb483cd43d996a48a19562aedab/`:

- Reference: `medsam2_hiera_t_8113ea7212e9`; initial overlap: 287 pixels.
- Corrected pool annotation: 452 overlapping pixels.
- Candidate: `medsam2_cafd6e4d58a745849e0b4eaff5d66ad2`.
- After promotion/restart: 801 overlapping pixels with identical box prompts;
  subsequent mask refinement retained 667 overlapping pixels.
- `result.json`, `management.json`, both UI logs, server log, PNGs and checkpoint
  artifacts are retained. This verifies data flow and model execution, not accuracy
  on clinical X-rays.

Regression commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/suman/deepnet/bin/python -m pytest -q
ctest --test-dir /home/suman/agentic_coding/orthoseg/build --output-on-failure
```

The default-device `scripts/smoke_sam2.py` check also passed with the updated reference, returning 162 Femur pixels, 64 Tibia pixels and 31 overlapping pixels.

The backend suite passed 52 tests. OrthoSeg passed all 8 CTest groups. The actual
MedSAM2 job completed and the existing UI management controls trained/promoted it.
UNet regression tests still exercise training, promotion and restarted inference. The existing `scripts/management_smoke.py` also passed against the rebuilt UI: real RGB correction upload, explicit Qt training/promotion, archive preservation and continued inference from the loaded version.

## Compatibility and migration

Existing grayscale 0/1/2 pool samples and validation masks require no migration.
MedSAM2 reads them as two independent binary targets. Newly submitted RGB files
are stored losslessly with new metadata; old immutable revisions remain readable.
Missing new metadata fields in old records are accepted.

`/segment/prompted` now returns RGB instead of grayscale. The updated OrthoSeg
client accepts both formats; external clients that require grayscale must be
updated to decode RGB channels. The automatic UNet and nnUNet responses remain
single-channel. The native OpenCV/ONNX model files are unchanged; this checkpoint
update applies to server MedSAM2.

UNet can read nonoverlapping RGB samples through a lossless conversion. It explicitly
rejects overlapping training or validation annotations and directs the operator to
MedSAM2. Existing shared pools therefore need no conversion, but a UNet job cannot
train on overlapping samples. Overlap already discarded by a historical indexed
mask cannot be recovered; re-import/re-submit the original RGB correction if available.
RGB masks for this two-bone backend must have B=0, so use the UI training-upload
path to exclude Fibula while retaining it in the full exported annotation.

## Modified and added files

Backend paths below are relative to `/run/media/suman/Data/monai`.

| File | Change |
| --- | --- |
| `.gitignore` | Excludes local build/edit staging artifacts. |
| `README.md` | Updates prompted RGB protocol, training and management usage. |
| `MEDSAM_UPDATE_IMPLEMENTATION.md` | This implementation and handover guide. |
| `configs/default.yaml` | Selects updated reference assets; adds a training-family setting. |
| `app/core/config.py` | Validates the supported training family. |
| `app/main.py` | Registers MedSAM2 management using the existing management class; keeps UNet serving independent. |
| `app/api/routes.py` | Adds labeled-mask prompting and RGB/count response headers. |
| `app/api/management.py` | Selects MedSAM2 or UNet for existing status/start/promote operations. |
| `app/management/jobs.py` | Reuses jobs and locks, tracks per-family candidates/state and reference/active metadata. |
| `app/ml/preprocessing.py` | Validates RGB labels, extracts independent binary masks and counts overlap; rejects lossy UNet conversion. |
| `app/ml/sam2_predictor.py` | Matches reference normalization/padding/logits; supports boxes/masks and independent RGB output. |
| `app/ml/sam2_model.py` | Adds shared local model construction and reference/active checkpoint resolution. |
| `app/ml/sam2_training.py` | Adds the independent-mask MedSAM2 adapter inside the existing training infrastructure. |
| `app/ml/trainer.py` | Dispatches to the adapter while retaining epochs, jobs, replay, snapshots and checkpoint bookkeeping. |
| `app/ml/checkpoint.py` | Loads MedSAM2 candidates and uses family-specific storage/promotion; extends the existing CLI. |
| `app/ml/dataset.py` | Accepts channel-mask geometry and prevents overlap loss in UNet datasets. |
| `app/ml/metrics.py` | Computes foreground Dice/IoU independently for each channel. |
| `app/ml/store.py` | Retains RGB bytes, statistics and encoding; separates consumed-revision state by model family. |
| `scripts/smoke_sam2.py` | Updates inference assertions for RGB output. |
| `scripts/medsam2_e2e.py` | Adds actual-checkpoint HTTP/Qt editing/training/promotion/reload integration testing. |
| `tests/test_api.py` | Updates malformed RGB rejection coverage. |
| `tests/test_sam2_api.py` | Updates square-padding, box-coordinate and output expectations. |
| `tests/test_overlap.py` | Tests overlap inference/refinement, strict validation, storage, statistics and legacy formats. |
| `models/pretrained/medsam2_updated/` | Copied reference checkpoint/config/package/code plus provenance manifest; ignored model assets. |

Frontend paths below are relative to `/home/suman/agentic_coding/orthoseg`.

| File | Change |
| --- | --- |
| `include/MonaiClient.h` | Extends existing methods for mask prompts and model-family selection. |
| `src/MonaiClient.cpp` | Validates/encodes RGB masks, uploads mask prompts and selects the management family. |
| `include/ModelManagementDialog.h` | Declares the model-family selector. |
| `src/ModelManagementDialog.cpp` | Adds the selector to the existing status, training and promotion UI. |
| `include/Document.h` | Documents independent anatomy-channel uploads. |
| `src/Document.cpp` | Applies RGB predictions with undo and preserves overlapping anatomy during upload. |
| `include/AIFill.h` | Documents indexed or RGB prediction representations. |
| `src/CanvasWidget.cpp` | Renders overlapping RGB predictions and extracts the selected refinement channel. |
| `src/MainWindow.cpp` | Enables server mask prompts, preserves the unprompted bone, uploads overlap and hooks batch saves. |
| `tests/test_monai.cpp` | Covers RGB transfer and mask prompts; adds real-MedSAM2 UI inference/edit/export mode. |
| `tests/test_management.cpp` | Covers family selection and actual MedSAM2 fine-tuning/promotion mode. |
| `tests/test_document.cpp` | Verifies lossless overlapping anatomy uploads. |
| `tests/test_ui.cpp` | Verifies selected-bone refinement and batch-save upload choices. |
| `tests/live_monai.py` | Updates the original live UNet workflow to validate RGB corrected uploads. |
| `README.md` | Updates AI Fill and corrected-mask workflow usage. |
| `MONAI_INTEGRATION.md` | Updates the mask/API integration contract. |
| `MODEL_MANAGEMENT.md` | Documents the shared UNet/MedSAM2 controls and lifecycle. |

Existing CMake targets already include these sources and tests, so no CMake change
was needed. Earlier unrelated working-tree edits were retained.
