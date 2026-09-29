# Offline femur/tibia X-ray backend

Local FastAPI service for OrthoSeg. Automatic UNet/nnUNet routes return single-channel uint8 IDs **0 background, 1 femur, 2 tibia**. Prompted MedSAM2 returns independent discrete RGB channels **R=0/1, G=0/2, B=0**, including valid Femur/Tibia overlap `(1,2,0)`. Every prediction uses the original image dimensions. Training runs only on explicit request.

## Local prompted MedSAM2 model

The updated checkpoint, model configuration, Python package and reference inference code from `/run/media/suman/Data/medsam2` are installed under `models/pretrained/medsam2_updated/`. The original source checkpoint and the older installed assets remain unchanged. The initial version is `medsam2_hiera_t_8113ea7212e9`.

`/segment/prompted` accepts Femur/Tibia boxes, including two boxes per bone for bilateral anatomy, or a multipart `mask` containing grayscale 0/1/2 or discrete RGB labels. OrthoSeg exposes Bounding Box, Paint Mask, Load Mask and Normal Fill Mask prompts for **MONAI MedSAM2 (boxes / masks)**. Pixels are predicted independently for each bone; overlap is never resolved by a winner.

Images are resized to **1024 pixels high**, preserving aspect ratio, and centered on a square canvas before SAM2's transform. Box coordinates and mask prompts use the same geometry; masks are unpadded and restored with nearest-neighbor interpolation. As in the reference implementation, uint8 intensity is preserved and higher-depth images use min/max normalization. Mask refinement uses thresholded bilinear 256×256 logits of −10/+10 plus a mask-derived box. Check `/health` and `/model/info` for the loaded version, checkpoint, encoding and supported prompts.

MedSAM2 uses the existing corrected-annotation pool, background job runner and model-management dialog. Choose **MedSAM2** in that dialog to train a new candidate, inspect progress and metrics, then promote it explicitly. Restart the backend to activate the promoted version. See [MEDSAM_UPDATE_IMPLEMENTATION.md](MEDSAM_UPDATE_IMPLEMENTATION.md) for the complete workflow, API examples, file list and tests.

The default `.venv` does not contain all SAM2 dependencies. On this workstation, `/home/suman/deepnet/bin/python` has the required packages; `scripts/run_server_sam2.sh` selects it by default, or set `XRAY_PYTHON` to another compatible environment. Install `requirements.txt` and `requirements-sam2.txt` into that environment if needed. Startup is local and downloads nothing. Verify the checkpoint and API together with:

```bash
PYTHONPATH=. XRAY_CONFIG=configs/management.yaml /home/suman/deepnet/bin/python scripts/smoke_sam2.py
./scripts/run_server_sam2.sh --config configs/management.yaml
curl --fail-with-body -F image=@xray.png -F 'femur_box=[20,30,220,330]' \
  -F 'tibia_box=[150,280,430,580]' http://127.0.0.1:8000/segment/prompted -o mask.png
```

Box coordinates must lie within the original image and have positive area. Supply at least one bone box or a nonempty labeled mask. Each bone accepts up to two boxes. The automatic `/segment` route still requires a separately trained UNet checkpoint; model families report readiness independently.

## Local nnUNet v2 automatic model

The supplied 2D nnUNet v2 `checkpoint_best.pth`, `plans.json`, and `dataset.json` from `/run/media/suman/Data/nnunet2` are installed under `models/pretrained/nnunet2/`. Its one input channel and label IDs match this service: 0 background, 1 femur, 2 tibia. Its `PlainConvUNet` architecture and nnUNet preprocessing do **not** match this backend's MONAI UNet production checkpoint format, so it runs through a separate automatic `/segment/nnunet` route and model choice. It does not replace `/segment` or participate in the UNet training/promotion workflow.

Input PNGs are converted to the model's uint8 grayscale convention, resized to **2048 pixels high** while preserving aspect ratio, and inferred with nnUNet's 2D sliding window. The returned 0/1/2 PNG is resized back to the original image dimensions with nearest-neighbor interpolation. Check `nnunet_model_loaded` and `nnunet_model_version` in `/health`, or `nnunet_model` in `/model/info`. The local `/home/suman/deepnet/bin/python` environment has nnUNet v2; `scripts/run_server_models.sh` selects it by default. For another Python environment, install `requirements.txt`, `requirements-sam2.txt`, and `requirements-nnunet.txt` as needed and set `XRAY_PYTHON`.

```bash
PYTHONPATH=. XRAY_CONFIG=configs/management.yaml /home/suman/deepnet/bin/python scripts/smoke_nnunet.py
./scripts/run_server_models.sh --config configs/management.yaml
curl --fail-with-body -F image=@xray.png \
  http://127.0.0.1:8000/segment/nnunet -o mask.png
```

The supplied training metadata describes an X-ray dataset, but label names alone do not establish accuracy for a new image source; review each predicted mask before use.

## Installation and environment

Linux is required for the filesystem locks and atomic symlink promotion. Use a local filesystem supporting `flock`, `fsync` and atomic rename. Paths in YAML resolve relative to this repository; absolute paths are accepted.

Initial inspection found Python 3.14.4, no installed PyTorch/MONAI or existing application, and an NVIDIA driver that could not initialize. A project `.venv` was created for verification with Python 3.14.4, PyTorch 2.14.0+cpu, MONAI 1.6.0, no CUDA runtime and no available GPU. GPU execution remains unverified on this workstation.

If a functioning CUDA environment already exists, activate it and install the requirements there; do not replace its PyTorch build. Otherwise, for CPU:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

Installation needs packages available locally or network access. For a disconnected machine, prepare a compatible wheelhouse on a machine with the same Python/platform, transfer it, then install using `pip install --no-index --find-links /path/to/wheels -r requirements.txt`. CUDA wheels must match the target hardware/driver. Runtime and all scripts perform no installation or network downloads.

Scripts select `XRAY_PYTHON`, then the active `VIRTUAL_ENV`, then `.venv/bin/python`, then `python3`. They set `PYTHONPATH` to this repository to avoid unrelated workstation packages being injected. To inspect the selected environment:

```bash
PYTHONPATH=. .venv/bin/python -c 'from app.ml.trainer import environment; print(environment())'
```

## Start and integrate

```bash
./scripts/run_server_models.sh --config configs/management.yaml
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info
curl http://127.0.0.1:8000/training/status
curl --fail-with-body -D response-headers.txt \
  -F image=@xray.png -F case_id=case001 \
  http://127.0.0.1:8000/segment -o mask.png
curl --fail-with-body \
  -F image=@xray.png -F mask=@corrected-mask.png \
  -F case_id=case001 -F original_filename=xray.png \
  -F model_version=your_model_version -F annotator=local_user \
  -F notes='Approved correction' \
  http://127.0.0.1:8000/training/cases
```

Inference returns `image/png` plus `X-Model-Version`, `X-Image-Width`, `X-Image-Height`, and `X-Inference-Time-Ms`. Missing production weights produce HTTP 503; the service still accepts approved cases. Malformed PNG/mask inputs return 422, limits return 413, inference failures return 500, and persistence failures return 507. Health reports service availability; inspect `model_loaded` for inference readiness.

The public routes are `/health`, `/model/info`, `/segment`, `/segment/nnunet`, `/segment/prompted`, `/training/cases`, and `/training/status`. `/segment` also accepts optional `filename`; training submission accepts optional `case_id`, `original_filename`, `model_version`, `annotator`, and `notes`. The machine-readable schema is `/openapi.json`; external-CDN documentation pages are disabled for offline operation.

Use `--config /path/to/override.yaml` or `XRAY_CONFIG` to override selected default settings. Bind defaults to `127.0.0.1:8000`. CORS defaults to no permitted cross-origin callers; set `server.cors_origins` to your exact local UI origin(s). This is a trusted local workstation service without authentication; changing its bind address changes that operating assumption.

## Images, geometry and labels

8-bit and 16-bit grayscale X-rays retain their original stored bytes. RGB/RGBA images are deterministically converted to grayscale for processing and the conversion is logged. Alpha is ignored. Palette, binary, grayscale-alpha and animated images are rejected. Label masks accept 8-bit grayscale 0/1/2 or discrete RGB R=0/1, G=0/2, B=0. Palette, 16-bit and malformed RGB labels are rejected. The pool stores the original mask bytes, channel encoding and Femur/Tibia/overlap pixel counts. UNet accepts RGB only when it can convert without losing overlap; overlapping samples require MedSAM2 training. Background-only labels are accepted because the service cannot infer anatomical correctness from pixels alone. Anatomical approval belongs to the annotator.

For the automatic UNet route, a single preprocessing module handles both training and inference: float32 conversion, percentile clipping, [0,1] scaling, aspect-preserving downscaling, and bottom/right zero padding to UNet stride compatibility. Constant images normalize to zero. Maximum side length defaults to 1024; images are never upscaled before automatic inference. Discrete labels use nearest-neighbor interpolation in both directions. Validation metrics are computed against unaugmented original-resolution labels after inverse mapping. Limits default to 32 MiB per file and 25 million decoded pixels, with an aggregate streamed request limit as well. Prompted SAM2 uses the separate 1024-high path described above.

## Dataset and revisions

```text
data/training/
  metadata/<case_id>.json                     # atomic active-case manifest
  revisions/<case_id>/<revision>/image.png    # original bytes, immutable
  revisions/<case_id>/<revision>/mask.png     # approved corrected labels
  revisions/<case_id>/<revision>/metadata.json
  training_state.json                        # incorporated revision IDs
  images/ and labels/                        # reserved; not scanned for training
data/validation/images/<name>.png
data/validation/labels/<name>.png
```

Training reads the active manifests, not loose directory files. Submit historical approved cases through `/training/cases` as well. Exact original-image SHA-256 deduplicates even submissions using a different identifier; the existing canonical identifier is returned. Reusing an identifier for a different original image is rejected. Revised labels produce a new immutable revision and atomically replace one manifest. Previous revisions remain available. A crash before the manifest commit may leave an unreferenced revision but never an active half-written pair. Back up the full data directory, including manifests and revisions.

Validation PNG filenames must match exactly between `images/` and `labels/`. Training checks PNG validity, dimensions, canonical labels, training hashes, filename pairing, and exact-image overlap between training and validation. PNGs re-encoded from the same image and multiple images of one patient cannot be detected as patient-level leakage; curate the fixed validation set accordingly. No workflow modifies validation files.

## Fine-tuning

Place curated validation pairs in the directories above and submit historical/new cases first. A new installation has **no trained weights**. Bootstrap explicitly:

```bash
./scripts/finetune.sh --from-scratch --dry-run
./scripts/finetune.sh --from-scratch --epochs 50
```

Subsequent runs default to the deployed checkpoint:

```bash
./scripts/finetune.sh --dry-run
./scripts/finetune.sh --epochs 50
./scripts/finetune.sh --from-checkpoint /absolute/path/best.pt
./scripts/finetune.sh --config configs/default.yaml
```

For UNet, no production checkpoint means default fine-tuning fails with an actionable message. `--dry-run` checks the environment, parent checkpoint and complete dataset without training. Custom checkpoints must use this backend's state-dict/metadata format with the canonical labels. Model/preprocessing configuration must match the parent. The model factory isolates the architecture; a future architecture can be implemented there without changing the HTTP contract.

Training includes every active historical example and repeats new/revised examples according to `new_case_repeat_factor`. Data snapshots reference immutable revisions, so concurrent annotation updates remain pending for the next training run. Only one trainer runs per data root. The fixed validation set selects the best epoch using dataset-global mean foreground Dice; femur/tibia Dice and IoU are also recorded. Classes empty in both reference and prediction have `null` scores and are excluded from the mean. All-undefined validation fails. HD95 is not implemented.

UNet defaults use MONAI Dice+cross-entropy loss, AdamW, cosine learning-rate scheduling, deterministic seeding, and conservative paired affine/intensity/noise augmentations. Horizontal flipping is disabled. Variable shapes are padded within each batch. CUDA and mixed precision are used only when available/configured; CPU works without AMP. Optimizer/scheduler restoration is opt-in via `training.resume_optimizer`. This resumes optimization state, not the exact random-number sequence of an interrupted run. Seeds and snapshots aid reproducibility; cross-platform bitwise reproducibility is not guaranteed.

Each run creates:

```text
models/candidates/<version>/
  best.pt
  metadata.json
  metrics.json
  training_config.yaml
  dataset_snapshot.json
  history.json
```

Metadata records parent/version, creation time, configuration hash, environment, incorporated case counts and best validation metrics. Candidate creation never overwrites production. Training status counts new **or revised** cases since the most recently completed training run, regardless of whether its candidate was promoted. Failed/interrupted runs do not mark cases incorporated.

## Promotion and restart

```bash
./scripts/promote_candidate.sh <candidate-version>
# Stop the running server (Ctrl-C), then:
./scripts/run_server.sh
```

Promotion validates weights, labels, metadata and metrics, copies the selected model into an immutable `models/archive/<release>/` directory, and atomically swaps the `models/production` symlink. Thus `models/production/model.pt` and `metadata.json` always belong to the same release. Old releases and candidates are retained. A legacy non-symlink production directory must first be moved into archive manually. Re-promoting an older candidate supports rollback. Only trusted local checkpoints should be installed; weights are loaded with PyTorch `weights_only=True`.

The server loads one production model at startup and serializes prediction/model publication. Restart after promotion; there is intentionally no HTTP reload endpoint. Running requests keep using the previously loaded model/version until restart.

## Verification

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
./scripts/smoke_test.sh
```

Tests cover PNG formats, geometric restoration, API contract, rejection paths, revisions, simulated interrupted writes, dataset leakage, a real tiny MONAI training/inference run, promotion and fine-tuning from production. The smoke script uses temporary synthetic data, runs the actual shell scripts, starts a local HTTP server, exercises segmentation and case submission, and checks returned dimensions. Synthetic weights are never installed into the project's production directory. No lint/type tooling existed in the original repository.

Implementation API reference: [MONAI 1.6 DiceCELoss](https://monai.readthedocs.io/en/1.6.0/losses.html). This reference is for development; runtime does not access it.

## Optional local model management

Administrative model management is disabled by default. Enable it deliberately:

```bash
./scripts/run_server.sh --config configs/management.yaml
```

For an existing custom configuration, add `management.enabled: true` to that
configuration instead. `management.cpu_threads` defaults to 2. The management
API accepts only direct loopback clients with a local Host header and no browser
Origin header. It provides explicit operations, accepts no commands, checkpoint
paths or ML parameter overrides, and does not change the annotation API.
This is a trusted local workstation control, not a multi-user authentication
system. Do not proxy administrative routes to remote callers.

| New route | Method | Purpose |
| --- | --- | --- |
| `/management/status` | GET, optional `?model=medsam2` | Production on disk, loaded version, dataset counts, validated latest completed candidate, latest job |
| `/training/start` | POST `{}` or `{"model":"medsam2"}` | Start one asynchronous fine-tuning job from production with backend defaults |
| `/training/jobs/{id}` | GET | Job state, epoch, best Dice, error and bounded recent log |
| `/models/{version}/promote` | POST `{}` or `{"model":"medsam2"}` | Deliberately promote the valid latest completed candidate |

The worker calls the **existing trainer**, and promotion calls the **existing
checkpoint promotion implementation**. Administrative jobs run on CPU with bounded
threads, keeping the production inference model and GPU allocation untouched.
Training hyperparameters, validation, replay and checkpoint semantics come from
the backend configuration. The CLI retains its existing CUDA/CPU device selection;
use the CLI for GPU training with appropriate workstation resource planning.

The API reserves the same filesystem lock used by CLI training before launching a
worker, then passes the locked descriptor to that worker. Duplicate API/CLI runs
and promotion during training return HTTP 409. Missing bootstrap weights return
422; bootstrap UNet with `finetune.sh --from-scratch`, or install the supplied MedSAM2 reference assets. CLI MedSAM2 training uses `finetune.sh --model medsam2`; CLI promotion accepts `--model medsam2` as well. Bad datasets or training
failures become failed jobs and never promote a model.

Job snapshots/status/full logs live locally under `data/training/jobs/<id>/`.
States are `queued`, `preparing`, `training`, `validating`, `completed`, `failed`.
The response contains at most 80 recent lines/16 KiB of log data. Workers survive
UI closure and backend restart; persisted process identity detects an interrupted
worker on the next status query. Jobs are never automatically retried. There is
no cancellation API. A crashed worker releases its OS lock and is reported failed.

Promotion still preserves old releases and requires a backend restart. Management
status explicitly distinguishes `production.version` on disk from
`loaded_model_version` and reports `restart_required`. No hot reload was added.
The normal `/model/info` and `/segment` continue to report the loaded model.

In OrthoSeg, launch with `ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1` to expose
**Tools → AI Model Management**. The modeless dialog requires confirmation for
training and promotion. It displays production/candidate Dice side by side,
reports missing metadata as N/A, and polls at five-second intervals only while
training is active. Normal Save/Export/Add to AI Training never starts a job.

Run the optional real Qt-to-backend smoke test using temporary synthetic data:

```bash
PYTHONPATH=. .venv/bin/python scripts/management_smoke.py \
  --ui-test /home/suman/agentic_coding/orthoseg/build/test_management \
  --annotation-test /home/suman/agentic_coding/orthoseg/build/test_monai
```

See [ADDONS_IMPLEMENTATION.md](ADDONS_IMPLEMENTATION.md) for the change inventory
and validation record.

### Corrected-case reminders and manual fine-tuning

Open OrthoSeg **Model Management**, select **UNet** or **MedSAM2**, and set
**Remind me to fine-tune after corrected cases**. Zero disables reminders.
The saved threshold counts distinct new/revised cases accepted through **Add to
AI Training**, not brush strokes or exports. After an accepted upload reaches
that model's threshold, the app offers to open Model Management. Declining does
not prompt again until the next threshold milestone or a completed training run.
Training starts only when the user selects **Fine-tune New Candidate** and
confirms. Promotion remains a separate action and requires a backend restart.

Enable the backend management API with `--config configs/management.yaml`.
The toolbar's Model Management button appears when the backend grants access;
`ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1` also exposes the Tools menu entry.
A fixed validation dataset and a parent model are required. UNet bootstrap from
scratch remains a CLI operation. If UI management is unavailable, use
`/run/media/suman/Data/monai/scripts/finetune.sh`:

```bash
# UNet fine-tuning from production; add --from-scratch for initial training.
/run/media/suman/Data/monai/scripts/finetune.sh --model unet --dry-run
/run/media/suman/Data/monai/scripts/finetune.sh --model unet --epochs 50

# MedSAM2 fine-tuning using the installed compatible Python environment.
XRAY_PYTHON=/home/suman/deepnet/bin/python /run/media/suman/Data/monai/scripts/finetune.sh --model medsam2 --dry-run
XRAY_PYTHON=/home/suman/deepnet/bin/python /run/media/suman/Data/monai/scripts/finetune.sh --model medsam2 --epochs 50
```

Promotion now requires the matching `completed.json` marker as well as valid
weights, metadata and metrics. Interrupted candidates cannot be promoted through
the CLI. Health/status metadata remains readable while inference is running.

Model Management also provides `GET /management/cases?model=unet&offset=0&limit=50`
(or `model=medsam2`). It uses the same local-only management permission check.
The paginated response contains filenames, actual labels, overlap pixels, and
pending/incorporated status for each active corrected-case revision. Maximum
page size is 100. OrthoSeg displays these cases in its model-management dialog.
