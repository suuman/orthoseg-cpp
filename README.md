# Offline femur/tibia X-ray backend

Local FastAPI service for an existing annotation UI. No UI, cloud services, telemetry, pretrained downloads, or automatic training. PNG mask IDs are **0 background, 1 femur, 2 tibia**. Output is single-channel uint8 at exactly the original image dimensions.

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
./scripts/run_server.sh
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

The public routes are `/health`, `/model/info`, `/segment`, `/training/cases`, and `/training/status`. `/segment` also accepts optional `filename`; training submission accepts optional `case_id`, `original_filename`, `model_version`, `annotator`, and `notes`. The machine-readable schema is `/openapi.json`; external-CDN documentation pages are disabled for offline operation.

Use `--config /path/to/override.yaml` or `XRAY_CONFIG` to override selected default settings. Bind defaults to `127.0.0.1:8000`. CORS defaults to no permitted cross-origin callers; set `server.cors_origins` to your exact local UI origin(s). This is a trusted local workstation service without authentication; changing its bind address changes that operating assumption.

## Images, geometry and labels

8-bit and 16-bit grayscale X-rays retain their original stored bytes. RGB/RGBA images are deterministically converted to grayscale for processing and the conversion is logged. Alpha is ignored. Palette, binary, grayscale-alpha and animated images are rejected. Label masks must be 8-bit grayscale with only 0, 1, 2; RGB/palette/16-bit labels are rejected without conversion. Background-only labels are accepted because the service cannot infer anatomical correctness from pixels alone. Anatomical approval belongs to the annotator.

A single preprocessing module handles both training and inference: float32 conversion, percentile clipping, [0,1] scaling, aspect-preserving downscaling, and bottom/right zero padding to UNet stride compatibility. Constant images normalize to zero. Maximum side length defaults to 1024; images are never upscaled before inference. Discrete labels use nearest-neighbor interpolation in both directions. Validation metrics are computed against unaugmented original-resolution labels after inverse mapping. Limits default to 32 MiB per file and 25 million decoded pixels, with an aggregate streamed request limit as well.

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

No production checkpoint means default fine-tuning fails with an actionable message. `--dry-run` checks the environment, parent checkpoint and complete dataset without training. Custom checkpoints must use this backend's state-dict/metadata format with the canonical labels. Model/preprocessing configuration must match the parent. The model factory isolates the architecture; a future architecture can be implemented there without changing the HTTP contract.

Training includes every active historical example and repeats new/revised examples according to `new_case_repeat_factor`. Data snapshots reference immutable revisions, so concurrent annotation updates remain pending for the next training run. Only one trainer runs per data root. The fixed validation set selects the best epoch using dataset-global mean foreground Dice; femur/tibia Dice and IoU are also recorded. Classes empty in both reference and prediction have `null` scores and are excluded from the mean. All-undefined validation fails. HD95 is not implemented.

Defaults use MONAI Dice+cross-entropy loss, AdamW, cosine learning-rate scheduling, deterministic seeding, and conservative paired affine/intensity/noise augmentations. Horizontal flipping is disabled. Variable shapes are padded within each batch. CUDA and mixed precision are used only when available/configured; CPU works without AMP. Optimizer/scheduler restoration is opt-in via `training.resume_optimizer`. This resumes optimization state, not the exact random-number sequence of an interrupted run. Seeds and snapshots aid reproducibility; cross-platform bitwise reproducibility is not guaranteed.

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
| `/management/status` | GET | Production on disk, loaded version, dataset counts, validated latest completed candidate, latest job |
| `/training/start` | POST `{}` | Start one asynchronous fine-tuning job from production with backend defaults |
| `/training/jobs/{id}` | GET | Job state, epoch, best Dice, error and bounded recent log |
| `/models/{version}/promote` | POST `{}` | Deliberately promote the valid latest completed candidate |

The worker calls the **existing trainer**, and promotion calls the **existing
checkpoint promotion implementation**. Administrative jobs run on CPU with bounded
threads, keeping the production inference model and GPU allocation untouched.
Training hyperparameters, validation, replay and checkpoint semantics come from
the backend configuration. The CLI retains its existing CUDA/CPU device selection;
use the CLI for GPU training with appropriate workstation resource planning.

The API reserves the same filesystem lock used by CLI training before launching a
worker, then passes the locked descriptor to that worker. Duplicate API/CLI runs
and promotion during training return HTTP 409. Missing bootstrap weights return
422; use `finetune.sh --from-scratch` explicitly first. Bad datasets or training
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
