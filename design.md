You are working as a senior medical-imaging / machine-learning engineer.

Implement a COMPLETE, PRODUCTION-QUALITY, FULLY OFFLINE MONAI-based backend for 2D X-ray femur/tibia segmentation.

The backend will be used by an EXISTING custom annotation UI.

DO NOT build any UI.

DO NOT use 3D Slicer.

DO NOT depend on MONAI Label's UI.

Use MONAI/PyTorch for inference and training, with a small FastAPI service exposing the functionality to our custom UI.

The system must work entirely on the local computer with no cloud dependency and no external network/API calls at runtime.

---

# 1. First inspect the repository

Before changing anything:

1. Inspect the complete codebase.
2. Identify:

   * existing Python environment/configuration
   * existing ML/model code
   * existing API/server code
   * existing CUDA/PyTorch setup
   * existing folder layout
   * existing logging/config conventions
   * existing tests
3. Reuse existing architecture where sensible.
4. Do NOT rewrite unrelated functionality.
5. Do NOT break existing features.
6. Keep changes modular so this MONAI service can be removed/replaced later.
7. Show me the files you changed at the end.

Do not merely give me example code. IMPLEMENT the complete feature in the repository.

---

# 2. Purpose

We are segmenting FEMUR and TIBIA from 2D X-ray PNG images.

Input:

```
PNG X-ray image
```

Output:

```
PNG segmentation mask
```

Canonical segmentation values MUST be:

```
0 = background
1 = femur
2 = tibia
```

The returned mask must:

* be single-channel
* be uint8
* contain ONLY values 0, 1, or 2
* have exactly the same width and height as the input PNG

The UI will overlay this mask and let the user manually correct it.

After correction, the UI may send:

```
original X-ray PNG
+
corrected segmentation mask
```

back to this backend to add it to the training dataset.

Training is NOT automatically started when a case is submitted.

Fine-tuning is started manually using a local bash script.

---

# 3. High-level architecture

Implement approximately:

```
xray_monai_backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── health.py
│   │   ├── inference.py
│   │   ├── training_cases.py
│   │   └── model.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── paths.py
│   ├── ml/
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── postprocessing.py
│   │   ├── predictor.py
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   ├── metrics.py
│   │   └── checkpoint.py
│   └── schemas/
│
├── data/
│   ├── training/
│   │   ├── images/
│   │   ├── labels/
│   │   ├── metadata/
│   │   └── revisions/
│   └── validation/
│       ├── images/
│       └── labels/
│
├── models/
│   ├── production/
│   │   ├── model.pt
│   │   └── metadata.json
│   ├── candidates/
│   └── archive/
│
├── configs/
│   └── default.yaml
│
├── scripts/
│   ├── run_server.sh
│   ├── finetune.sh
│   └── promote_candidate.sh
│
├── tests/
├── requirements.txt or pyproject.toml
└── README.md
```

This is a suggested structure. Adapt it to the existing repository rather than unnecessarily duplicating infrastructure.

---

# 4. Technology

Use:

* Python
* PyTorch
* MONAI
* FastAPI
* Uvicorn
* Pillow and/or OpenCV for PNG I/O
* NumPy
* pytest
* CUDA when available

Prefer current MONAI 1.6.x APIs compatible with the repository's installed PyTorch/CUDA environment.

Do NOT blindly reinstall PyTorch if the workstation already has a functioning CUDA-enabled PyTorch environment.

Detect and document the currently installed:

```
Python
PyTorch
CUDA runtime
GPU
MONAI
```

versions.

The application must not download pretrained weights or other resources at runtime.

---

# 5. Model

Implement a configurable 2D segmentation model.

Default model:

```
MONAI UNet
spatial_dims = 2
in_channels = 1
out_channels = 3
```

Make architecture parameters configurable.

Suggested initial configuration:

```
channels:
  - 32
  - 64
  - 128
  - 256
  - 512

strides:
  - 2
  - 2
  - 2
  - 2

num_res_units: 2
```

Do not hard-wire the application so that changing the architecture later is difficult.

We may later replace it with:

* DynUNet
* SegResNet
* transformer model
* custom network

without changing the API exposed to the UI.

---

# 6. PNG handling

Support:

* 8-bit grayscale PNG
* 16-bit grayscale PNG

If an RGB PNG is supplied:

* convert it deterministically to grayscale
* log that conversion

Do NOT destructively convert the stored original X-ray.

Maintain the original image for training.

The preprocessing pipeline used during inference MUST be exactly compatible with the preprocessing used during training.

Implement preprocessing in ONE shared module used by both.

---

# 7. Image preprocessing

Medical X-rays may have different dimensions and intensity ranges.

Implement configurable preprocessing.

At minimum:

1. Decode PNG without unnecessarily reducing bit depth.
2. Convert to a single channel.
3. Convert to float32.
4. Apply configurable robust intensity normalization.
5. Preserve aspect ratio.
6. Prepare image for model inference.
7. Record all resize/padding parameters necessary to map the prediction back to original image coordinates.

Default normalization can use percentile clipping followed by normalization to [0,1], for example configurable lower/upper percentiles.

DO NOT independently normalize training and inference differently.

---

# 8. Spatial preprocessing

The prediction returned to the UI MUST line up pixel-for-pixel with the original image.

Do NOT distort the X-ray by simply forcing arbitrary rectangular images into a square without preserving aspect ratio.

Implement configurable inference strategy.

A good default is:

```
original image
    ↓
aspect-ratio-preserving scaling if necessary
    ↓
padding to model-compatible dimensions
    ↓
network inference
    ↓
remove padding
    ↓
restore original image dimensions
    ↓
nearest-neighbor mask resize
    ↓
original H × W segmentation
```

All transformations and inverse transformations must be tested.

For segmentation masks always use nearest-neighbor interpolation.

NEVER use bilinear/bicubic interpolation for final discrete label masks.

---

# 9. Inference

Implement GPU inference if CUDA is available.

Use:

```
model.eval()
torch.inference_mode()
```

Use mixed precision where safe/configured.

The production model should be loaded once when the server starts, not for every request.

Add a model loading/reloading abstraction.

No concurrent request should observe a partially reloaded model.

---

# 10. Required REST API

The UI integration must use this exact public contract unless the existing repository has a strong reason to wrap it under an existing API prefix.

Default server:

```
http://127.0.0.1:8000
```

Bind to:

```
127.0.0.1
```

by default.

Do NOT expose 0.0.0.0 by default.

---

# 10.1 Health

Implement:

```
GET /health
```

Response:

```
{
  "status": "ok",
  "device": "cuda",
  "gpu": "...",
  "model_loaded": true,
  "model_version": "...",
  "labels": {
    "0": "background",
    "1": "femur",
    "2": "tibia"
  }
}
```

Do not expose sensitive filesystem information unnecessarily.

---

# 10.2 Model information

Implement:

```
GET /model/info
```

Return:

* model version
* architecture
* checkpoint identifier
* input configuration
* label mapping
* whether CUDA is active
* model loaded status
* creation/training timestamp if available

---

# 10.3 Segmentation

Implement:

```
POST /segment
```

Content type:

```
multipart/form-data
```

Required field:

```
image
```

Optional fields:

```
case_id
filename
```

The request contains the ORIGINAL PNG X-ray.

Response MUST normally be:

```
Content-Type: image/png
```

containing the uint8 class-index segmentation mask:

```
0 background
1 femur
2 tibia
```

The response image MUST have exactly the same dimensions as the submitted image.

Include useful response headers such as:

```
X-Model-Version
X-Image-Width
X-Image-Height
X-Inference-Time-Ms
```

Do not return a colored visualization mask as the segmentation result.

Return the actual class-ID mask.

Validate after inference that the output contains only:

```
{0,1,2}
```

If not, fail loudly rather than silently returning corrupt labels.

---

# 10.4 Submit corrected case for training

Implement:

```
POST /training/cases
```

Content type:

```
multipart/form-data
```

Required:

```
image
mask
```

Optional:

```
case_id
original_filename
model_version
annotator
notes
```

The submitted mask represents the FINAL USER-CORRECTED segmentation.

Validate:

* image decodes correctly
* mask decodes correctly
* image and mask dimensions match exactly
* mask is single-channel
* mask contains only 0,1,2
* image is not empty
* femur/tibia mask is not malformed
* file sizes are within configurable limits

On success:

1. Store the original X-ray.
2. Store the corrected mask.
3. Store metadata.
4. Add the case to the approved training pool.
5. DO NOT start training automatically.

Return something like:

```
{
  "status": "accepted",
  "case_id": "...",
  "training_status": "queued",
  "training_case_count": 123,
  "new_cases_since_last_training": 14
}
```

---

# 11. Duplicate and re-submitted cases

Users may correct the same X-ray more than once.

Use SHA-256 of the original image plus a stable case identifier to prevent accidental duplicates.

If the same case is submitted again:

* do NOT create multiple active copies of the same training example
* use the newest approved corrected label as the active training label
* optionally archive the previous label under revisions/
* preserve metadata/revision history

Use atomic file writes.

Never leave half-written image/label pairs if a process is interrupted.

---

# 12. Training status

Implement:

```
GET /training/status
```

Return at minimum:

```
{
  "total_approved_cases": ...,
  "new_cases_since_last_training": ...,
  "last_training_time": ...,
  "production_model_version": ...,
  "latest_candidate_version": ...
}
```

The endpoint is informational only.

Do not initiate training from this endpoint.

---

# 13. Fine-tuning philosophy

IMPORTANT:

DO NOT fine-tune after every newly submitted case.

The UI merely adds approved cases to the training pool.

Fine-tuning is started manually using:

```
./scripts/finetune.sh
```

Fine-tuning must use:

```
historical approved training data
+
newly corrected training data
```

Do NOT fine-tune only on the most recently corrected images.

Implement configurable oversampling/replay so new corrected cases can receive extra weight while old cases remain represented.

For example, configuration could support:

```
new_case_repeat_factor: 3
```

or a sampling ratio.

Avoid catastrophic forgetting.

---

# 14. Fixed validation dataset

The validation set MUST remain separate from adaptive/fine-tuning training cases.

Directory:

```
data/validation/images
data/validation/labels
```

Never automatically move newly submitted UI annotations into the validation set.

Do not modify the validation dataset during fine-tuning.

The candidate model must be evaluated on this fixed validation set.

---

# 15. Loss

For 3-class segmentation implement a sensible segmentation loss.

Preferred default:

```
Dice + Cross Entropy
```

Use MONAI's appropriate loss implementation where possible.

Background handling should be configurable.

The training code must correctly handle:

```
background
femur
tibia
```

as mutually exclusive classes.

---

# 16. Metrics

At minimum calculate:

* mean foreground Dice
* femur Dice
* tibia Dice

Also provide optional/support for:

* IoU
* HD95 if practical

Do not compute validation metrics on augmented labels.

Save validation results with every candidate checkpoint.

---

# 17. Training augmentations

Use sensible 2D X-ray augmentation, configurable and conservative.

Possible augmentations:

* small rotations
* small scaling
* translation
* intensity variation
* contrast variation
* mild noise

Be careful with left/right flipping.

DO NOT enable horizontal flipping by default unless it is explicitly safe for the dataset semantics.

Never introduce transforms that change label class identity.

---

# 18. Checkpoint handling

Use:

```
models/production/model.pt
```

for the currently deployed inference model.

Fine-tuning should create:

```
models/candidates/<version>/best.pt
models/candidates/<version>/metadata.json
models/candidates/<version>/metrics.json
models/candidates/<version>/training_config.yaml
```

Do NOT automatically overwrite:

```
models/production/model.pt
```

at the end of training.

The candidate must remain separate until intentionally promoted.

---

# 19. Starting from existing production model

Fine-tuning should:

1. load the current production checkpoint if available
2. continue training from it
3. preserve optimizer/scheduler state only where appropriate/configured
4. support explicitly training from scratch

Provide CLI options such as:

```
--from-production
--from-checkpoint PATH
--from-scratch
```

Default for finetune.sh:

```
--from-production
```

If no production checkpoint exists, fail with a useful message or require explicit --from-scratch.

Do not silently do something unexpected.

---

# 20. Fine-tuning bash script

Create an executable script:

```
scripts/finetune.sh
```

It must be simple for us to run locally, for example:

```
./scripts/finetune.sh
```

Optional arguments can include:

```
./scripts/finetune.sh --epochs 50
```

or:

```
./scripts/finetune.sh --config configs/default.yaml
```

The script should:

1. activate/use the expected Python environment without making dangerous assumptions
2. verify Python
3. verify CUDA/PyTorch
4. verify MONAI
5. verify production checkpoint
6. validate dataset integrity
7. count training cases
8. ensure image/mask dimensions match
9. ensure masks contain only 0,1,2
10. start fine-tuning
11. evaluate candidate on fixed validation set
12. save best checkpoint
13. save metrics
14. print a clear final summary

Do NOT install packages every time finetune.sh runs.

Do NOT access the internet.

Use:

```
set -euo pipefail
```

and good error handling.

---

# 21. Example terminal output

Make the fine-tuning workflow easy to understand.

Something like:

```
========================================
Femur/Tibia X-ray MONAI Fine-tuning
========================================

Device: NVIDIA ...
PyTorch: ...
MONAI: ...

Production model:
    femur_tibia_v003

Approved training cases:
    527

New cases since last training:
    18

Validation cases:
    75

Starting candidate:
    femur_tibia_v004

...

Best validation results:
    Mean foreground Dice: 0.9824
    Femur Dice:           0.9862
    Tibia Dice:           0.9786

Candidate saved:
    models/candidates/femur_tibia_v004/best.pt

Production model has NOT been replaced.
```

---

# 22. Candidate promotion

Also create:

```
scripts/promote_candidate.sh
```

Usage should be something like:

```
./scripts/promote_candidate.sh femur_tibia_v004
```

The script must:

1. verify candidate exists
2. verify candidate metadata
3. archive current production checkpoint
4. atomically install selected candidate as production
5. preserve version metadata
6. never silently delete old models

The running API can either:

* provide POST /model/reload

or

* document that the server should be restarted

after promotion.

If implementing:

```
POST /model/reload
```

ensure it is local-only and safely swaps the loaded model.

---

# 23. Model versioning

Every model must have an explicit version.

Examples:

```
femur_tibia_2d_v001
femur_tibia_2d_v002
```

Metadata should record:

* architecture
* model version
* parent model version
* creation time
* training case count
* new cases incorporated
* validation metrics
* MONAI version
* PyTorch version
* CUDA version
* configuration hash if practical

The /segment response must identify which model produced the prediction.

---

# 24. Configuration

Provide a YAML or TOML configuration.

Example concepts:

```
server:
  host: 127.0.0.1
  port: 8000

labels:
  background: 0
  femur: 1
  tibia: 2

model:
  input_size: 1024
  architecture: unet

preprocessing:
  lower_percentile: 0.5
  upper_percentile: 99.5

training:
  epochs: 50
  batch_size: 4
  learning_rate: ...
  new_case_repeat_factor: ...
  num_workers: ...
  mixed_precision: true
```

Do not scatter important hyperparameters throughout source files.

---

# 25. Offline requirement

The application must work after internet access is disconnected.

No:

* cloud APIs
* Hugging Face downloads
* WandB network logging
* pretrained-weight downloads
* telemetry
* external databases

If experiment logging is useful, write locally.

TensorBoard may be used locally if desired.

---

# 26. Server startup script

Create:

```
scripts/run_server.sh
```

It should run approximately:

```
uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Use the repository's actual Python environment.

Do not install dependencies every server startup.

---

# 27. CORS

If the existing custom UI needs browser-based access from localhost, configure only the necessary local origins.

For example configurable:

```
http://localhost:...
http://127.0.0.1:...
```

Do NOT default to:

```
Access-Control-Allow-Origin: *
```

unless absolutely required.

---

# 28. Error handling

Return proper HTTP errors for:

* invalid PNG
* image too large
* unsupported image
* malformed mask
* mismatched dimensions
* invalid mask labels
* model not loaded
* CUDA/model inference failure
* disk write failure

Inference failures must never corrupt existing annotation data.

Use structured logging.

---

# 29. Tests

Add meaningful automated tests.

At minimum:

## Inference tests

* valid 8-bit grayscale PNG
* valid 16-bit grayscale PNG
* RGB PNG handling
* returned dimensions exactly equal input dimensions
* output dtype uint8
* output contains only {0,1,2}
* invalid file rejected
* model-not-loaded behavior

## Training-case tests

* valid image/mask accepted
* mismatched dimensions rejected
* mask containing value 3 rejected
* RGB label mask rejected unless deliberately supported
* duplicate submission handled correctly
* replacement/revision behavior works

## Preprocessing tests

* forward/inverse spatial transformation restores original dimensions
* aspect ratio preserved
* mask inverse transform uses nearest neighbor

## Dataset tests

* image and labels remain correctly paired
* validation set never mixed into training set

Do not make tests depend on a large GPU model where unnecessary.

Mock inference where appropriate for API tests.

---

# 30. README

Document:

## Installation

## Environment

## Starting server

```
./scripts/run_server.sh
```

## Health check

```
curl http://127.0.0.1:8000/health
```

## Example inference request

Provide curl example for:

```
POST /segment
```

## Add corrected training case

Provide curl example for:

```
POST /training/cases
```

## Fine-tune

```
./scripts/finetune.sh
```

## Candidate promotion

```
./scripts/promote_candidate.sh <candidate>
```

## Restart/reload model

## Dataset structure

## Label values

```
0 background
1 femur
2 tibia
```

## Offline operation

---

# 31. API contract that the UI team will rely on

DO NOT casually change this contract:

### Health

```
GET /health
```

### Model info

```
GET /model/info
```

### Segment

```
POST /segment
```

multipart:

```
image=<PNG>
case_id=<optional>
```

response:

```
image/png
```

pixel IDs:

```
0 background
1 femur
2 tibia
```

response dimensions:

```
EXACTLY same as input
```

header:

```
X-Model-Version
```

### Submit corrected annotation

```
POST /training/cases
```

multipart:

```
image=<original PNG>
mask=<corrected class-ID PNG>
case_id=<optional>
original_filename=<optional>
model_version=<optional>
```

response JSON with:

```
status
case_id
training_status
training_case_count
new_cases_since_last_training
```

### Training status

```
GET /training/status
```

---

# 32. Important safety/data-integrity requirements

This is medical-image annotation infrastructure.

Prioritize:

* deterministic data handling
* geometric correctness
* no silent resizing mismatch
* no accidental label remapping
* reproducibility
* traceability
* explicit model versions
* atomic writes
* dataset integrity

Never silently "fix" an invalid training mask.

Reject it and report the exact problem.

---

# 33. Deliverables

Do not stop at architecture recommendations.

Implement everything needed.

At completion:

1. Run unit tests.
2. Run lint/type checks if the repository uses them.
3. Verify server starts.
4. Verify /health.
5. Test /segment using a small PNG.
6. Verify returned mask dimensions.
7. Test submission of a corrected mask.
8. Test dataset validation.
9. Test fine-tuning script at least in dry-run/smoke-test mode.
10. Give me:

    * files added
    * files modified
    * commands to install dependencies
    * command to run server
    * command to fine-tune
    * command to promote candidate
    * any assumptions made

Do not modify unrelated features.
Do not remove existing code unless required.
Do not leave TODO placeholders for core functionality.

