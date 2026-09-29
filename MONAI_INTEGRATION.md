# MONAI integration record

## Actual API and startup

Backend startup with local SAM2 and nnUNet v2, from its repository:
`./scripts/run_server_models.sh --config configs/management.yaml`. Default bind:
`http://127.0.0.1:8000`.

| Route | Method | Contract |
| --- | --- | --- |
| `/health` | GET | JSON including separate production, SAM2, and nnUNet readiness/version fields |
| `/model/info` | GET | Production, prompted SAM2, and nnUNet model information |
| `/segment` | POST | Multipart `image`, optional `case_id`, `filename`; returns `image/png` |
| `/segment/nnunet` | POST | Multipart original PNG; nnUNet v2 automatic prediction, source-sized 0/1/2 PNG |
| `/segment/prompted` | POST | Multipart original PNG and Femur/Tibia boxes or labeled mask; source-sized discrete RGB PNG |
| `/training/cases` | POST | Multipart `image`, `mask`, `case_id`, `original_filename`, `model_version` |
| `/training/status` | GET | Informational counts; never initiates training |

The backend also accepts optional `annotator` and `notes`; the UI does not invent
these values. Errors are JSON with `detail` (a string or validation-error array).
Model provenance arrives in `X-Model-Version`; its absence is allowed.
The nnUNet v2 checkpoint has the correct labels but an incompatible architecture
for the MONAI production UNet. It has its own route and AI Fill model choice;
model management supports the MONAI UNet and prompted MedSAM2 families separately.

## Minimal changes

Modified OrthoSeg files:

- `CMakeLists.txt`: Qt Network, isolated client library and one test target.
- `include/Document.h`, `src/Document.cpp`: retain original PNG bytes/source path;
  apply a validated anatomy mask with one existing undo snapshot.
- `include/MainWindow.h`, `src/MainWindow.cpp`: one AI Segment button, request
  callbacks, session model/revision tracking and a successful-Export prompt.
- `README.md`: usage/build instructions.

Added files:

- `include/MonaiClient.h`, `src/MonaiClient.cpp`.
- `tests/test_monai.cpp`, `tests/live_monai.py`.
- `MONAI_INTEGRATION.md`.

`replaceAnatomyMask()` updates Femur and Tibia channels in one undo step while
preserving the independent Fibula channel, including overlaps. The desktop
app also imports complete annotations through **Import Mask**; **Load Prompt
Mask** remains a separate AI Fill prompt action.

Successful Export Mask and batch Save Mask & Next offer the training upload before advancing to another case. Export writes exact RGB
label channel values (R=1 Femur, G=2 Tibia, B=3 Fibula) to lossless PNG.
This checkout has no `ocv/`
directory: actual MedSAM2 code is in `src/medsam2_ocv.cpp`, `src/MedSAM2Inference.cpp`
and related headers/controller files. MONAI MedSAM2 inference and training uploads preserve Femur/Tibia overlap as RGB(1,2,0). The native ONNX models remain separate from the updated server checkpoint.

## Images and labels

The exact original PNG bytes are retained at successful image load. HTTP uploads
use those bytes rather than `sourceGray()`, `sourceColor()` or the canvas. This
preserves 16-bit data and still works if the original disk file is later removed.
UI rendering remains its existing 8-bit path.

Response validation checks HTTP status/content type, PNG signature/IHDR, 8-bit
grayscale 0/1/2 or discrete RGB encoding, exact dimensions, successful decoding and valid channel values. No resizing or color guessing occurs. Complete validation happens before
any annotation mutation. Malformed RGB, palette, 16-bit labels, unexpected values and
mismatched dimensions fail without changing the annotation.

Semantic mapping resolves the existing label names:

| Canonical MONAI | OrthoSeg label | Current ID |
| --- | --- | --- |
| 0 | Background | 0 |
| 1 | Femur | 1 |
| 2 | Tibia | 2 |

Mapping tests also use Femur=5 and Tibia=8. Missing/invalid mappings are rejected.
Training reverse-maps the **current anatomy channels**, including brush/erase/fill
and applied AI Fill corrections. Fibula maps to training background unless a
Femur or Tibia channel is also present at that pixel. The normal document and
lossless RGB export retain every channel.

SHA-256 of the original PNG is the stable `case_id`. A hash of the canonical
current mask suppresses repeat prompts/uploads for unchanged annotations in the
same process. Revised masks retain the case ID, so the backend replaces one
active case and preserves revisions. The prediction's model version remains
attached to the case for corrected-data submission. Saving/submission never starts training. The separate Model Management dialog can explicitly launch UNet or MedSAM2 fine-tuning.

## Verification commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/orthoseg
```

Existing `ai`, `segmentation`, `document`, and `ui` tests are retained. The new
`monai` test uses a local HTTP test server and the real Qt dialogs/document to
cover health/offline/HTTP/timeout behavior, PNG validation, model version,
noncanonical label mapping, original 16-bit bytes, replacement/undo, preserved
Fibula, current edited training masks, Save Only, unchanged/revised annotations,
upload retry, failed exports and stale/invalid inference results.

Optional test against the actual backend, run with that backend's Python:

```bash
/path/to/backend/.venv/bin/python tests/live_monai.py \
  --backend /path/to/backend --binary ./build/test_monai
```

This starts the real backend with its `run_server.sh`, uses a temporary tiny
synthetic MONAI checkpoint and a 16-bit PNG, drives the actual OrthoSeg upload,
AI Segment, brush/erase, AI-result apply and Export dialogs, and verifies the
backend's stored original bytes and corrected mask against the UI's current
annotation. It checks that no training was started. No synthetic model/data is
installed into production. This is integration verification, not a claim of
anatomical model quality. GPU MedSAM2 inference requires a working NVIDIA driver
and remains a hardware-dependent manual check; existing AI Fill regression and
apply/undo tests exercise the unchanged paths without GPU inference.

The URL default exists only in `MonaiClient::configuredUrl()` and can be overridden
with `MONAI_BACKEND_URL`. Runtime UI code has no dependency on the backend's
filesystem location. Qt HTTP API reference: [QNetworkAccessManager](https://doc.qt.io/qt-6/qnetworkaccessmanager.html).

## Updated MedSAM2 workflow

The backend's `MEDSAM_UPDATE_IMPLEMENTATION.md` documents the supplied checkpoint,
full test and API examples. Choose MedSAM2 in Model Management to view its reference,
active version, corrected-case counts and candidate. Both families reuse the same
job APIs; MedSAM2 requests include `model=medsam2`. The original source checkpoint
is preserved and candidates are activated only by explicit promotion and backend
restart. Existing grayscale pool samples need no migration; overlap already lost
in an old indexed file cannot be reconstructed automatically.

## Annotation workflow corrections

Changing the selected anatomy clears a painted/loaded mask prompt to prevent
relabelling another bone's prompt. Bone-specific bounding boxes are retained.
Both ordinary and batch export require applying or clearing an AI preview.
Editing a MONAI preview directly on the canvas records its model version for
subsequent corrected-case uploads, just like the explicit Apply button.
Local MedSAM2 refinement replaces the prompted bone channels, including an
empty result, while retaining other bones and supporting Undo. Its ONNX
competitive-argmax output remains unchanged.
