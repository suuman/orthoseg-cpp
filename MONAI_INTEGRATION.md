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
| `/segment/prompted` | POST | Multipart original PNG and one or two Femur/Tibia boxes; source-sized 0/1/2 PNG |
| `/training/cases` | POST | Multipart `image`, `mask`, `case_id`, `original_filename`, `model_version` |
| `/training/status` | GET | Informational counts; never initiates training |

The backend also accepts optional `annotator` and `notes`; the UI does not invent
these values. Errors are JSON with `detail` (a string or validation-error array).
Model provenance arrives in `X-Model-Version`; its absence is allowed.
The nnUNet v2 checkpoint has the correct labels but an incompatible architecture
for the MONAI production UNet. It has its own route and AI Fill model choice;
model management remains tied to the MONAI production UNet.

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

The document's existing `applyAIResult()` only merges foreground and therefore
cannot remove old Femur/Tibia pixels. A small `replaceAnatomyMask()` method uses
the same editable mask and undo stack, replaces the two relevant classes, and
preserves all unrelated label pixels, including overlapping Fibula. No alternate
annotation layer was added. The existing mask import loads an AI Fill prompt,
not a complete multi-class annotation; that path is unchanged.

This checkout has Export Mask, not a separate annotation Save action, so only
the successful Export path gets the training hook. Its colored PNG encoding,
file dialog and failure semantics remain unchanged. This checkout has no `ocv/`
directory: actual MedSAM2 code is in `src/medsam2_ocv.cpp`, `src/MedSAM2Inference.cpp`
and related headers/controller files, all unchanged. Rendering, shortcuts,
label IDs, paint/fill/erase and existing AI Fill logic are unchanged.

## Images and labels

The exact original PNG bytes are retained at successful image load. HTTP uploads
use those bytes rather than `sourceGray()`, `sourceColor()` or the canvas. This
preserves 16-bit data and still works if the original disk file is later removed.
UI rendering remains its existing 8-bit path.

Response validation checks HTTP status/content type, PNG signature/IHDR, 8-bit
grayscale encoding, exact dimensions, successful decoding and values limited to
0/1/2. No resizing or color guessing occurs. Complete validation happens before
any annotation mutation. RGB, palette, 16-bit labels, unexpected values and
mismatched dimensions fail without changing the annotation.

Semantic mapping resolves the existing label names:

| Canonical MONAI | OrthoSeg label | Current ID |
| --- | --- | --- |
| 0 | Background | 0 |
| 1 | Femur | 1 |
| 2 | Tibia | 2 |

Mapping tests also use Femur=5 and Tibia=8. Missing/invalid mappings are rejected.
Training reverse-maps the **current document mask**, including brush/erase/fill
and applied AI Fill corrections. Unrelated classes such as Fibula map to training
background. The normal document and colored export retain them.

SHA-256 of the original PNG is the stable `case_id`. A hash of the canonical
current mask suppresses repeat prompts/uploads for unchanged annotations in the
same process. Revised masks retain the case ID, so the backend replaces one
active case and preserves revisions. The prediction's model version remains
attached to the case for corrected-data submission. No fine-tuning endpoint or
script is invoked by the UI.

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
