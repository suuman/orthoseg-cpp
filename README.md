# OrthoSeg

OrthoSeg is a desktop annotation tool for segmenting the **femur, tibia, and
fibula** in lower-limb X-rays. The native application uses **C++17**, **Qt 6
Widgets**, and **OpenCV**, with manual painting and six interactive segmentation
algorithms, plus **AI Fill** with native MedSAM2 and nnUNet v2 options.
Processing runs locally without API keys. Native nnUNet v2 runs through OpenCV 5
DNN with no Python, PyTorch, or MONAI server. Native MedSAM2 requires two ONNX
models and CUDA-enabled OpenCV DNN; MONAI SAM2 uses the separate local backend.

The original React/TypeScript implementation is included in [`example/`](example/).
The instructions below describe the native application; see
[Web reference](#web-reference) to run the browser version.

## Features

- Brush and eraser tools with adjustable stroke size.
- Three fills that grow a region from a single click.
- Three algorithms that segment from labeled scribbles, including background seeds.
- Color mask overlay with adjustable opacity, zoom, and middle-button panning.
- AI Fill with bounding-box, painted-mask, and loaded-mask prompts for femur/tibia.
- Undo for the last 20 edits, restoring both the mask and seed layer.
- Export of a color PNG mask at the source image dimensions.

## Build and run

### Requirements

- A C++17 compiler.
- CMake 3.16 or newer.
- Qt 6 development files for the `Widgets` component.
- OpenCV development files for `core`, `imgproc`, `imgcodecs`, and `dnn` (OpenCV 4+).
- For AI Fill: an NVIDIA GPU/driver and OpenCV DNN built with CUDA and cuDNN.
  Distribution OpenCV packages can build the application but may lack CUDA support.

On Debian/Ubuntu, install the build dependencies:

```bash
sudo apt update
sudo apt install build-essential cmake qt6-base-dev libopencv-dev
```

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/orthoseg
```

If Qt or OpenCV is installed outside the usual system locations, set
`CMAKE_PREFIX_PATH`, or provide `Qt6_DIR` and `OpenCV_DIR`, when configuring CMake.
The normal application launch requires a graphical desktop session.

### Run the tests

After building, run CTest from the build directory:

```bash
cd build
ctest --output-on-failure
```

CTest runs four suites:

- `segmentation`: synthetic-image checks for the Sobel edge map, all six algorithms,
  background erasing, and seed requirements.
- `document`: image I/O, preservation of annotations and seeds during resizing,
  validation, and segmentation undo.
- `ui`: existing controls and dialogs, AI box/paint gestures under zoom/pan,
  prompt import validation, result overlays, and applying/undoing a preview using
  Qt's offscreen platform.
- `ai`: box/mask/result validation, the actual `ocv` prompt adapter, and worker
  error delivery/duplicate-request prevention. No model inference is simulated.

The core and document tests run without Qt GUI code. The current CMake
configuration still requires Qt 6 to configure the project. To run only the
segmentation engine checks, use `./build/test_seg` from the repository root.

## Annotating an image

### Open and paint

1. Click **Upload X-ray** and open a PNG, JPEG, BMP, or TIFF image.
2. Select **Femur**, **Tibia**, or **Fibula** under **Select Anatomy**.
3. Choose **Brush**, set **Brush Size**, and drag with the left mouse button to
   paint. Choose **Eraser**, or paint with **Background**, to remove labels.
4. Use **Mask Opacity** to adjust the overlay. The mouse wheel and toolbar
   **+**/**−** buttons change zoom; **Reset Zoom** returns to the fitted view.
5. Click **Export Mask** to save the result. The default filename is
   `bone_segmentation_mask.png`.

Opening another image starts a new mask and seed layer and resets undo history.
Annotations are held in memory, so export the mask before changing images or
closing the application.

### Fill from a click

Choose **Fill**, select **Standard Growing**, **Embedded Boundary**, or
**Split-and-Merge**, then click inside the region to label. Adjust the intensity
and edge settings as needed. A fill writes the active label over every pixel it
reaches, including existing annotations. Selecting **Background** erases the
filled region.

### Segment from scribbles

1. Choose **Fill** and select **Grow from Seeds**, **Random Walker**, or
   **Graph Cut**.
2. Select a bone label and draw short strokes inside that structure. Switch labels
   to seed other structures and use **Background** outside the bones or between
   neighboring structures. At least two distinct seed labels are required.
3. For **Graph Cut**, select the target bone again before running: the active
   label is foreground, and every other seed label is background for that run.
4. Click **Run Segmentation**. Add corrective seeds and rerun as needed, then use
   Brush or Eraser for manual refinements.

In this workflow, strokes modify a separate seed layer. **Background** creates a
competing seed, displayed in gray. The Eraser tool edits the result mask; use
**Undo** to revert a seed stroke or **Clear Seeds** to remove all seeds. Seed
strokes use **Brush Size**, which remains available while drawing seeds.

**Grow from Seeds** and **Random Walker** replace the entire result mask on each
run. **Graph Cut** assigns the active label to foreground pixels and clears its
previous pixels where the cut selects background; other labels at background
pixels are retained. Foreground pixels can overwrite another label.

### AI Fill (MedSAM2)

1. Upload an X-ray and select **Femur** or **Tibia** using the existing anatomy selector.
2. Select **AI Fill**, then choose a prompt type:
   - **Bounding Box**: left-drag a rectangle; drag again to replace it.
   - **Paint Mask**: paint foreground with the existing **Brush Size** control;
     check **Erase Prompt** to erase. Brush size is in source-image pixels.
   - **Load Mask**: click **Load Prompt Mask** and choose PNG, TIFF, BMP, or JPEG.
     Dimensions must exactly match the displayed image. Nonzero RGB/grayscale
     values become foreground for the selected anatomy; alpha is ignored. Supply
     a binary mask for one bone, not a combined femur/tibia label map. JPEG artifacts
     can add foreground pixels, so lossless formats are preferable.
3. Open **Local Models** in the top bar and set the MedSAM2 directory to one containing
   `medsam2_image_encoder.onnx` and `medsam2_mask_decoder.onnx`. The default is this
   checkout's `models`; `MEDSAM2_MODEL_DIR` can override it at launch.
4. Click **Run AI Fill**. Status reports loading/inference, duplicate requests are
   disabled, and the viewer remains interactive. First use loads the models on a
   persistent worker thread; subsequent runs reuse them. Changing model directory
   or recovering from a failed forward releases the old engine.
5. The result appears in the selected anatomy's color, separately from the cyan
   prompt and existing annotations. **Mask Opacity** controls result opacity.
   Use **Show AI segmentation**, **Clear AI Segmentation**, and **Clear Prompt**
   independently. **Clear All** clears AI prompts/results as well as annotations.
6. Click **Apply AI Result to Mask** to copy foreground pixels into the
   existing label mask, then use Brush/Eraser, Undo, or Export Mask. Applying can
   overwrite other labels at foreground pixels; pixels outside the AI foreground
   are retained. The preview is consumed on apply. Unapplied previews are not exported.

Use the mouse wheel or existing buttons to zoom; middle-drag pans in any tool.
Reset Zoom also resets pan. Mouse events and drawing share Qt logical coordinates,
so zoom, fit, aspect ratio, pan, and HiDPI do not change source prompt coordinates.
Box endpoints follow `ocv`'s inclusive pixel convention (`0..width-1`, `0..height-1`).
Painted and loaded masks remain binary `CV_8UC1` at original resolution.

The original image loader is unchanged: it supplies the displayed 8-bit BGR pixels,
including its existing grayscale/16-bit-to-8-bit conversion. AI Fill converts this
in-memory source to RGB exactly once; it does not reload the file or normalize it
again. `ocv` handles centered 1024-square padding, blob creation, mask logits, and
restoration to source dimensions with its existing defaults.

The implemented call path is:

```text
CanvasWidget gestures / MainWindow::onLoadPromptMask
  -> Document::aiFill (box or source-resolution binary prompt)
  -> MainWindow::onRunAIFill
  -> AIFillController::run (persistent QThread)
  -> MedSAM2Inference::preparePrompt / run
  -> ocv boxFromMask / makeMaskLogits (mask prompts only)
  -> MedSAM2Engine::segment (existing OpenCV DNN CUDA implementation)
  -> source-sized indexed cv::Mat
  -> queued AIFillController::completed on UI thread
  -> Document::aiFill().resultMask -> CanvasWidget::paintEvent
  -> optional Document::applyAIResult -> existing mask editing/undo/export
```

Opening another image or clearing the result while inference runs invalidates its
pending result. Source images are retained by reference counting for the worker;
prompt masks are snapshotted. Model loading, preprocessing, and forward passes run
on that worker, with errors delivered through the existing message-box mechanism.
CUDA absence is reported explicitly; AI Fill does not silently fall back to CPU.
Closing during inference waits for the current CUDA call to finish safely.

To build with the CUDA-enabled OpenCV installation used by `examples/run_infer.sh` on
this machine (no OpenCV/CUDA version change is needed):

```bash
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/home/suman/soft/opencv/install_new/lib/cmake/opencv5
cmake --build build/cuda --parallel 4
ctest --test-dir build/cuda --output-on-failure
MEDSAM2_MODEL_DIR="$PWD/models" ./build/cuda/orthoseg
```

On another machine, set `OpenCV_DIR` to its CUDA-enabled OpenCV CMake directory.
The default build commands above remain supported, but AI Fill reports an error
if that OpenCV build has no CUDA DNN target. See `examples/README.md` for the existing
model export and standalone inference instructions.

For manual acceptance, run each of the three prompt workflows above on an X-ray,
repeat after zooming and panning, and verify femur and tibia in separate runs.
Check result visibility, clearing, applying, undo, and exporting; then load a
wrong-sized prompt and verify the error. Repeat inference to check model reuse
(the engine's model-loading messages should appear only on first use). Load a new
image during inference and confirm the old result is discarded.

Implementation verification in this environment: the system OpenCV 4.10 and custom
OpenCV 5.1 builds and automated suites were exercised. The standalone `ocv`
executable ran box and femur/tibia mask inference with the real supplied models,
returning source-sized masks, but selected CPU fallback. CUDA end-to-end AI Fill
must still be verified on a machine with an available NVIDIA driver/device.

### Controls and settings

| Control | Range / default | Behavior |
| --- | --- | --- |
| Brush Size | 2–100 px / 20 px | Stroke width in source image pixels; also used for erasing and seeds. |
| Intensity Thresh | 1–50 / 5 | Allowed intensity difference for click fills; larger values allow more variation. |
| Edge Penalty | 1–255 / 30 | Edge cutoff for Embedded Boundary and Split-and-Merge; lower values block growth at weaker edges. |
| Edge Sensitivity (β) | 1–100 / 30 | Grow from Seeds and Random Walker only; larger values reduce propagation across intensity changes. Internally, β = slider value × 0.0001. |
| Mask Opacity | 10–100% / 50% | Display opacity; exported mask colors stay fully opaque. |
| Zoom | 25–400% / 100% | Changes in 25 percentage-point steps relative to the fitted view. |
| Undo | Up to 20 edits | Restores the mask and seeds before a stroke, fill, segmentation run, or clear action. |
| Clear Seeds | — | Removes seed strokes while retaining the result mask. |
| Clear All | — | Clears both the mask and seeds while keeping the source image. |

## Segmentation algorithms

| Algorithm | Input | Implementation |
| --- | --- | --- |
| Standard Growing | Single click | Visits four-connected neighbors while their intensity differs from the seed intensity by at most the threshold. |
| Embedded Boundary | Single click | Applies the same intensity test and stops at pixels whose Sobel edge magnitude exceeds Edge Penalty. |
| Split-and-Merge | Single click | Divides the image into 4 × 4 blocks, then joins adjacent blocks with similar mean intensity when both maximum edge magnitudes are within the cutoff. Fills the seed block's connected component. |
| Grow from Seeds (GrowCut) | Labeled scribbles | Neighboring pixels compete for labels using strength weighted by `exp(−β × ΔI²)`. Updates synchronously until stable or the iteration limit is reached. |
| Random Walker | Labeled scribbles | Solves one weighted harmonic probability field per seeded label using successive over-relaxation, then chooses the label with the highest value at each pixel. Uses the same intensity-based edge weights as GrowCut. |
| Graph Cut | Foreground and background scribbles | Uses OpenCV `grabCut` with color mixture models and a graph cut, running three iterations by default. Segments one active label per run. |

Click fills run at full resolution. Scribble algorithms run with the longest
working image dimension capped at **512 pixels**, then resize the result to the
original dimensions using nearest-neighbor interpolation. Seed pixels are mapped
into the reduced grid so thin strokes are not simply skipped, but competing seeds
can collide in the same reduced pixel. Original seed constraints are restored at
full resolution. If resizing removes an entire seed label, the run is rejected
without changing the mask or undo history; draw larger, separated strokes and
retry. Fine boundaries can still be affected by resizing; refine the result at
full resolution with Brush or Eraser. Graph Cut merges its foreground selection
into the original mask, preserving other labels at background pixels exactly.

The Sobel edge map has a one-pixel zero border, matching the web reference.
Embedded Boundary can therefore grow around a barrier along the image boundary
when the intensity threshold permits it.

## Image and mask formats

The native loader reads images as 8-bit BGR color and computes grayscale intensity
as the mean of the three channels. The internal mask is a single-channel
`CV_8UC1` image containing these label IDs:

| ID | Label | Export color (RGB hex) |
| --- | --- | --- |
| 0 | Background | `#000000` |
| 1 | Femur | `#ef4444` |
| 2 | Tibia | `#22c55e` |
| 3 | Fibula | `#3b82f6` |

The separate seed layer uses IDs 0–3 and `255` for unseeded pixels. In the result
mask, ID 0 means background; in the seed layer, ID 0 is an explicit background
seed.

**Export Mask** writes a three-channel color PNG with a black background. It
contains the mask only, with no source X-ray, seed strokes, or transparency. To
recover label IDs from an exported PNG, map its RGB colors using the table above.

The current application works with individual raster images. It has no direct
DICOM reader, project save/reload, or indexed-label export. AI Fill imports binary
prompt masks but does not import a full multi-label annotation project. Convert
DICOM images to a supported raster format before opening them; original
high-bit-depth image values are not preserved by the current loader.

## Project layout

| Path | Responsibility |
| --- | --- |
| [`CMakeLists.txt`](CMakeLists.txt) | Unified CMake configuration defining the libraries (`medsam2_ocv`, `orthoseg_core`, `orthoseg_ai`), executables (`orthoseg`, `medsam2_ocv_infer`), and test suites. |
| [`include/`](include/) | All header files: labels ([`Labels.h`](include/Labels.h)), document ([`Document.h`](include/Document.h)), algorithms ([`SegmentationEngine.h`](include/SegmentationEngine.h)), AI inference ([`MedSAM2Inference.h`](include/MedSAM2Inference.h), [`medsam2_ocv.h`](include/medsam2_ocv.h)), and UI widgets. |
| [`src/`](src/) | All C++ implementation files: GUI application entry ([`main.cpp`](src/main.cpp)), canvas ([`CanvasWidget.cpp`](src/CanvasWidget.cpp)), window ([`MainWindow.cpp`](src/MainWindow.cpp)), document ([`Document.cpp`](src/Document.cpp)), segmentation core ([`SegmentationEngine.cpp`](src/SegmentationEngine.cpp)), and MedSAM2 inference ([`MedSAM2Inference.cpp`](src/MedSAM2Inference.cpp), [`medsam2_ocv.cpp`](src/medsam2_ocv.cpp)). |
| [`models/`](models/) | Local ONNX models for MedSAM2 and `nnunet2/model.onnx` for native nnUNet v2. |
| [`examples/`](examples/) | Standalone CLI inference tool ([`medsam2_main.cpp`](examples/medsam2_main.cpp)), runner script ([`run_infer.sh`](examples/run_infer.sh)), exporter ([`export_onnx.py`](examples/export_onnx.py)), and standalone guide. |
| [`tests/test_seg.cpp`](tests/test_seg.cpp) | Headless checks for the segmentation engine. |
| [`tests/test_document.cpp`](tests/test_document.cpp) | Regression checks for image I/O, segmentation resizing, validation, and undo. |
| [`tests/test_ui.cpp`](tests/test_ui.cpp) | Offscreen Qt checks for controls, repainting, and file dialogs. |
| [`tests/test_ai.cpp`](tests/test_ai.cpp) | Prompt adapter, box/mask validation, and AI worker tests. |

### Capture the UI without a display

From the repository root, use Qt's offscreen platform and the built-in screenshot
arguments:

```bash
QT_QPA_PLATFORM=offscreen ./build/orthoseg \
  --fill-algo 3 --screenshot build/orthoseg-growcut.png
```

`--fill-algo` selects an algorithm by index: `0` Standard Growing, `1` Embedded
Boundary, `2` Split-and-Merge, `3` Grow from Seeds, `4` Random Walker, or `5` Graph
Cut. `--screenshot` captures the window shortly after startup and exits. This
checks UI rendering; it does not load an image or run segmentation.

## Web reference

The browser version provides Brush, Eraser, and the three click-fill algorithms.
Its annotation logic runs in the browser using canvas; the native application
adds the scribble algorithms and stores labels in an indexed mask. The web
version exports a canvas PNG with transparent background pixels.

With Node.js and npm installed, run from the repository root:

```bash
cd example
npm install
npm run dev
```

Open the local URL printed by Vite (port 3000 by default). `npm run build` creates
the production bundle in `example/dist/`, and `npm run lint` runs the TypeScript
type checker.

The example retains AI Studio configuration and a `GEMINI_API_KEY` placeholder,
but its current annotation code makes no Gemini API calls and needs no API key.

## Native nnUNet v2 in AI Fill

Select **AI Fill → Native nnUNet v2 (OpenCV 5)**, open an X-ray, and click
**Run Native nnUNet**. The default model is `models/nnunet2/model.onnx`, copied
from `/run/media/suman/Data/xray/ocv`; the model field in **Local Models** or
`ORTHOSEG_NNUNET_MODEL` can point to another ONNX file. No MONAI server or
Python process is used. The choice is built when `ENABLE_NATIVE_NNUNET=ON`
(default) and CMake finds OpenCV 5 or newer. The supplied OpenCV 5.1 build
has CUDA/cuDNN and TBB support.

The image is resized to **2048 pixels high** while preserving its aspect ratio.
OpenCV DNN runs 2048×768 tiles; the returned 0/1/2 mask is restored to source
size. **Auto** uses CUDA FP16 if a CUDA device is available, otherwise the
multithreaded CPU backend. The device selector in **Local Models** also offers
CUDA FP32 and CPU. The result appears as a preview; use **Show AI segmentation**,
**Clear AI Segmentation**, and **Apply AI Result to Mask** to review or commit it.
Applying replaces Femur/Tibia in one undoable edit, preserves Fibula, and allows
normal editing and export. Edits or an image change during inference cause
the result to be discarded. The ONNX file is kept locally and ignored by Git.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/home/suman/soft/opencv/install_new/lib/cmake/opencv5
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/orthoseg
```

The native inference and UI tests use the real ONNX file on CPU. GPU execution
was not available for verification on this workstation.

## Optional MONAI AI Segment integration

The **AI Fill** panel also offers **MedSAM2**, **MONAI production (UNet)**,
**MONAI SAM2 (boxes)**, and **MONAI nnUNet v2**. Both MONAI production and nnUNet v2
provide automatic Femur/Tibia predictions. The supplied nnUNet checkpoint has
the right labels but a different architecture from the MONAI production UNet,
so it is a separate model choice. MONAI SAM2 uses Femur/Tibia boxes; MedSAM2
keeps its ONNX prompt workflow. The top-bar **AI Segment** action runs the
MONAI production UNet.
The build additionally requires the Qt 6 `Network` component (provided by
`qt6-base-dev`). No Python, PyTorch or MONAI dependency is added to the UI.

```bash
# Start the independently installed MONAI backend from its own repository:
/run/media/suman/Data/monai/scripts/run_server_models.sh \
  --config /run/media/suman/Data/monai/configs/management.yaml

# In this OrthoSeg repository:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
MONAI_BACKEND_URL=http://127.0.0.1:8000 ./build/orthoseg
```

To start both applications with one command after building OrthoSeg:

```bash
./scripts/launch_with_monai.sh
```

The launcher uses the separate MONAI checkout at
`/run/media/suman/Data/monai`, starts its model-enabled server with
`configs/management.yaml`, waits for `/health`, then opens OrthoSeg. If a
healthy backend is already running at the URL, the launcher reuses it. When
OrthoSeg closes, the launcher stops only the server it started. Override
`MONAI_REPO`, `MONAI_CONFIG`, `MONAI_BACKEND_URL`, or `ORTHOSEG_BIN` for a
different checkout, config, port, or build location. If the server has no
production checkpoint, OrthoSeg still opens with prompted SAM2 and automatic
nnUNet v2 inference if their local checkpoints loaded. The MONAI production
UNet requires a separately trained checkpoint and backend restart.

`MONAI_BACKEND_URL` is optional and defaults to `http://127.0.0.1:8000`.
The AI Fill panel shows this URL and lets you change it for the current session.
Its status dot turns green when `/health` reports the model selected in the
panel as loaded; red means the service or selected model is unavailable. Every
MONAI segmentation checks health again before sending the image. The panel
shows that model's version. The backend does not offer arbitrary archived-version inference.
Only local HTTP URLs are accepted; proxy use and redirects are disabled.
Requests run asynchronously; health and management availability checks time out
after five seconds, while segmentation and uploads retain a three-minute deadline.
The normal annotation workflow never launches, trains, promotes, or reloads a
model. A missing selected model is reported without changing the annotation.

Open an original PNG, select **MONAI production (UNet)** in AI Fill and click
**Run MONAI Segment**, or use the top-bar **AI Segment** action. Review the preview
with **Show AI segmentation**; use **Apply AI Result to Mask** to edit or export,
or **Clear AI Segmentation** to discard it. Existing Femur/Tibia annotations
require replacement confirmation; Fibula is preserved. Applying a prediction is
one undoable edit. Changes made during inference cause the
result to be discarded. Finish/apply/clear an outstanding AI Fill preview before
AI Segment. PNG bytes are retained at load, including original 16-bit data;
the display image and original file remain unchanged. JPEG/TIFF/etc. loading
still works normally, but MONAI operations require an original PNG up to 32 MiB.

For automatic inference with the supplied nnUNet v2 checkpoint, select
**MONAI nnUNet v2** in AI Fill and click **Run nnUNet v2**. The backend resizes
the image to 2048 pixels high with its aspect ratio preserved, runs 2D sliding
window inference, and restores the 0/1/2 mask to the original size. This
prediction follows the same editable, undoable, and exportable mask workflow.
Its health indicator and version are independent of the MONAI production UNet.
Model Management trains and promotes the MONAI production UNet only.

For prompted inference, open a PNG X-ray, choose **MONAI SAM2 (boxes)** in AI Fill,
select Femur or Tibia, and drag a box on the image. For a bilateral case, click
**Next bounding box (2)** and draw the second box for that bone; repeat after
selecting the other bone if needed. The button switches back to box 1 for edits,
and each box has its own Clear control. Click **Run MONAI SAM2**. The server
resizes the image to 1024 pixels high while preserving aspect ratio, maps the
boxes into that image, and restores the mask to the original dimensions. The
source-sized 0/1/2 result becomes one
undoable annotation edit and can be exported with **Export Mask**. If only one
bone is prompted, the current annotation for the other bone is preserved.
SAM2 model readiness and version are separate from the automatic production
model. The model management window manages the automatic production model and
its training jobs, not this prompted checkpoint.

After a successful **Export Mask**, **Add to AI Training** optionally uploads
the original PNG plus a fresh canonical 0/1/2 mask made from the current editable
annotation. The existing colored PNG export remains authoritative and unchanged.
Apply AI Fill previews before export as usual. Fibula is background in the
Femur/Tibia training mask and remains present in the normal export.
**Save Only** sends nothing. Unchanged accepted/declined annotations are not
prompted again in that application session; a changed annotation can be offered
again. Failed uploads preserve the export and allow retry on the next export.
A first-time empty annotation is not offered; clearing a previously offered or
MONAI-segmented case can be submitted as an all-background correction.

See [MONAI_INTEGRATION.md](MONAI_INTEGRATION.md) for the API, mapping, tests,
and the optional real-backend integration check.

## Optional administrative model management

The AI Fill panel shows **Model Management**. It becomes enabled only when the
local backend accepts `/management/status`. Click it to open the separate
technical-user dialog. The optional legacy Tools menu action is available when launched with:

```bash
ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1 ./build/orthoseg
```

The backend must also have
`management.enabled: true`; its supplied `configs/management.yaml` enables it.
The dialog uses the URL currently shown in AI Fill.

The modeless dialog shows backend/device status, disk and loaded production
versions, training/validation counts, candidate metrics, and recent job logs.
**Fine-tune New Candidate** and **Promote Candidate** each require explicit
confirmation. Backend defaults control training; administrative jobs use bounded
CPU threads to leave the production GPU inference model available. Existing CLI
GPU training remains separate. Promotion preserves older releases and requires
a backend restart; the panel shows when loaded and on-disk versions differ.

Opening/refreshing the dialog never trains. Idle panels do not poll; active jobs
refresh every five seconds while the panel is visible. Closing the UI does not
cancel backend training. [MODEL_MANAGEMENT.md](MODEL_MANAGEMENT.md) documents the
API and validation. CTest now includes the focused `management` suite as well as
all existing annotation and MONAI integration tests.
