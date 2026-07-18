# OrthoSeg (Native C++ Edition)

A desktop medical-image annotation tool for segmenting lower-limb X-rays
(Femur, Tibia, Fibula) with intensity/edge-aware region-growing fills. This is a
native C++ port of the web reference in [`example/`](example/), built on **Qt 6**
(GUI) and **OpenCV 4** (image processing).

## Layout

| File | Responsibility |
|------|----------------|
| `src/Labels.h` | Label/Tool/FillAlgorithm enums, label colors (BGR) |
| `src/SegmentationEngine.{h,cpp}` | Pure-OpenCV edge map + 3 fill algorithms (no Qt) |
| `src/Document.{h,cpp}` | App state: source, indexed mask, undo history, brush/fill/export |
| `src/CanvasWidget.{h,cpp}` | Canvas rendering, zoom, mouse gestures |
| `src/MainWindow.{h,cpp}` | Sidebar controls + top toolbar |
| `src/main.cpp` | Entry point + global dark theme |
| `tests/test_seg.cpp` | Headless tests for the segmentation core |

The mask is a single-channel `CV_8U` **indexed** image (each pixel holds a label
id 0–3); color is applied only at render/export time. Painting the Background id
(0) erases.

## Algorithms

Click-seed (single click grows one region):

- **Standard** — 4-way BFS; grows while `|intensity − seedIntensity| ≤ intensityThreshold`.
- **Embedded Boundary** — same, but halts at any pixel whose Sobel edge magnitude
  exceeds `edgePenaltyThreshold`.
- **Split-and-Merge** — divides the image into blocks, computes per-block mean
  intensity and max edge magnitude, then unions adjacent blocks (union-find) when
  both are below the edge penalty and similar in mean; fills the seed's component.

Seed-competition (scribble seeds for several labels — including Background —
then press **Run Segmentation**; the labels compete for the ambiguous pixels,
which lets you force a clean split at the knee or hip by seeding both sides):

- **Grow from Seeds (GrowCut)** — synchronous cellular automaton; each seeded
  cell attacks its neighbours with strength `exp(−β·ΔI²)`, iterated to
  convergence. Multi-label.
- **Random Walker** — solves the weighted harmonic (Dirichlet) problem per
  label with edge weights `exp(−β·ΔI²)` via SOR relaxation; each pixel takes the
  argmax label. Degrades gracefully at weak/blurry boundaries. Multi-label.
- **Graph Cut** — OpenCV `grabCut` (color GMMs + min-cut/max-flow). The active
  label's scribbles are hard foreground; all other scribbles are hard
  background. Binary per run; run once per bone with corrective seeds as needed.

Scribble algorithms run at a capped ≤512 px working resolution (seed-preserving
downscale) to stay interactive, then the labeling is upsampled to full size.

## Build & Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/orthoseg          # launch the app
ctest --test-dir build    # or: ./build/test_seg
```

Requires `qt6-base-dev` and `libopencv-dev`.

## Usage

Upload an X-ray → pick an anatomy label → choose Brush, Fill, or Eraser. In Fill
mode, select an algorithm and tune the Intensity/Edge-Penalty sliders. Undo (top
bar) keeps the last 20 mask states. Export writes a color PNG mask.
