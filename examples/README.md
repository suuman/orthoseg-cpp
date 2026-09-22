# MedSAM2 OpenCV 5 + CUDA Standalone C++ Inference

This folder contains a standalone, high-performance C++ implementation of MedSAM2 (SAM2.1-Hiera-Tiny) for 2D medical X-ray anatomical segmentation (Femur and Tibia).

It converts the pipeline to **exclusively use OpenCV 5 and CUDA** without **any PyTorch / LibTorch runtime dependencies**.

---

## Key Highlights & Differences from `cpp/`

| Feature | `cpp/` (LibTorch) | `ocv/` (OpenCV 5 + CUDA) |
| :--- | :--- | :--- |
| **Inference Runtime** | LibTorch (`libtorch.so`, `libc10.so`) | **OpenCV 5 DNN (`cv::dnn`) + CUDA** |
| **PyTorch Dependency** | Required at runtime (~2 GB libraries) | **ZERO runtime PyTorch dependencies** |
| **Binary Size** | ~420 KB (+ ~2 GB LibTorch shared objects) | **69 KB** standalone compiled binary |
| **Model Format** | TorchScript (`.pt`) | **ONNX (`.onnx`) Opset 17** |
| **Image I/O & Processing** | `stb_image` header-only | **Native OpenCV 5 (`imgcodecs`, `imgproc`)** |
| **16-bit X-ray Support** | Custom normalization loop | **OpenCV `minMaxLoc` + `convertTo`** |
| **Compute Acceleration** | LibTorch CUDA backend | **OpenCV DNN CUDA Target (`DNN_TARGET_CUDA`)** |
| **Mean Dice on Test Case** | ~0.9920 | **0.9921** |

---

## Architecture & Pipeline

1. **Standalone Engine (`MedSAM2Engine`)**:
   - Uses `cv::dnn::readNetFromONNX` to load the Image Encoder and Mask Decoder.
   - Automatically selects `cv::dnn::DNN_BACKEND_CUDA` with `cv::dnn::DNN_TARGET_CUDA` when available, falling back gracefully to CPU (`DNN_BACKEND_OPENCV` / `DNN_TARGET_CPU`) if requested.
2. **Generic Image Sizing & Aspect-Ratio Preservation**:
   - Accepts **any input image dimensions** ($W \times H$), whether small ($333 \times 667$), narrow ($128 \times 1024$), square ($512 \times 512$), or large ($2048 \times 1536$).
   - Automatically computes scale factors and centers images onto a $1024 \times 1024$ square canvas.
   - Accurately maps bounding box prompts and prompt masks between original image space and canvas space.
   - **Guaranteed Output Sizing**: Crops and un-scales output masks and color overlays back to the **exact original image dimensions ($W \times H$)**.
3. **Multi-Prompt Input Modes**:
   - **Prompt Mask Input (`--prompt_mask`)**: Automatically discovers all anatomical class IDs in the mask (or filters by `--class_ids`), derives bounding boxes and $256 \times 256$ logit maps, and refines them.
   - **Flexible Bounding Box Input (`--box`, `--box2`, `--boxes`)**: Accepts single or multiple bounding box coordinates in original image space.
4. **Dual Output Generation**:
   - **Refined Mask (`--out_mask`)**: Single-channel PNG matching the exact original image size with pixel labels (0 = background, 1 = femur, 2 = tibia, etc.).
   - **Color Overlay (`--out_overlay`)**: Semi-transparent alpha-blended visualization matching the exact original image dimensions.

---

## Directory Layout

```
orthoseg/
├── CMakeLists.txt             # Unified build configuration linking Qt 6, OpenCV + CUDA
├── include/                   # All project headers (.h)
│   ├── medsam2_ocv.h          # MedSAM2 C++ engine header
│   ├── image_utils.h          # OpenCV 5 image loader, square pad/unpad, boxes, overlay
│   └── ...                    # OrthoSeg GUI and segmentation engine headers
├── src/                       # All project source files (.cpp)
│   ├── medsam2_ocv.cpp        # OpenCV 5 DNN CUDA inference implementation
│   └── ...                    # OrthoSeg GUI and segmentation engine implementation
├── models/                    # Exported ONNX models
│   ├── medsam2_image_encoder.onnx  # Exported Image Encoder ONNX model (104 MB)
│   └── medsam2_mask_decoder.onnx   # Exported Mask Decoder ONNX model (15 MB)
├── examples/                  # Standalone CLI inference and export utilities
│   ├── medsam2_main.cpp       # CLI executable matching medsam2_infer_custom.py
│   ├── export_onnx.py         # ONNX model exporter script
│   ├── run_infer.sh           # Convenience build & run script
│   └── README.md              # Detailed documentation
└── tests/                     # Unit test suites
```

---

## ONNX Model Export

To generate or re-export the ONNX models from checkpoint:

```bash
cd examples
/home/suman/deepnet/bin/python export_onnx.py
```

The exporter produces:
- `models/medsam2_image_encoder.onnx`: Input `image` $[1, 3, 1024, 1024]$, outputs `image_embed`, `high_res_0`, `high_res_1`.
- `models/medsam2_mask_decoder.onnx`: Inputs `image_embed`, `high_res_0`, `high_res_1`, `box`, `mask_input`, `has_mask`, outputs `mask` $[1, 1, 1024, 1024]$ and `iou_pred` $[1, 1]$.

---

## How to Build

### Quick Run with the Script
```bash
cd examples
./run_infer.sh
```

### Manual Build with CMake
From the repository root:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR=/home/suman/soft/opencv/install_new/lib/cmake/opencv5

cmake --build build --target medsam2_ocv_infer -j4
```

The resulting executable `./build/medsam2_ocv_infer` embeds RPATH to `/home/suman/soft/opencv/install_new/lib` and CUDA libraries, so it runs standalone without needing `LD_LIBRARY_PATH`.

---

## CLI Usage & Examples

### 1. Run with a Prompt Mask
```bash
./build/medsam2_ocv_infer \
    --image /home/suman/plai/xraydata/Dataset101_PNG2D/imagesTr/XR00375104.png \
    --prompt_mask /home/suman/plai/xraydata/Dataset101_PNG2D/labelsTr/XR00375104.png \
    --out_mask out_refined_mask.png \
    --out_overlay out_overlay.png \
    --device cuda
```

### 2. Run with Bounding Boxes
```bash
./build/medsam2_ocv_infer \
    --image /home/suman/plai/xraydata/Dataset101_PNG2D/imagesTr/XR00375104.png \
    --box "10,275,113,622" --class_id 1 \
    --box2 "51,621,121,913" --class_id2 2 \
    --out_mask box_mask.png \
    --out_overlay box_overlay.png \
    --device cuda
```

### 3. Run on CPU
```bash
./build/medsam2_ocv_infer \
    --image /home/suman/plai/xraydata/Dataset101_PNG2D/imagesTr/XR00375104.png \
    --box "10,275,113,622" --class_id 1 \
    --out_mask cpu_mask.png \
    --device cpu
```

---

## Options Reference

| Option | Description | Default |
| :--- | :--- | :--- |
| `--image <path>` | **Required.** Path to input X-ray image (PNG/JPG) | None |
| `--out_mask <path>` | **Required.** Path to output refined mask (PNG) | `output_mask.png` |
| `--out_overlay <path>` | Path to output overlay visualization (PNG) | None |
| `--prompt_mask <path>` | Ground-truth or coarse mask PNG for prompts & evaluation | None |
| `--box "x0,y0,x1,y1"` | First bounding box prompt coordinates | None |
| `--class_id <int>` | Class ID for first box (1 = Femur) | `1` |
| `--box2 "x0,y0,x1,y1"` | Second bounding box prompt coordinates | None |
| `--class_id2 <int>` | Class ID for second box (2 = Tibia) | `2` |
| `--encoder <path>` | Path to Image Encoder ONNX model | `models/medsam2_image_encoder.onnx` |
| `--decoder <path>` | Path to Mask Decoder ONNX model | `models/medsam2_mask_decoder.onnx` |
| `--device <cuda\|cpu>` | Compute device | `cuda` |
| `--pad_to_square <bool>`| Center-pad rectangular images to preserve aspect ratio | `true` |
| `--pad_size <int>` | Canvas size for square padding | `1024` |
| `--box_pad <int>` | Pixels to grow derived bounding boxes | `0` |
| `--score_thresh <float>`| Logit threshold for foreground | `0.0` |

---

## Accuracy & Verification

Evaluating on test case `XR00375104.png` on CUDA:

- **Class 1 (Femur)**: Dice = **0.9920**, IoU = **0.9841**
- **Class 2 (Tibia)**: Dice = **0.9923**, IoU = **0.9847**
- **Overall Mean Dice**: **0.9921** (matches/exceeds baseline PyTorch and LibTorch implementation)
