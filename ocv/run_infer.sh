#!/bin/bash
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${HERE}/build"
BIN="${BUILD_DIR}/medsam2_ocv_infer"
MODELS_DIR="${HERE}/models"

IMAGE="/home/suman/plai/xraydata/Dataset101_PNG2D/imagesTr/XR00375104.png"
PROMPT_MASK="/home/suman/plai/xraydata/Dataset101_PNG2D/labelsTr/XR00375104.png"
OUT_MASK="${HERE}/out_refined_mask.png"
OUT_OVERLAY="${HERE}/out_overlay.png"

# 1. Export ONNX models if missing
if [ ! -f "${MODELS_DIR}/medsam2_image_encoder.onnx" ] || [ ! -f "${MODELS_DIR}/medsam2_mask_decoder.onnx" ]; then
    echo "[run_infer.sh] Exporting ONNX models..."
    VENV_DIR="${VIRTUAL_ENV:-/home/suman/deepnet}"
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
    fi
    python3 "${HERE}/export_onnx.py"
fi

# 2. Build executable if missing
if [ ! -f "${BIN}" ]; then
    echo "[run_infer.sh] Compiling OpenCV 5 + CUDA C++ executable..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. \
        -DOpenCV_DIR=/home/suman/soft/opencv/install_new/lib/cmake/opencv5
    cmake --build . -j4
    cd "${HERE}"
fi

# 3. Run Inference
if [ "$#" -gt 0 ]; then
    echo "[run_infer.sh] Running medsam2_ocv_infer with custom arguments..."
    "${BIN}" "$@"
else
    echo "[run_infer.sh] Running sample inference with prompt mask (CUDA)..."
    "${BIN}" \
        --image "${IMAGE}" \
        --prompt_mask "${PROMPT_MASK}" \
        --out_mask "${OUT_MASK}" \
        --out_overlay "${OUT_OVERLAY}" \
        --encoder "${MODELS_DIR}/medsam2_image_encoder.onnx" \
        --decoder "${MODELS_DIR}/medsam2_mask_decoder.onnx" \
        --device cuda
fi
