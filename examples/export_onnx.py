#!/usr/bin/env python3
"""Export MedSAM2 (SAM2.1-Hiera-Tiny) to standalone ONNX models for OpenCV 5 + CUDA inference.

Exports:
1. medsam2_image_encoder.onnx:
   Input:  image (1, 3, 1024, 1024) float32 (values [0, 255])
   Output: image_embed (1, 256, 64, 64)
           high_res_0 (1, 32, 256, 256)
           high_res_1 (1, 64, 128, 128)

2. medsam2_mask_decoder.onnx:
   Input:  image_embed (1, 256, 64, 64)
           high_res_0 (1, 32, 256, 256)
           high_res_1 (1, 64, 128, 128)
           box (1, 4) [x0, y0, x1, y1] in [0, 1024]
           mask_input (1, 1, 256, 256) logit map
           has_mask (1,) or (1, 1) float32 (1.0 = use mask_input, 0.0 = box only)
   Output: mask (1, 1, 1024, 1024) logits
           iou_pred (1, 1) float32
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from sam2.build import build_sam2
from export_models import ImageEncoderWrapper, MaskDecoderWrapper, find_default_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export MedSAM2 to standalone ONNX models for OpenCV 5 + CUDA inference"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT_DIR, "sam2.1_hiera_t.yaml"),
        help="Path to YAML config (default: sam2.1_hiera_t.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained checkpoint .pt (default: auto-detected)",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(ROOT_DIR, "models"),
        help="Output directory for exported .onnx models (default: models/)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to build model on (default: cpu for clean deterministic ONNX tracing)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = args.config
    checkpoint = args.checkpoint or find_default_checkpoint()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if not os.path.isfile(config):
        raise FileNotFoundError(f"Config file not found: {config}")

    print("=" * 65)
    print("MedSAM2 ONNX Model Exporter for OpenCV 5 + CUDA")
    print("=" * 65)
    print(f"Config:      {config}")
    print(f"Checkpoint:  {checkpoint}")
    print(f"Output Dir:  {out_dir}")
    print(f"ONNX Opset:  {args.opset}")
    print(f"Trace Dev:   {args.device}")
    print("=" * 65)

    print("\nLoading base SAM2 model...")
    model = build_sam2(config, checkpoint, device=args.device).eval()

    # 1. Export Image Encoder
    print("\n--- 1. Exporting Image Encoder to ONNX ---")
    encoder = ImageEncoderWrapper(model).to(args.device).eval()
    dummy_img = torch.zeros(1, 3, 1024, 1024, device=args.device)

    encoder_path = os.path.join(out_dir, "medsam2_image_encoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_img,),
            encoder_path,
            input_names=["image"],
            output_names=["image_embed", "high_res_0", "high_res_1"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    enc_size_mb = os.path.getsize(encoder_path) / 1024 / 1024
    print(f"✓ Saved Image Encoder to: {encoder_path} ({enc_size_mb:.2f} MB)")

    # 2. Export Mask Decoder
    print("\n--- 2. Exporting Mask Decoder to ONNX ---")
    decoder = MaskDecoderWrapper(model).to(args.device).eval()
    dummy_ie = torch.zeros(1, 256, 64, 64, device=args.device)
    dummy_hr0 = torch.zeros(1, 32, 256, 256, device=args.device)
    dummy_hr1 = torch.zeros(1, 64, 128, 128, device=args.device)
    dummy_box = torch.tensor([[448.0, 100.0, 576.0, 400.0]], device=args.device)
    dummy_mask = torch.zeros(1, 1, 256, 256, device=args.device)
    dummy_has_mask = torch.tensor([0.0], device=args.device)

    decoder_path = os.path.join(out_dir, "medsam2_mask_decoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (dummy_ie, dummy_hr0, dummy_hr1, dummy_box, dummy_mask, dummy_has_mask),
            decoder_path,
            input_names=[
                "image_embed",
                "high_res_0",
                "high_res_1",
                "box",
                "mask_input",
                "has_mask",
            ],
            output_names=["mask", "iou_pred"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    dec_size_mb = os.path.getsize(decoder_path) / 1024 / 1024
    print(f"✓ Saved Mask Decoder to:  {decoder_path} ({dec_size_mb:.2f} MB)")

    print("\n" + "=" * 65)
    print("✓ All MedSAM2 ONNX models exported successfully!")
    print(f"Image Encoder: {encoder_path}")
    print(f"Mask Decoder:  {decoder_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
