#!/usr/bin/env python3
"""Export Dataset105 region logits for the local OpenCV 5 UI (no argmax)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from app.ml.nnunet_region import load_region_network, REGION_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', default='checkpoint_best.pth',
                        choices=['checkpoint_best.pth', 'checkpoint_final.pth'])
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    if args.fold < 0:
        parser.error('--fold must be nonnegative')
    torch.set_num_threads(4)
    network, _, configuration, _ = load_region_network(args.model_dir, args.checkpoint, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(network, torch.zeros(1, 1, *configuration.patch_size), str(args.output),
                          input_names=['input'], output_names=['region_logits'],
                          opset_version=14, dynamo=False)
    import onnx
    onnx.checker.check_model(str(args.output))
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = {'model': args.output.name, 'source': str(args.model_dir.resolve()),
                'checkpoint': args.checkpoint, 'source_fold': args.fold,
                'checkpoint_sha256': digest(args.model_dir / f'fold_{args.fold}' / args.checkpoint),
                'sha256': digest(args.output), 'labels': REGION_LABELS,
                'output': 'two independent logits, femur then tibia; threshold > 0',
                'input_shape': [1, 1, *map(int, configuration.patch_size)], 'input_height': 2048,
                'runtime': 'OpenCV 5 DNN', 'mask_encoding': '0 background, 1 femur, 2 tibia, 3 overlap'}
    args.output.with_name('manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {args.output}', flush=True)


if __name__ == '__main__':
    main()
