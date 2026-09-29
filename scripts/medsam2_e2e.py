"""Real-checkpoint smoke test through HTTP, OrthoSeg editing and model management.

Uses an isolated synthetic X-ray dataset; never promotes into the user's models.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import httpx
import numpy as np
from PIL import Image
import yaml

from app.core.config import ROOT, load_config
from app.ml.preprocessing import mask_statistics
from app.ml.sam2_model import asset_root


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orthoseg-build', type=Path, default=Path('/home/suman/agentic_coding/orthoseg/build'))
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--inference-only', action='store_true')
    args = parser.parse_args()
    work = (args.work_dir or ROOT / 'data' / 'medsam2-e2e' / uuid.uuid4().hex).resolve()
    work.mkdir(parents=True, exist_ok=False)
    config = load_config()
    assets = asset_root(config).resolve()
    reference = assets / 'checkpoint.pt'
    original_digest = hashlib.file_digest(reference.open('rb'), 'sha256').hexdigest()
    config['paths'] = {'data': str(work / 'data'), 'models': str(work / 'models')}
    config['sam2']['asset_directory'] = str(assets)
    config['inference']['device'] = 'cpu'
    config['management'].update(enabled=True, cpu_threads=2)
    config['training'].update(epochs=1, batch_size=1, new_case_repeat_factor=1)
    config['training']['augmentation']['probability'] = 0
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        config['server']['port'] = listener.getsockname()[1]
    path = work / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    base = f"http://127.0.0.1:{config['server']['port']}"
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'XRAY_CONFIG': str(path),
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'QT_QPA_PLATFORM': 'offscreen',
           'MONAI_BACKEND_URL': base, 'MPLCONFIGDIR': str(work / 'matplotlib')}
    process = None
    log = (work / 'server.log').open('ab')
    client = httpx.Client(base_url=base, trust_env=False, timeout=180)
    def start():
        nonlocal process
        process = subprocess.Popen([sys.executable, '-m', 'app.serve', '--config', str(path)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Backend exited; see {work / "server.log"}')
            try:
                health = client.get('/health', timeout=2)
                if health.status_code == 200:
                    assert health.json()['sam2_model_loaded'], health.text
                    return health.json()
            except httpx.TransportError:
                pass
            time.sleep(.2)
        raise RuntimeError('Backend startup timed out')
    def stop():
        nonlocal process
        if process is not None:
            process.terminate()
            process.wait(timeout=30)
            process = None
    try:
        health = start()
        image = np.zeros((128, 64), np.uint16)
        image[10:118, 18:46] = 32000
        image[22:106, 24:40] = 50000
        image_path = work / 'xray.png'
        image_path.write_bytes(png(image))
        boxes = {'femur_box': '[15,8,48,120]', 'tibia_box': '[15,8,48,120]'}
        def infer(name):
            response = client.post('/segment/prompted', files={'image': ('xray.png', image_path.read_bytes())}, data=boxes)
            response.raise_for_status()
            (work / name).write_bytes(response.content)
            mask = np.array(Image.open(io.BytesIO(response.content)))
            stats = mask_statistics(mask)
            assert mask.shape == (*image.shape, 3)
            assert stats['overlap_pixels'] > 0, stats
            return response.headers['x-model-version'], stats
        first, initial_stats = infer('initial.png')
        assert first == health['sam2_model_version']
        print('Reference inference:', first, initial_stats, flush=True)
        if args.inference_only:
            return
        corrected = work / 'corrected.png'
        with (work / 'ui-edit.log').open('wb') as output:
            subprocess.run([str(args.orthoseg_build / 'test_monai'), '--live-medsam',
                            str(image_path), str(corrected)], env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True, timeout=240)
        manifests = list((work / 'data/training/metadata').glob('*.json'))
        assert len(manifests) == 1
        record = json.loads(manifests[0].read_text())
        stored = work / 'data/training' / record['mask']
        target = np.array(Image.open(stored))
        np.testing.assert_array_equal(target, np.array(Image.open(corrected)))
        assert mask_statistics(target)['overlap_pixels'] > 0
        assert record['model_version'] == first
        validation = work / 'data/validation'
        for folder in ('images', 'labels'):
            (validation / folder).mkdir(parents=True)
        (validation / 'images/validation.png').write_bytes(png(np.roll(image, 1, axis=1)))
        (validation / 'labels/validation.png').write_bytes(png(np.roll(target, 1, axis=1)))
        with (work / 'ui-management.log').open('wb') as output:
            subprocess.run([str(args.orthoseg_build / 'test_management'), '--live-medsam',
                            str(work / 'management.png')], env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True, timeout=660)
        state = client.get('/management/status', params={'model': 'medsam2'}).json()
        assert state['job']['status'] == 'completed', state
        second = state['production']['version']
        assert second != first and state['candidate']['parent_model_version'] == first
        assert state['restart_required']
        (work / 'management.json').write_text(json.dumps(state, indent=2))
        stop()
        reloaded = start()
        assert reloaded['sam2_model_version'] == second
        final, final_stats = infer('finetuned.png')
        assert final == second
        # Exercise real mask refinement as well as boxes after loading the candidate.
        response = client.post('/segment/prompted', files={
            'image': ('xray.png', image_path.read_bytes()), 'mask': ('prompt.png', png(target))})
        response.raise_for_status()
        refined = np.array(Image.open(io.BytesIO(response.content)))
        mask_statistics(refined)
        assert refined.shape == target.shape
        assert hashlib.file_digest(reference.open('rb'), 'sha256').hexdigest() == original_digest
        (work / 'result.json').write_text(json.dumps({'reference': first, 'candidate': second,
            'initial': initial_stats, 'corrected': mask_statistics(target), 'final': final_stats,
            'refined': mask_statistics(refined), 'passed': True}, indent=2))
        print('Full UI fine-tuning workflow passed:', second, final_stats, flush=True)
    finally:
        stop()
        client.close()
        log.close()
        print('Artifacts:', work, flush=True)


if __name__ == '__main__':
    main()
