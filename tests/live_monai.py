"""Optional end-to-end check. Run with the MONAI backend's Python environment.
Creates synthetic weights/data in a temporary directory, never in production.
"""
import argparse
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
from PIL import Image
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    backend = args.backend.resolve()
    binary = args.binary.resolve()
    sys.path.insert(0, str(backend))
    from app.core.config import load_config, LABELS
    from app.ml.model import build_model
    from app.ml.checkpoint import save_checkpoint
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='orthoseg-monai-live-') as directory:
        root = Path(directory)
        config = load_config()
        config['paths'] = {'data': str(root/'data'), 'models': str(root/'models')}
        config['model'].update(channels=[4, 8], strides=[2], num_res_units=1)
        config['preprocessing']['max_size'] = 32
        config['inference']['device'] = 'cpu'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        config['server']['port'] = port
        config_path = root/'config.yaml'
        config_path.write_text(yaml.safe_dump(config))
        save_checkpoint(root/'models/production/model.pt', build_model(config),
                        {'version': 'synthetic_integration_only', 'labels': LABELS, 'config': config})
        source = root/'original.png'
        Image.fromarray(np.arange(24*37, dtype=np.uint16).reshape(24, 37)*50).save(source)
        env = {**os.environ, 'XRAY_PYTHON': sys.executable, 'OMP_NUM_THREADS': '1',
               'MONAI_BACKEND_URL': f'http://127.0.0.1:{port}', 'QT_QPA_PLATFORM': 'offscreen'}
        with (root/'server.log').open('w+') as log:
            server = subprocess.Popen([str(backend/'scripts/run_server.sh'), '--config', str(config_path)],
                                      env=env, stdout=log, stderr=log)
            try:
                with httpx.Client(base_url=env['MONAI_BACKEND_URL'], trust_env=False) as client:
                    for _ in range(150):
                        if server.poll() is not None:
                            raise RuntimeError('Backend exited before startup')
                        try:
                            if client.get('/health').json()['model_loaded']:
                                break
                        except httpx.ConnectError:
                            pass
                        time.sleep(.1)
                    else:
                        raise RuntimeError('Backend startup timed out')
                    output = root/'export.png'
                    subprocess.run([str(binary), '--live', str(source), str(output)], env=env,
                                   check=True, timeout=210)
                    records = list((root/'data/training/metadata').glob('*.json'))
                    assert len(records) == 1
                    record = json.loads(records[0].read_text())
                    training = root/'data/training'
                    assert (training/record['image']).read_bytes() == source.read_bytes()
                    actual = np.asarray(Image.open(training/record['mask']))
                    expected = np.asarray(Image.open(str(output)+'.canonical.png'))
                    np.testing.assert_array_equal(actual, expected)
                    assert actual.dtype == np.uint8 and actual.shape == (24,37)
                    assert actual[4,4] == 0 and actual[15,15] == 2 and actual[20,20] == 1
                    assert record['model_version'] == 'synthetic_integration_only'
                    assert np.asarray(Image.open(output)).shape == (24,37,3)
                    status = client.get('/training/status').json()
                    assert status['total_approved_cases'] == 1
                    assert status['last_training_time'] is None and status['latest_candidate_version'] is None
                    assert not (training/'training_state.json').exists()
                    print('LIVE PASS: original 16-bit bytes, current brush/erase/AI-apply edits, colored export, '
                          'canonical training PNG, model version; no training started.')
            except Exception:
                log.flush(); log.seek(0); print(log.read())
                raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait()


if __name__ == '__main__':
    main()
