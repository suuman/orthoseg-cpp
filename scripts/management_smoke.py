"""Optional real Qt-to-backend management check using isolated synthetic data."""
import argparse
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

from app.core.config import ROOT, load_config
from app.ml.checkpoint import promote, production_path
from app.ml.store import CaseStore
from app.ml.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ui-test', type=Path, required=True)
    parser.add_argument('--annotation-test', type=Path)
    parser.add_argument('--screenshot', default='/tmp/orthoseg-management.png')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='management-smoke-') as directory:
        root = Path(directory)
        config = load_config()
        config['paths'] = {'data': str(root/'data'), 'models': str(root/'models')}
        config['model'].update(channels=[4,8], strides=[2], num_res_units=1)
        config['preprocessing']['max_size'] = 32
        config['inference']['device'] = 'cpu'
        config['training'].update(epochs=1, batch_size=1, new_case_repeat_factor=1)
        config['training']['augmentation']['probability'] = 0
        config['management']['enabled'] = True
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        config['server']['port'] = port
        source = root/'original.png'
        Image.fromarray(np.arange(24*37, dtype=np.uint16).reshape(24,37)*50).save(source)
        label = np.zeros((24,37), dtype=np.uint8)
        label[2:12,3:30]=1
        label[14:22,4:32]=2
        label_file = root/'mask.png'
        Image.fromarray(label).save(label_file)
        store = CaseStore(config)
        store.submit(source.read_bytes(), label_file.read_bytes())
        for folder in ('images','labels'):
            (root/'data/validation'/folder).mkdir(parents=True)
        Image.fromarray(np.arange(24*37, dtype=np.uint8).reshape(24,37)).save(root/'data/validation/images/v.png')
        Image.fromarray(label).save(root/'data/validation/labels/v.png')
        parent = train(config, source='scratch')
        promote(config, parent)
        prior_release = production_path(config).resolve()
        config_file = root/'config.yaml'
        config_file.write_text(yaml.safe_dump(config))
        env = {**os.environ, 'XRAY_PYTHON': sys.executable, 'OMP_NUM_THREADS':'1',
               'QT_QPA_PLATFORM':'offscreen', 'MONAI_BACKEND_URL':f'http://127.0.0.1:{port}'}
        with (root/'server.log').open('w+') as log:
            server = subprocess.Popen([str(ROOT/'scripts/run_server.sh'), '--config', str(config_file)],
                                      stdout=log, stderr=log, env=env)
            try:
                with httpx.Client(base_url=env['MONAI_BACKEND_URL'], trust_env=False) as client:
                    for _ in range(150):
                        if server.poll() is not None: raise RuntimeError('Backend exited')
                        try:
                            if client.get('/health').json()['model_loaded']: break
                        except httpx.ConnectError: pass
                        time.sleep(.1)
                    else: raise RuntimeError('Backend startup timed out')
                    if args.annotation_test:
                        subprocess.run([str(args.annotation_test.resolve()), '--live', str(source), str(root/'export.png')],
                                       check=True, timeout=60, env=env)
                        assert not (store.root/'jobs/latest.json').exists()
                        record = store.records()[0]
                        expected = np.asarray(Image.open(root/'export.png.canonical.png'))
                        np.testing.assert_array_equal(np.asarray(Image.open(store.root/record['mask'])), expected)
                        assert (store.root/record['image']).read_bytes() == source.read_bytes()
                    subprocess.run([str(args.ui_test.resolve()), '--live', args.screenshot],
                                   check=True, timeout=90, env=env)
                    state = client.get('/management/status').json()
                    assert state['job']['status']=='completed'
                    assert state['production']['version']==state['candidate']['version']!=parent
                    assert state['loaded_model_version']==parent and state['restart_required']
                    assert prior_release.exists()
                    image = client.post('/segment', files={'image': ('x.png',source.read_bytes())})
                    assert image.status_code==200 and image.headers['x-model-version']==parent
                    print('LIVE MANAGEMENT PASS: separate annotation upload, explicit Qt training, completed candidate, '
                          'confirmed Qt promotion, archive retained, original loaded inference unaffected.')
            except Exception:
                log.flush();log.seek(0);print(log.read());raise
            finally:
                server.terminate()
                try: server.wait(timeout=10)
                except subprocess.TimeoutExpired: server.kill();server.wait()


if __name__=='__main__':
    main()
