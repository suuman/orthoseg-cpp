"""Exercise actual CLI scripts and HTTP service using temporary synthetic data."""
import io
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import yaml
from PIL import Image

from app.core.config import ROOT, load_config
from app.ml.store import CaseStore


def png(array):
    out = io.BytesIO()
    Image.fromarray(array).save(out, format='PNG')
    return out.getvalue()


def main():
    with tempfile.TemporaryDirectory(prefix='xray-smoke-') as temp:
        root = Path(temp)
        config = load_config()
        config['paths'] = {'data': str(root / 'data'), 'models': str(root / 'models')}
        config['model'].update(channels=[4, 8], strides=[2], num_res_units=1)
        config['preprocessing']['max_size'] = 32
        config['inference']['device'] = 'cpu'
        config['training'].update(epochs=1, batch_size=1, new_case_repeat_factor=1)
        config['training']['augmentation']['probability'] = 0.25
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            config['server']['port'] = sock.getsockname()[1]
        config_path = root / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config))
        image = png(np.arange(24*37, dtype=np.uint16).reshape(24, 37))
        label = np.zeros((24, 37), np.uint8)
        label[2:12, 3:30] = 1
        label[14:22, 4:32] = 2
        mask = png(label)
        store = CaseStore(config)
        store.submit(image, mask, case_id='smoke')
        for kind, data in [('images', png(np.arange(24*37, dtype=np.uint8).reshape(24, 37))), ('labels', mask)]:
            path = root / 'data' / 'validation' / kind / 'validation.png'
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        def run(script, *args):
            subprocess.run([str(ROOT / 'scripts' / script), *args, '--config', str(config_path)],
                           check=True, env=env, timeout=120)
        run('finetune.sh', '--from-scratch', '--dry-run')
        run('finetune.sh', '--from-scratch')
        version = store.status()['latest_candidate_version']
        run('promote_candidate.sh', version)
        run('finetune.sh', '--dry-run')
        with (root / 'server.log').open('w+') as log:
            process = subprocess.Popen([str(ROOT / 'scripts/run_server.sh'), '--config', str(config_path)],
                                       env=env, stdout=log, stderr=log)
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{config['server']['port']}", timeout=20,
                                  trust_env=False) as client:
                    for _ in range(100):
                        if process.poll() is not None:
                            raise RuntimeError('Server exited before readiness')
                        try:
                            health = client.get('/health')
                            if health.status_code == 200:
                                break
                        except httpx.ConnectError:
                            pass
                        time.sleep(0.1)
                    else:
                        raise RuntimeError('Server startup timed out')
                    assert health.json()['model_loaded']
                    response = client.post('/segment', files={'image': ('x.png', image)})
                    response.raise_for_status()
                    result = np.asarray(Image.open(io.BytesIO(response.content)))
                    assert result.shape == (24, 37) and result.dtype == np.uint8
                    assert set(np.unique(result)) <= {0, 1, 2}
                    assert response.headers['x-model-version'] == version
                    response = client.post('/training/cases', files={
                        'image': ('x.png', image), 'mask': ('mask.png', mask)}, data={'case_id': 'smoke'})
                    response.raise_for_status()
                    assert response.json()['training_case_count'] == 1
                    assert client.get('/training/status').json()['new_cases_since_last_training'] == 1
                    print('HTTP smoke passed: health, real segmentation (37×24 uint8), corrected case, training status.')
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read())
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
