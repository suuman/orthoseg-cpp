import hashlib
import json
from pathlib import Path
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from app.core.config import load_config
from app.core.paths import file_lock, write_json
from app.main import create_app
from app.management.jobs import ACTIVE, Management
from app.ml.checkpoint import production_path, promote
from app.ml.predictor import Predictor
from app.ml.trainer import train
from test_training import populate


@pytest.mark.parametrize('threads', [True, 1.5, '2', 0, 33])
def test_invalid_admin_cpu_thread_config_is_rejected(tmp_path, threads):
    override = tmp_path / 'config.yaml'
    override.write_text(yaml.safe_dump({'management': {'cpu_threads': threads}}))
    with pytest.raises(ValueError, match='management.cpu_threads'):
        load_config(override)


def client(config, address='127.0.0.1'):
    return TestClient(create_app(config), base_url='http://127.0.0.1', client=(address, 12345))


def completed(api, identifier):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        response = api.get('/training/jobs/' + identifier)
        assert response.status_code == 200
        value = response.json()
        if value['status'] not in ACTIVE:
            return value
        time.sleep(.05)
    pytest.fail('Administrative job timed out')


def bootstrap(config, pair):
    populate(config, pair)
    version = train(config, source='scratch')
    promote(config, version)
    return version


def test_admin_access_is_opt_in_and_local(config):
    with client(config) as api:
        assert api.get('/management/status').status_code == 403
    config['management']['enabled'] = True
    with client(config, '192.168.1.50') as api:
        assert api.get('/management/status').status_code == 403
        assert api.post('/training/start', json={}).status_code == 403
    with client(config) as api:
        assert api.get('/management/status').status_code == 200
        assert api.get('/management/status', headers={'Origin': 'http://evil.test'}).status_code == 403
        assert api.get('/management/status', headers={'Host': 'evil.test'}).status_code == 403
        assert api.post('/training/start', json={'path': '/tmp/untrusted'}).status_code == 422
        assert api.post('/training/start', json={}).status_code == 422
        assert api.get('/training/jobs/not-a-job').status_code == 404


def test_annotation_does_not_start_jobs(config, pair, monkeypatch):
    config['management']['enabled'] = True
    def forbidden(*args, **kwargs):
        pytest.fail('Annotation/refresh attempted to launch a process')
    monkeypatch.setattr('app.management.jobs.subprocess.Popen', forbidden)
    with client(config) as api:
        response = api.post('/training/cases', files={'image': ('x.png', pair[0]), 'mask': ('m.png', pair[1])})
        assert response.status_code == 200
        state = api.get('/management/status').json()
        assert state['dataset']['total_approved_cases'] == 1
        assert state['job'] is None and not state['training_active']
        assert state['production'] is None and state['candidate'] is None
        assert not (Path(config['paths']['data'])/'training/jobs').exists()


def test_actual_async_training_manual_promotion_and_reload_status(config, pair):
    first = bootstrap(config, pair)
    config['management']['enabled'] = True
    original = production_path(config).resolve()
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    with client(config) as api:
        state = api.get('/management/status').json()
        assert state['loaded_model_version'] == first
        assert state['production']['version'] == first
        assert state['production']['training_case_count'] == 1
        assert state['dataset']['validation_case_count'] == 1
        assert state['candidate']['valid']
        response = api.post('/training/start', json={})
        assert response.status_code == 202
        identifier = response.json()['id']
        assert api.post('/training/start', json={}).status_code == 409
        # Inference keeps using its original production version during CPU training.
        segment = api.post('/segment', files={'image': ('x.png', pair[0])})
        assert segment.status_code == 200 and segment.headers['x-model-version'] == first
        job = completed(api, identifier)
        assert job['status'] == 'completed', job
        assert job['epoch'] == 1 and job['best_validation_dice'] is not None
        assert job['recent_log'] and len(job['recent_log']) <= 80
        assert production_path(config).resolve() == original
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
        state = api.get('/management/status').json()
        second = state['candidate']['version']
        assert second != first and state['candidate']['parent_model_version'] == first
        assert 'femur_dice' in state['candidate']['validation_metrics']
        assert api.post('/models/unknown/promote', json={}).status_code == 422
        assert production_path(config).resolve() == original
        assert api.post('/models/' + second + '/promote', json={}).status_code == 200
        state = api.get('/management/status').json()
        assert state['production']['version'] == second
        assert state['loaded_model_version'] == first and state['restart_required']
        assert original.exists()
        segment = api.post('/segment', files={'image': ('x.png', pair[0])})
        assert segment.headers['x-model-version'] == first
    with client(config) as restarted:
        state = restarted.get('/management/status').json()
        assert state['loaded_model_version'] == second and not state['restart_required']


def test_failed_training_preserves_production_and_inference(config, pair):
    version = bootstrap(config, pair)
    config['management']['enabled'] = True
    # Corrupt validation intentionally: all dataset checks still come from the trainer.
    label = Path(config['paths']['data'])/'validation/labels/v.png'
    label.write_bytes(b'not png')
    before = production_path(config).read_bytes()
    with client(config) as api:
        job = api.post('/training/start', json={}).json()
        outcome = completed(api, job['id'])
        assert outcome['status'] == 'failed' and outcome['error']
        assert production_path(config).read_bytes() == before
        assert api.get('/health').json()['model_version'] == version


def test_cli_lock_prevents_start_and_promotion(config, pair):
    version = bootstrap(config, pair)
    config['management']['enabled'] = True
    lock = Path(config['paths']['data'])/'training/.training.lock'
    with client(config) as api, file_lock(lock):
        assert api.get('/management/status').json()['training_active']
        assert api.post('/training/start', json={}).status_code == 409
        assert api.post('/models/' + version + '/promote', json={}).status_code == 409


def test_invalid_candidate_disabled_and_failed_promotion_preserves_disk(config, pair):
    version = bootstrap(config, pair)
    config['management']['enabled'] = True
    original = production_path(config).resolve()
    with client(config) as api:
        assert api.get('/management/status').json()['candidate']['valid']
        meta = Path(config['paths']['models'])/'candidates'/version/'metadata.json'
        meta.write_text('{}')
        assert not api.get('/management/status').json()['candidate']['valid']
        assert api.post('/models/' + version + '/promote', json={}).status_code == 422
        assert production_path(config).resolve() == original


def test_stale_job_recovered_and_log_tail_bounded(config):
    management = Management(config, Predictor(config))
    identifier = 'a' * 32
    root = management.root/identifier
    write_json(root/'status.json', {'id': identifier, 'status': 'training'})
    write_json(root/'process.json', {'pid': 99999999, 'identity': 'missing'})
    (root/'training.log').write_text('progress\n'*10000)
    state = management.job(identifier)
    assert state['status'] == 'failed'
    assert len(state['recent_log']) == 80


def test_removed_completion_marker_invalidates_cached_candidate(config, pair):
    version = bootstrap(config, pair)
    config['management']['enabled'] = True
    with client(config) as api:
        assert api.get('/management/status').json()['candidate']['valid']
        marker = Path(config['paths']['models']) / 'candidates' / version / 'completed.json'
        marker.unlink()
        assert not api.get('/management/status').json()['candidate']['valid']
        assert api.post('/models/' + version + '/promote', json={}).status_code == 422


def test_management_case_list_labels_pagination_and_family_state(config, pair):
    import numpy as np
    from conftest import png
    from app.ml.store import CaseStore
    store = CaseStore(config)
    mask = np.zeros((24, 37, 3), dtype=np.uint8)
    mask[2:10, 3:12] = (1, 2, 0)
    store.submit(pair[0], png(mask), case_id='a', original_filename='knee.png')
    record = store.records()[0]
    write_json(store.state_path, {'revisions': {'a': record['revision']},
        'last_training_time': None, 'latest_candidate_version': 'unet_done'})
    store.submit(png(np.zeros((24, 37), np.uint8)), png(np.zeros((24, 37), np.uint8)),
                 case_id='b', original_filename='empty.png')
    with client(config) as api:
        assert api.get('/management/cases').status_code == 403
    config['management']['enabled'] = True
    with client(config) as api:
        result = api.get('/management/cases', params={'limit': 1}).json()
        assert result['total'] == 2 and len(result['items']) == 1
        item = result['items'][0]
        assert item['filename'] == 'knee.png' and item['labels'] == ['Femur', 'Tibia']
        assert item['overlap_pixels'] == 72 and item['status'] == 'incorporated'
        assert 'image' not in item and 'mask' not in item
        assert api.get('/management/cases?model=medsam2&limit=1').json()['items'][0]['status'] == 'pending'
        page2 = api.get('/management/cases?offset=1&limit=1').json()['items'][0]
        assert page2['filename'] == 'empty.png' and page2['labels'] == ['Background only']
        assert api.get('/management/cases?offset=99').json()['items'] == []
        for query in ('offset=-1', 'limit=0', 'limit=101', 'model=unknown'):
            assert api.get('/management/cases?' + query).status_code == 422
        assert api.get('/management/cases', headers={'Origin': 'http://example.com'}).status_code == 403


def test_legacy_case_labels_are_read_without_changing_records(config, pair):
    from app.ml.store import CaseStore
    store = CaseStore(config)
    store.submit(*pair, case_id='legacy')
    record = store.records()[0]
    for key in ('femur_pixels', 'tibia_pixels', 'overlap_pixels'):
        record.pop(key)
    manifest = store.root / 'metadata/legacy.json'
    write_json(manifest, record)
    before = manifest.read_bytes()
    item = store.case_page()['items'][0]
    assert item['filename'] == 'legacy' and item['labels'] == ['Femur', 'Tibia']
    assert manifest.read_bytes() == before
