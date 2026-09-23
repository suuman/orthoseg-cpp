import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.ml.predictor import Predictor
from conftest import png


class FakePredictor(Predictor):
    def load(self):
        pass

    def predict(self, arr):
        return np.ones(arr.shape, np.uint8), "test_v001", 1.0


@pytest.fixture
def client(config):
    with TestClient(create_app(config, FakePredictor(config))) as c:
        yield c


@pytest.mark.parametrize('mode', ['u8', 'u16', 'rgb'])
def test_segment(client, mode):
    shape = (43, 71, 3) if mode == 'rgb' else (43, 71)
    arr = np.ones(shape, np.uint16 if mode == 'u16' else np.uint8)
    response = client.post('/segment', files={'image': ('x.png', png(arr))})
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    mask = np.asarray(Image.open(io.BytesIO(response.content)))
    assert mask.shape == (43, 71) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1, 2}
    assert response.headers['x-model-version'] == 'test_v001'


def test_errors(config, pair, client):
    assert client.post('/segment', files={'image': ('x.png', b'bad')}).status_code == 422
    with TestClient(create_app(config)) as c:
        assert c.get('/health').json()['model_loaded'] is False
        assert c.post('/segment', files={'image': ('x.png', pair[0])}).status_code == 503
    config['limits']['max_file_bytes'] = 2
    with TestClient(create_app(config)) as c:
        assert c.post('/segment', files={'image': ('x.png', pair[0])}).status_code == 413


def test_submit_revision(client, pair):
    files = {'image': ('x.png', pair[0]), 'mask': ('y.png', pair[1])}
    first = client.post('/training/cases', files=files, data={'case_id': 'case1'})
    assert first.status_code == 200
    store = client.app.state.store
    original = store.records()[0]
    for identifier in ('case1', 'another_alias'):
        result = client.post('/training/cases', files=files, data={'case_id': identifier}).json()
        assert result['training_case_count'] == 1 and result['case_id'] == 'case1'
    assert store.records()[0]['revision'] != original['revision']
    assert (store.root / original['mask']).exists()
    assert (store.root / original['image']).read_bytes() == pair[0]
    assert client.get('/training/status').json()['new_cases_since_last_training'] == 1
    assert not list(__import__('pathlib').Path(client.app.state.config['paths']['models']).glob('candidates/*'))


@pytest.mark.parametrize('mask', [np.zeros((1, 1), np.uint8), np.full((24, 37), 3, np.uint8),
                                 np.zeros((24, 37, 3), np.uint8)])
def test_reject_masks(client, pair, mask):
    response = client.post('/training/cases', files={'image': ('x.png', pair[0]), 'mask': ('y.png', png(mask))})
    assert response.status_code == 422
    assert client.app.state.store.records() == []


def test_identifier_conflict(client, pair):
    files = {'image': ('x.png', pair[0]), 'mask': ('y.png', pair[1])}
    assert client.post('/training/cases', files=files, data={'case_id': '../escape'}).status_code == 422
    assert client.post('/training/cases', files=files, data={'case_id': 'same'}).status_code == 200
    files['image'] = ('x.png', png(np.zeros((24, 37), np.uint8)))
    assert client.post('/training/cases', files=files, data={'case_id': 'same'}).status_code == 422


def test_model_corruption_is_server_error(client, pair):
    client.app.state.predictor.predict = lambda arr: (np.full(arr.shape, 3, np.uint8), 'bad', 1.)
    response = client.post('/segment', files={'image': ('x.png', pair[0])})
    assert response.status_code == 500


def test_pixel_limit(config, pair):
    config['limits']['max_pixels'] = 10
    with TestClient(create_app(config)) as c:
        assert c.post('/segment', files={'image': ('x.png', pair[0])}).status_code == 413


def test_storage_failure_returns_507(client, pair, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(client.app.state.store, 'submit', fail)
    response = client.post('/training/cases', files={'image': ('x.png', pair[0]), 'mask': ('y.png', pair[1])})
    assert response.status_code == 507
