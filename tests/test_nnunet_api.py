import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.ml.nnunet_predictor import prepare_nnunet_image
from app.ml.preprocessing import ImageTooLarge
from conftest import png


def test_nnunet_route_and_health(config):
    app = create_app(config)
    seen = []
    app.state.nnunet.load = lambda: None
    app.state.nnunet.info = lambda: {"model_loaded": True, "model_version": "nnunet_test",
                                     "architecture": "nnUNet v2", "input_height": 2048}

    def predict(arr):
        seen.append(arr.shape)
        mask = np.zeros(arr.shape, dtype=np.uint8)
        mask[2:8, 3:9] = 1
        mask[10:17, 12:20] = 2
        return mask, "nnunet_test", 7.5

    app.state.nnunet.predict = predict
    with TestClient(app) as client:
        health = client.get('/health').json()
        assert health['nnunet_model_loaded'] is True
        assert health['nnunet_model_version'] == 'nnunet_test'
        assert client.get('/model/info').json()['nnunet_model']['input_height'] == 2048
        response = client.post('/segment/nnunet', files={'image': ('x.png', png(np.ones((24, 37), np.uint16)))})
        assert response.status_code == 200
        assert response.headers['content-type'] == 'image/png'
        assert response.headers['x-model-version'] == 'nnunet_test'
        assert response.headers['x-image-width'] == '37'
        assert response.headers['x-image-height'] == '24'
        mask = np.asarray(Image.open(io.BytesIO(response.content)))
        assert mask.shape == (24, 37) and mask.dtype == np.uint8
        assert set(np.unique(mask)) == {0, 1, 2}
        assert seen == [(24, 37)]


def test_nnunet_resize_and_limits():
    image = np.full((512, 128), 32768, dtype=np.uint16)
    data, original = prepare_nnunet_image(image, 25_000_000)
    assert data.shape == (1, 1, 2048, 512)
    assert original == (512, 128)
    assert data.dtype == np.float32 and np.all(data == 128)
    try:
        prepare_nnunet_image(np.zeros((10, 1000), dtype=np.uint8), 25_000_000)
    except ImageTooLarge:
        pass
    else:
        raise AssertionError('Oversized resized image accepted')


def test_nnunet_missing_model(config):
    app = create_app(config)
    app.state.nnunet.load = lambda: None
    with TestClient(app) as client:
        image = png(np.ones((24, 37), np.uint8))
        assert client.post('/segment/nnunet', files={'image': ('x.png', image)}).status_code == 503


def test_region_probabilities_preserve_overlap():
    import pytest
    from app.ml.nnunet_region import region_probabilities_to_rgb
    probabilities = np.array([[[0.1, 0.9, 0.1, 0.9]], [[0.1, 0.1, 0.9, 0.9]]])
    mask = region_probabilities_to_rgb(probabilities)
    assert mask.tolist() == [[[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]]]
    with pytest.raises(ValueError):
        region_probabilities_to_rgb(np.zeros((3, 4, 5)))


def test_region_route_and_health(config):
    app = create_app(config)
    app.state.nnunet_region.load = lambda: None
    app.state.nnunet_region.info = lambda: {"model_loaded": True, "model_version": "region_test"}
    def predict(arr):
        mask = np.zeros((*arr.shape, 3), np.uint8)
        mask[2:8, 3:9] = (1, 2, 0)
        return mask, "region_test", 1.0
    app.state.nnunet_region.predict = predict
    with TestClient(app) as client:
        assert client.get('/health').json()['nnunet_region_model_loaded']
        response = client.post('/segment/nnunet_region', files={'image': ('x.png', png(np.ones((24, 37), np.uint8)))})
        assert response.status_code == 200
        mask = np.asarray(Image.open(io.BytesIO(response.content)))
        assert mask.shape == (24, 37, 3)
        assert mask[4, 5].tolist() == [1, 2, 0]
        assert response.headers['x-model-version'] == 'region_test'
