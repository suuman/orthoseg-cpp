import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.ml.preprocessing import InvalidImage
from app.ml.sam2_predictor import Sam2Predictor, parse_box, parse_boxes, sam2_input
from conftest import png


def test_prompted_route_and_health(config):
    app = create_app(config)
    seen = []
    app.state.sam2.load = lambda: None
    app.state.sam2.info = lambda: {"model_loaded": True, "model_version": "sam2_test",
                                   "architecture": "SAM2", "requires_prompts": True}

    def predict(image, boxes):
        seen.append((image.shape, boxes))
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[2:8, 3:9] = 1
        mask[10:17, 12:20] = 2
        return mask, "sam2_test", 12.5

    app.state.sam2.predict = predict
    with TestClient(app) as client:
        health = client.get('/health').json()
        assert health['sam2_model_loaded'] is True
        assert health['sam2_model_version'] == 'sam2_test'
        assert client.get('/model/info').json()['prompted_model']['requires_prompts'] is True
        response = client.post('/segment/prompted', files={'image': ('x.png', png(np.ones((24, 37), np.uint16)))},
                               data={'femur_box': '[2,1,15,12]', 'tibia_box': '[11,8,25,20]'})
        assert response.status_code == 200
        assert response.headers['content-type'] == 'image/png'
        assert response.headers['x-model-version'] == 'sam2_test'
        assert response.headers['x-image-width'] == '37'
        assert response.headers['x-image-height'] == '24'
        mask = np.asarray(Image.open(io.BytesIO(response.content)))
        assert mask.shape == (24, 37) and mask.dtype == np.uint8
        assert set(np.unique(mask)) == {0, 1, 2}
        assert seen == [((24, 37), {1: [2, 1, 15, 12], 2: [11, 8, 25, 20]})]
        bilateral = client.post('/segment/prompted',
                                files={'image': ('x.png', png(np.ones((24, 37), np.uint16)))},
                                data={'femur_box': '[[2,1,15,12],[18,2,30,19]]'})
        assert bilateral.status_code == 200
        assert seen[-1][1] == {1: [[2, 1, 15, 12], [18, 2, 30, 19]]}


def test_prompted_boxes_rejected(config):
    app = create_app(config)
    app.state.sam2.load = lambda: None
    image = png(np.ones((24, 37), np.uint8))
    with TestClient(app) as client:
        for data in ({}, {'femur_box': 'oops'}, {'femur_box': '[0,0,37,10]'},
                     {'femur_box': '[0,0,0,10]'}, {'femur_box': '[0.5,0,10,10]'},
                     {'femur_box': '[[1,1,5,5],[2,2,6,6],[3,3,7,7]]'}):
            response = client.post('/segment/prompted', files={'image': ('x.png', image)}, data=data)
            assert response.status_code == 422


def test_parse_box_bounds():
    assert parse_box([0, 0, 36, 23], (24, 37)).tolist() == [0, 0, 36, 23]
    for value in ([0, 0, 37, 23], [1, 1, 1, 2], [True, 0, 2, 2]):
        try:
            parse_box(value, (24, 37))
        except InvalidImage:
            pass
        else:
            raise AssertionError(f'Accepted invalid box: {value}')


def test_bilateral_boxes_and_resize(config):
    class FakeImagePredictor:
        def __init__(self):
            self.image_shape = None
            self.seen = []

        def set_image(self, image):
            self.image_shape = image.shape

        def predict(self, box, multimask_output):
            self.seen.append(box.tolist())
            mask = np.zeros((1, *self.image_shape[:2]), dtype=bool)
            mask[0, 100:200, 50:100] = True
            return mask, np.array([0.9]), None

    arr = np.ones((200, 100), dtype=np.uint16)
    rgb, sx, sy = sam2_input(arr, config)
    assert rgb.shape == (1024, 512, 3) and sx == 5.12 and sy == 5.12
    fake = FakeImagePredictor()
    predictor = Sam2Predictor(config)
    predictor._predictor, predictor._version = fake, 'fake'
    mask, version, _ = predictor.predict(arr, {1: [[10, 20, 30, 60], [45, 80, 75, 150]]})
    assert fake.image_shape == (1024, 512, 3)
    assert len(fake.seen) == 2
    assert np.allclose(fake.seen[0], [51.2, 102.4, 153.6, 307.2])
    assert mask.shape == arr.shape and set(np.unique(mask)) <= {0, 1} and version == 'fake'
    assert len(parse_boxes([10, 20, 30, 60], arr.shape)) == 1
    assert len(parse_boxes([[10, 20, 30, 60], [45, 80, 75, 150]], arr.shape)) == 2
