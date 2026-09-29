import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.ml.preprocessing import InvalidImage, decode_png, encode_mask, exclusive_mask, mask_statistics
from app.ml.sam2_predictor import Sam2Predictor
from app.ml.store import CaseStore
from conftest import png


def test_independent_boxes_and_mask_prompt(config):
    class FakePredictor:
        def set_image(self, image):
            self.shape = image.shape[:2]
            self.prompts = []
        def predict(self, **kwargs):
            self.prompts.append(kwargs)
            mask = np.full((1, *self.shape), -10., np.float32)
            mask[:, 100:900, 300:700] = 10
            return mask, np.array([0.9]), None
    engine = Sam2Predictor(config)
    fake = FakePredictor()
    engine._predictor, engine._version = fake, 'independent'
    raw = np.zeros((100, 50), np.uint16)
    predicted, _, _ = engine.predict(raw, {1: [5, 5, 40, 90], 2: [5, 5, 40, 90]})
    assert predicted.shape == (100, 50, 3)
    stats = mask_statistics(predicted)
    assert stats['femur_pixels'] == stats['tibia_pixels'] == stats['overlap_pixels'] > 0
    refined, _, _ = engine.predict(raw, {}, predicted)
    assert mask_statistics(refined)['overlap_pixels'] > 0
    assert len(fake.prompts) == 2
    for prompt in fake.prompts:
        assert prompt['mask_input'].shape == (1, 256, 256)
        assert set(np.unique(prompt['mask_input'])) <= {-10, 10}
        assert prompt['return_logits']


def test_pool_roundtrip_and_statistics(config, pair):
    mask = np.zeros((24, 37, 3), np.uint8)
    mask[3:12, 5:19, 0] = 1
    mask[6:16, 8:24, 1] = 2
    store = CaseStore(config)
    first = store.submit(pair[0], png(mask), annotator='test', model_version='base')
    assert first['overlap_pixels'] == 6 * 11
    record = store.records()[0]
    restored = decode_png((store.root / record['mask']).read_bytes(), config['limits'], mask=True)
    np.testing.assert_array_equal(restored, mask)
    assert record['annotator'] == 'test' and record['model_version'] == 'base'
    assert record['mask_encoding'] == 'rgb_discrete'
    with pytest.raises(InvalidImage, match='MedSAM2'):
        exclusive_mask(restored)
    # Legacy files remain readable and retain their original bytes and metadata.
    store.submit(pair[0], pair[1], case_id=record['case_id'])
    revised = store.records()[0]
    assert revised['previous_revision'] == record['revision']
    assert (store.root / revised['mask']).read_bytes() == pair[1]


@pytest.mark.parametrize('channel,value', [(0, 2), (1, 1), (2, 3)])
def test_malformed_rgb_rejected(config, pair, channel, value):
    mask = np.zeros((24, 37, 3), np.uint8)
    mask[4, 4, channel] = value
    with pytest.raises(InvalidImage):
        CaseStore(config).submit(pair[0], png(mask))
    assert CaseStore(config).records() == []


def test_rgb_prompt_api_and_metadata(config, pair):
    app = create_app(config)
    app.state.sam2.load = lambda: None
    mask = np.zeros((24, 37, 3), np.uint8)
    mask[3:10, 5:15] = (1, 2, 0)
    def predict(image, boxes, prompt=None):
        np.testing.assert_array_equal(prompt, mask)
        assert boxes == {}
        return prompt, 'medsam2_candidate', 1.5
    app.state.sam2.predict = predict
    with TestClient(app) as api:
        response = api.post('/segment/prompted', files={
            'image': ('x.png', pair[0]), 'mask': ('prompt.png', png(mask))})
        assert response.status_code == 200, response.text
        assert response.headers['x-mask-encoding'] == 'rgb_discrete'
        assert int(response.headers['x-overlap-pixels']) == 70
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(response.content))), mask)
