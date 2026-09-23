import numpy as np
import pytest

from app.ml.preprocessing import decode_png, prepare, prepare_mask, restore_mask
from conftest import png


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_depth(config, dtype):
    arr = np.arange(100, dtype=dtype).reshape(10, 10) * (500 if dtype == np.uint16 else 1)
    decoded = decode_png(png(arr), config['limits'])
    np.testing.assert_array_equal(decoded, arr)


@pytest.mark.parametrize('shape', [(11, 79), (79, 11), (1, 17), (24, 37), (32, 32)])
def test_geometry(config, shape):
    x = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    prepared, g = prepare(x, config)
    rh, rw = g.resized
    scale = min(1, 32/max(shape))
    assert abs(rh-shape[0]*scale) <= 0.5
    assert abs(rw-shape[1]*scale) <= 0.5
    mask = np.zeros(shape, np.uint8)
    mask[:, shape[1]//2:] = 2
    transformed = prepare_mask(mask, g)[0].astype(np.uint8)
    restored = restore_mask(transformed, g)
    assert restored.shape == shape
    assert set(np.unique(restored)) <= {0, 2}
    assert prepared.dtype == np.float32
    assert prepared.min() >= 0 and prepared.max() <= 1
    if scale == 1:
        np.testing.assert_array_equal(restored, mask)


def test_rgb(config):
    arr = np.zeros((20, 30, 3), np.uint8)
    arr[..., 0] = 255
    assert np.all(decode_png(png(arr), config['limits']) == 76)


def test_inverse_matches_nearest_neighbor(config):
    from PIL import Image
    _, g = prepare(np.zeros((53, 97), np.uint16), config)
    mask = (np.indices(g.padded).sum(axis=0) % 3).astype(np.uint8)
    restored = restore_mask(mask, g)
    expected = np.asarray(Image.fromarray(mask[:g.resized[0], :g.resized[1]]).resize(
        (97, 53), Image.Resampling.NEAREST))
    np.testing.assert_array_equal(restored, expected)
