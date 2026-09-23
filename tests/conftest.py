import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.core.config import load_config


def png(arr):
    stream = io.BytesIO()
    Image.fromarray(arr).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(1)
    c = load_config()
    c["paths"] = {"data": str(tmp_path / "data"), "models": str(tmp_path / "models")}
    c["model"].update(channels=[4, 8], strides=[2], num_res_units=1)
    c["preprocessing"]["max_size"] = 32
    c["inference"]["device"] = "cpu"
    c["training"].update(epochs=1, batch_size=1, new_case_repeat_factor=1)
    c["training"]["augmentation"]["probability"] = 0
    return c


@pytest.fixture
def pair():
    x = np.arange(24 * 37, dtype=np.uint16).reshape(24, 37)
    y = np.zeros(x.shape, dtype=np.uint8)
    y[2:12, 4:30] = 1
    y[14:22, 6:32] = 2
    return png(x), png(y)
