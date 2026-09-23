"""Exercise the installed nnUNet v2 checkpoint through the HTTP API in process."""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import load_config
from app.main import create_app


def main():
    image = np.zeros((512, 128), dtype=np.uint8)
    image[50:460, 20:105] = 120
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    with TestClient(create_app(load_config())) as client:
        health = client.get('/health').json()
        assert health['nnunet_model_loaded'], health
        response = client.post('/segment/nnunet',
                               files={'image': ('synthetic.png', stream.getvalue(), 'image/png')})
        assert response.status_code == 200, response.text
        mask = np.asarray(Image.open(io.BytesIO(response.content)))
        assert mask.shape == image.shape and mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1, 2}
        assert response.headers['x-model-version'] == health['nnunet_model_version']
        print(f"nnUNet v2 API ready: {health['nnunet_model_version']}; mask labels {sorted(np.unique(mask).tolist())}")


if __name__ == '__main__':
    main()
