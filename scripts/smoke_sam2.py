"""Exercise the locally installed SAM2 checkpoint through the HTTP API in process."""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import load_config
from app.main import create_app


def main():
    config = load_config()
    image = np.zeros((64, 64), dtype=np.uint16)
    image[10:54, 10:54] = 32000
    image[20:44, 20:44] = 50000
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    with TestClient(create_app(config)) as client:
        health = client.get("/health").json()
        assert health["sam2_model_loaded"], health
        response = client.post("/segment/prompted",
                               files={"image": ("synthetic.png", stream.getvalue(), "image/png")},
                               data={"femur_box": "[[8,8,30,55],[31,8,55,55]]",
                                     "tibia_box": "[18,18,45,45]"})
        assert response.status_code == 200, response.text
        mask = np.asarray(Image.open(io.BytesIO(response.content)))
        assert mask.shape == image.shape and mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1, 2}
        assert response.headers["x-model-version"] == health["sam2_model_version"]
        print(f"SAM2 API ready: {health['sam2_model_version']}; mask labels {sorted(np.unique(mask).tolist())}")


if __name__ == "__main__":
    main()
