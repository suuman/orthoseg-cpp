# Implementation record

The initial repository contained only `design.md` and empty protected tool directories. No existing application, Python environment, tests, model weights, dataset, or repository conventions were available to reuse. `design.md` was not modified.

## Files added

- `.gitignore`
- `README.md`
- `IMPLEMENTATION.md`
- `requirements.txt`
- `configs/default.yaml`
- `app/__init__.py`
- `app/main.py`
- `app/serve.py`
- `app/api/__init__.py`
- `app/api/routes.py`
- `app/core/__init__.py`
- `app/core/config.py`
- `app/core/logging.py`
- `app/core/paths.py`
- `app/ml/__init__.py`
- `app/ml/checkpoint.py`
- `app/ml/dataset.py`
- `app/ml/metrics.py`
- `app/ml/model.py`
- `app/ml/predictor.py`
- `app/ml/preprocessing.py`
- `app/ml/store.py`
- `app/ml/trainer.py`
- `scripts/common.sh`
- `scripts/run_server.sh`
- `scripts/finetune.sh`
- `scripts/promote_candidate.sh`
- `scripts/smoke_test.sh`
- `scripts/smoke_test.py`
- `tests/conftest.py`
- `tests/test_api.py`
- `tests/test_preprocessing.py`
- `tests/test_training.py`

Generated, ignored local artifacts: `.venv/`, Python/pytest caches, and empty data/model working directories. Synthetic test data and checkpoints live only in temporary directories.

## Verification

- Unit/integration tests: 27 passed.
- Real HTTP smoke: server startup, health, MONAI inference, original 37×24 output dimensions, uint8 canonical labels, corrected-case submission, training status.
- Actual shell workflow: from-scratch dry run, one-epoch synthetic training, candidate promotion, from-production dry run.
- Unit tests additionally exercise actual fine-tuning from production and preservation of archived releases and validation data.
- Python compile checks and shell syntax checks passed. No pre-existing lint or type-check configuration was present.
- Third-party deprecation warnings from MONAI/PyTorch and the Starlette/httpx test client remain; tests pass.

## Assumptions and limits

- Trusted local Linux workstation and local filesystem with atomic rename/symlink and `flock` support.
- CUDA auto-selection is implemented, but this workstation's NVIDIA driver was unavailable; execution was verified on CPU.
- No trained anatomical model or clinical data was supplied. Real use requires curated training/validation data and explicit training/promotion of suitable weights. Synthetic smoke weights are not deployed.
- Entirely offline runtime; dependency installation is a separate setup operation.
- PNG byte hashes identify exact duplicate originals. Patient-level validation separation must be curated externally.
- Masks are checked structurally and for canonical class IDs; anatomical correctness requires annotator approval. Background-only masks are accepted.
- Server restart applies a promoted model. No HTTP reload endpoint is exposed.
- HD95 is optional in the design and is not implemented; Dice and IoU are provided.
