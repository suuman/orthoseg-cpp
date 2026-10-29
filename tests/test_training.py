import hashlib
from pathlib import Path

import numpy as np
import pytest

from app.ml.checkpoint import production_path, promote
from app.ml.dataset import snapshot
from app.ml.predictor import Predictor
from app.ml.preprocessing import decode_png
from app.ml.store import CaseStore
from app.ml.trainer import train
from conftest import png


def populate(config, pair):
    store = CaseStore(config)
    store.submit(*pair, case_id='case1')
    root = Path(config['paths']['data']) / 'validation'
    for name in ('images', 'labels'):
        (root / name).mkdir(parents=True)
    (root / 'images/v.png').write_bytes(png(np.arange(24*37, dtype=np.uint8).reshape(24, 37)))
    (root / 'labels/v.png').write_bytes(pair[1])
    return store, root


def test_training_and_promotion(config, pair):
    store, root = populate(config, pair)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.png')}
    with pytest.raises(ValueError, match='No production'):
        train(config, dry_run=True)
    assert train(config, source='scratch', dry_run=True) is None
    version = train(config, source='scratch')
    assert not production_path(config).exists()
    assert store.status()['new_cases_since_last_training'] == 0
    promote(config, version)
    predictor = Predictor(config)
    predictor.load()
    arr = decode_png(pair[0], config['limits'])
    mask, actual_version, _ = predictor.predict(arr)
    assert mask.shape == arr.shape and mask.dtype == np.uint8
    assert actual_version == version and set(np.unique(mask)) <= {0, 1, 2}
    second = train(config)
    promote(config, second)
    assert len(list((Path(config['paths']['models']) / 'archive').iterdir())) == 2
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.png')} == before
    store.submit(*pair, case_id='case1')
    assert store.status()['new_cases_since_last_training'] == 1


def test_integrity(config, pair):
    store, root = populate(config, pair)
    records, training, validation = snapshot(store, config)
    assert len(training) == len(validation) == 1
    assert training[0][0].parent != validation[0][0].parent
    (root / 'images/v.png').write_bytes(pair[0])
    with pytest.raises(ValueError, match='also exists'):
        snapshot(store, config)


def test_atomic_failure(config, pair, monkeypatch):
    store = CaseStore(config)
    store.submit(*pair, case_id='case1')
    old = store.records()
    import app.ml.store as module
    original = module.write_json
    def fail(path, data):
        if Path(path).parent == store.root / 'metadata':
            raise OSError('simulated disk failure')
        return original(path, data)
    monkeypatch.setattr(module, 'write_json', fail)
    with pytest.raises(OSError):
        store.submit(*pair, case_id='case1')
    assert store.records() == old


def test_concurrent_submissions(config, pair):
    from concurrent.futures import ThreadPoolExecutor
    store = CaseStore(config)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda i: store.submit(*pair, case_id=f'alias{i}'), range(8)))
    assert len(store.records()) == 1
    assert len({r['case_id'] for r in results}) == 1
    assert len(list((store.root / 'revisions').glob('*/*/metadata.json'))) == 8


def test_unpaired_validation(config, pair):
    store, root = populate(config, pair)
    (root / 'labels/extra.png').write_bytes(pair[1])
    with pytest.raises(ValueError, match='matching filenames'):
        snapshot(store, config)


def test_corrupt_candidate_does_not_replace_production(config, pair):
    import json
    populate(config, pair)
    version = train(config, source='scratch')
    promote(config, version)
    old = production_path(config).resolve()
    meta_path = Path(config['paths']['models']) / 'candidates' / version / 'metadata.json'
    metadata = json.loads(meta_path.read_text())
    metadata['version'] = 'tampered'
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='metadata'):
        promote(config, version)
    assert production_path(config).resolve() == old


def test_interrupted_candidate_cannot_be_promoted(config, pair):
    populate(config, pair)
    config['training']['epochs'] = 2
    def interrupt(**changes):
        if changes.get('epoch') == 2:
            raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train(config, source='scratch', progress=interrupt)
    candidate = next((Path(config['paths']['models']) / 'candidates').iterdir())
    assert (candidate / 'best.pt').is_file()
    assert not (candidate / 'completed.json').exists()
    with pytest.raises(ValueError, match='did not complete'):
        promote(config, candidate.name)
    assert not production_path(config).exists()


def test_completion_marker_is_validated(config, pair):
    import json
    populate(config, pair)
    version = train(config, source='scratch')
    marker = Path(config['paths']['models']) / 'candidates' / version / 'completed.json'
    marker.write_text(json.dumps({'version': 'another_run', 'completed_at': 'now'}))
    with pytest.raises(ValueError, match='completion marker'):
        promote(config, version)
    assert not production_path(config).exists()


def test_decoded_duplicate_guard(config, pair):
    import io
    from PIL import Image
    store, root = populate(config, pair)
    buffer = io.BytesIO()
    Image.open(io.BytesIO(pair[0])).save(buffer, format='PNG', compress_level=0)
    reencoded = buffer.getvalue()
    assert reencoded != pair[0]
    store.submit(reencoded, pair[1], case_id='alias')
    assert len(store.records()) == 1
    (root / 'images/v.png').write_bytes(reencoded)
    with pytest.raises(ValueError, match='also exists'):
        snapshot(store, config)


def test_padding_has_no_loss_or_gradient():
    import torch
    from app.ml.trainer import masked_segmentation_loss
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.ones(1, 1, 4, 4, dtype=torch.long)
    valid = torch.ones(1, 1, 4, 4)
    baseline = masked_segmentation_loss(logits, target, valid)
    padded = torch.nn.functional.pad(logits.detach(), (0, 5, 0, 3), value=10).requires_grad_()
    padded_target = torch.nn.functional.pad(target, (0, 5, 0, 3))
    padded_valid = torch.nn.functional.pad(valid, (0, 5, 0, 3))
    loss = masked_segmentation_loss(padded, padded_target, padded_valid)
    assert torch.allclose(loss, baseline)
    loss.backward()
    assert torch.count_nonzero(padded.grad[:, :, 4:, :]) == 0
    assert torch.count_nonzero(padded.grad[:, :, :, 4:]) == 0


def test_frozen_validation(config, pair, tmp_path):
    from app.ml.trainer import freeze_validation
    store, root = populate(config, pair)
    _, _, pairs = snapshot(store, config)
    frozen = freeze_validation(pairs, tmp_path / 'frozen', config)
    before = frozen[0][0].read_bytes()
    pairs[0][0].write_bytes(pair[0])
    assert frozen[0][0].read_bytes() == before


def test_prompt_perturbations():
    from app.ml.sam2_training import perturb_box, perturb_mask
    binary = np.zeros((32, 32), bool)
    binary[8:24, 8:24] = True
    result = perturb_mask(binary, np.random.default_rng(5))
    assert result.shape == binary.shape and not np.array_equal(result, binary)
    box = perturb_box([0, 0, 31, 31], binary.shape, np.random.default_rng(5))
    assert 0 <= box[0] < box[2] <= 31 and 0 <= box[1] < box[3] <= 31


def test_sam2_robustness_reports_separate_protocols(config, pair, tmp_path, monkeypatch):
    import types
    import app.ml.sam2_training as module
    calls = []
    class Engine:
        def __init__(self, config): pass
        def predict(self, raw, boxes, prompt=None):
            calls.append((boxes, prompt))
            for box in boxes.values():
                assert 0 <= box[0] < box[2] < raw.shape[1]
                assert 0 <= box[1] < box[3] < raw.shape[0]
            return np.zeros((*raw.shape, 3), np.uint8), 'test', 0
    monkeypatch.setattr(module, 'Sam2Predictor', Engine)
    image, mask = tmp_path/'image.png', tmp_path/'mask.png'
    image.write_bytes(pair[0]); mask.write_bytes(pair[1])
    adapter = module.Sam2TrainingAdapter.__new__(module.Sam2TrainingAdapter)
    adapter.model = types.SimpleNamespace(eval=lambda: None)
    adapter.config, adapter.predictor = config, object()
    metrics = adapter.evaluate([(image, mask)])
    assert len(calls) == 3 and calls[2][1] is not None
    assert 'perturbed_box_metrics' in metrics and 'perturbed_mask_metrics' in metrics
    calls.clear(); mask.write_bytes(png(np.zeros((24, 37), np.uint8)))
    adapter.evaluate([(image, mask)])
    assert len(calls) == 2 and set(calls[0][0]) == {1, 2}
