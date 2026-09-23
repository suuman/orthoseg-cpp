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
