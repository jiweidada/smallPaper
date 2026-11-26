from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
A49 = Path('/root/uav_a49_20261101_v1')
sys.path.insert(0, str(A49))
from runtime_A49 import native, ASSETS
from contrastive_read import install


def components(arm):
    assert arm in ('B', 'M', 'Q')
    model, criterion, post = native.create_components('B')
    branches = install(model, arm) if arm != 'B' else []
    return model, criterion, post, branches


def validate_files(filename):
    plan = json.loads((ROOT / filename).read_text())
    for name, expected in plan['files_sha256'].items():
        assert native.digest(ROOT / name) == expected, name
    for name, expected in plan['external_sha256'].items():
        assert native.digest(Path(name)) == expected, name
    expected = json.loads((A49 / 'asset_expectations.json').read_text())
    assert native.digest(ASSETS / 'common_init_10class.pth') == expected['common_init_sha256']
    assert native.digest(ASSETS / 'runtime_common.py') == expected['runtime_common_sha256']
    for name, value in expected['native_source_sha256'].items():
        assert native.digest(native.SOURCE / name) == value, name
    return plan
