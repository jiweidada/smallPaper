from pathlib import Path
import importlib.util, json, sys

ROOT = Path(__file__).resolve().parent
ASSETS = Path('/root/uav_a47_20261101_v1/assets_A43_baseline')
PYDEPS = Path('/root/uav_a47_20261101_v1/pydeps')
sys.path.insert(0, str(PYDEPS))
spec = importlib.util.spec_from_file_location('runtime_common', ASSETS/'runtime_common.py')
native = importlib.util.module_from_spec(spec)
sys.modules['runtime_common'] = native
spec.loader.exec_module(native)
DATA = Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')
native.TRAIN_IMAGES = DATA/'VisDrone2019-DET-train/images'
native.TRAIN_ANNS = DATA/'VisDrone2019-DET-train/annotations'
native.VAL_IMAGES = ROOT/'val_inputs'

def plan_check():
    plan = json.loads((ROOT/'protocol_FROZEN.json').read_text())
    assert plan['arms'] == ['B','M','Q'] and plan['epochs'] == 12
    for name, expected in plan['files_sha256'].items():
        assert native.digest(ROOT/name) == expected, name
    expect = json.loads((ROOT/'asset_expectations.json').read_text())
    assert native.digest(ASSETS/'common_init_10class.pth') == expect['common_init_sha256']
    assert native.digest(ASSETS/'runtime_common.py') == expect['runtime_common_sha256']
    for name, expected in expect['native_source_sha256'].items():
        assert native.digest(native.SOURCE/name) == expected, name
    return plan
