from pathlib import Path
import importlib.util,sys
ROOT=Path(__file__).resolve().parent
ASSETS=Path('/root/uav_a47_20261101_v1/assets_A43_baseline')
sys.path.insert(0,'/root/uav_a47_20261101_v1/pydeps')
spec=importlib.util.spec_from_file_location('runtime_common',ASSETS/'runtime_common.py')
native=importlib.util.module_from_spec(spec);sys.modules['runtime_common']=native;spec.loader.exec_module(native)
DATA=Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')
native.TRAIN_IMAGES=DATA/'VisDrone2019-DET-train/images'
native.TRAIN_ANNS=DATA/'VisDrone2019-DET-train/annotations'
native.VAL_IMAGES=Path('/root/uav_a48_20261101_v1/val_inputs')


def validate_files(plan_name):
    import json
    p=json.loads((ROOT/plan_name).read_text())
    for name,h in p['files_sha256'].items():assert native.digest(ROOT/name)==h,name
    for path,h in p['external_sha256'].items():assert native.digest(path)==h,path
    return p
