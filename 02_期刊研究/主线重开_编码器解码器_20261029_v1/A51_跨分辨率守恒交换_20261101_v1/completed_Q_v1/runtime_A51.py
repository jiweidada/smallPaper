from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
A49=Path('/root/uav_a49_20261101_v1')
sys.path.insert(0,str(A49))
from runtime_A49 import native,ASSETS
from conservative_scale_exchange import install


def components(arm):
    model,criterion,post=native.create_components('B')
    module=install(model,arm)
    return model,criterion,post,module


def validate_files(filename):
    plan=json.loads((ROOT/filename).read_text())
    for name,h in plan['files_sha256'].items():assert native.digest(ROOT/name)==h,name
    for name,h in plan['external_sha256'].items():assert native.digest(Path(name))==h,name
    return plan
