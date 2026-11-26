from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent;A49=Path('/root/uav_a49_20261101_v1')
sys.path.insert(0,str(A49))
from runtime_A49 import native,ASSETS
from scene_scale_prior import install

def components(arm):
    assert arm in ['B','M','Q']
    model,criterion,post=native.create_components('B')
    module=None
    if arm!='B':
        stats=json.loads((ROOT/'scale_statistics_FROZEN_v1.json').read_text())
        module=install(model,criterion,arm,stats['class_offsets'],stats['initial_log_scale'])
    return model,criterion,post,module

def validate_files(filename):
    plan=json.loads((ROOT/filename).read_text())
    for name,h in plan['files_sha256'].items():assert native.digest(ROOT/name)==h,name
    for name,h in plan['external_sha256'].items():assert native.digest(Path(name))==h,name
    expect=json.loads((A49/'asset_expectations.json').read_text())
    assert native.digest(ASSETS/'common_init_10class.pth')==expect['common_init_sha256']
    assert native.digest(ASSETS/'runtime_common.py')==expect['runtime_common_sha256']
    for name,h in expect['native_source_sha256'].items():assert native.digest(native.SOURCE/name)==h,name
    return plan
