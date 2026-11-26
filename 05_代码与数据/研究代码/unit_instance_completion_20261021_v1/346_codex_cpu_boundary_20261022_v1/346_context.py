"""346 CPU boundary acceptance only, authorized by the user's current message."""
import importlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
S345=WORK/'345_codex_contracts_20261022_v1'
sys.path.insert(0,str(S345))
BASE=importlib.import_module('345_context')
sha=BASE.sha
read=BASE.read
E=importlib.import_module('train345')
T=E.T
CASES={
    'epoch_201_204':{'start':201,'end':204,'maximum_actual_updates':3,
                     'expected_batch_sizes':[16,3,16],'end_cursor':[1,1,16]},
    'final_1216_1218':{'start':1216,'end':1218,'maximum_actual_updates':2,
                       'expected_batch_sizes':[16,3],'end_cursor':[6,0,0]}}
KIND='346_synthetic_boundary_acceptance_only'
RUN_ROOT=HERE/'346_boundary_runs'

def write(name,document):
    assert name.startswith('346_')
    E.write_json(HERE/name,document)

def text(name,contents):
    assert name.startswith(('346_','train346'))
    with open(HERE/name,'x',encoding='utf-8',newline='\n') as f:f.write(contents)

def verify_sources():
    anchor=importlib.import_module('346_bindings_anchor')
    path=HERE/'346_冻结源码与旧封存绑定.json'
    E.verify_file(path,anchor.BUNDLE_SHA256,'346_bundle')
    bundle=read(path)
    for p,expected in bundle['sources'].items():E.verify_file(p,expected,'346_source')
    E.verify_file(bundle['old_seal']['path'],bundle['old_seal']['sha256'],'345_seal')
    return anchor.BUNDLE_SHA256

def binding_for(old_binding,case):
    s=CASES[case]
    return {**old_binding,'execution_kind':KIND,'case':case,
            'synthetic_origin_step':s['start'],'allowed_endpoint':s['end'],
            'maximum_actual_updates':s['maximum_actual_updates'],
            '346_source_bundle_sha256':verify_sources()}

def schedule(order,batch=16):
    """Independent complete schedule: materialize nested loops, no divmod cursor."""
    result=[]
    for epoch,ids in enumerate(order['epochs']):
        for offset in range(0,len(ids),batch):
            result.append({'step':len(result),'epoch':epoch,'batch_index':offset//batch,
                           'position_in_epoch':offset,'image_ids':ids[offset:offset+batch]})
    return result

def source_sha():
    paths=sorted(HERE.glob('*.py'))+[S345/'train345.py',BASE.S344/'train344.py',
          BASE.S344/'344_geometry/geom342.py',BASE.S2/'342_head.py',
          BASE.SPEC/'311_unit_instance_core_v2/core.py',BASE.SPEC/'320_geometry_calibrated_core_v2/gcal.py',
          WORK/'uic_losses.py']
    return {str(p):sha(p) for p in paths}
