"""347: new CPU entry plus G_seed1 ten-step acceptance; no complete runs."""
import importlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
S345=WORK/'345_codex_contracts_20261022_v1'
S346=WORK/'346_codex_cpu_boundary_20261022_v1'
sys.path.insert(0,str(S345))
BASE=importlib.import_module('345_context')
E=importlib.import_module('train345')
T=E.T
sha=BASE.sha
read=BASE.read

def write(name,document):
    assert name.startswith('347_')
    E.write_json(HERE/name,document)

def text(name,contents):
    assert name.startswith(('347_','train347'))
    with open(HERE/name,'x',encoding='utf-8',newline='\n') as f:f.write(contents)

def sources():
    paths=sorted(HERE.glob('*.py'))+[S345/'train345.py',BASE.S344/'train344.py',
          BASE.S344/'344_geometry/geom342.py',BASE.S2/'342_head.py',
          BASE.SPEC/'311_unit_instance_core_v2/core.py',BASE.SPEC/'320_geometry_calibrated_core_v2/gcal.py',
          WORK/'uic_losses.py']
    return {str(p):sha(p) for p in paths}

def verify_sources():
    anchor=importlib.import_module('347_bindings_anchor')
    path=HERE/'347_入口源码SHA绑定.json'
    E.verify_file(path,anchor.BUNDLE_SHA256,'347_bundle')
    bundle=read(path)
    for p,expected in bundle['sources'].items():E.verify_file(p,expected,'347_source')
    for record in bundle['old_seals']:
        E.verify_file(record['path'],record['sha256'],'347_old_seal')
    return anchor.BUNDLE_SHA256
