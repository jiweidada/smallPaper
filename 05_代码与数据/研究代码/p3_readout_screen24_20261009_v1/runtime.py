"""Independent training runtime for P3 readout; shared/native code is read-only."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
COMMON_INIT_SHA='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
IMPLEMENTATION=('model.py','readout.py','integration.py','runtime.py','experiment.py',
 'results.py','stage_trace.py','geometry_observer.py','training_audit.py','preflight.py',
 'supervise.py','compare_reused.py','evaluator_binding.py','evaluator_sources.json')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def implementation_hashes():return {name:sha(HERE/name) for name in IMPLEMENTATION}
def verify_training_snapshot(root):
    root=Path(root).resolve()
    binding=json.loads((HERE/'training_source_binding.json').read_text(encoding='utf-8'))
    for rel,digest in binding['files'].items():
        if hashlib.sha256((root/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise RuntimeError('Bound training source changed: '+rel)
    actual={p.relative_to(root).as_posix() for p in (root/'engine').rglob('*.py')}
    if actual!=set(binding['files']):raise RuntimeError('Training source file set differs')
    return binding
def activate_training_snapshot(root):
    verify_training_snapshot(root)
    if 'engine' in sys.modules:raise RuntimeError('A fresh training interpreter is required')
    sys.path.insert(0,str(Path(root).resolve()))
def register_components():
    from integration import register_components as register
    register()
def load_common_initialization(model,path):
    import torch
    from integration import load_baseline_state
    if sha(path)!=COMMON_INIT_SHA:raise RuntimeError('Common initialization differs')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    state=payload['ema']['module'] if 'ema' in payload else payload['model']
    missing=load_baseline_state(model,state)
    return {'common_init_sha':COMMON_INIT_SHA,'native_tensor_count':len(state),
            'readout_arm':model.readout_arm,'new_parameter_count':78500 if missing else 0,
            'new_tensor_count':missing,'initial_output_equality':'zero residual'}
