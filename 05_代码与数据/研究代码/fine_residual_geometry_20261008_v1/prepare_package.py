"""Generate isolated reviewed source/config scaffolding using standard library only."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'fine_geometry_20261008_v1'
ARMS = ('baseline','fused_residual','fine_residual')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,s):
    with p.open('x',encoding='utf-8') as f: f.write(s)
def writej(p,s): write(p,json.dumps(s,ensure_ascii=False,indent=2))

def main():
    old_receipt=read(OLD/'03_交付回执与SHA.json')
    for name in ('geometry.py','runtime.py','experiment.py'):
        assert sha(OLD/name)==old_receipt['implementation'][name],name
    old_geometry=(OLD/'geometry.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(old_geometry).body if isinstance(n,ast.ClassDef) and n.name=='FineGeometryEncoder')
    encoder=ast.get_source_segment(old_geometry,node).replace('FineGeometryEncoder','FeatureBundleEncoder')
    header='''"""Isolated local geometry residual experiment. Native encoder path retained.
Original encoder/decoder interface copyright: DEIM and D-FINE Authors, 2024.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from residual import LocalGeometryResidual
ARMS = ('baseline','fused_residual','fine_residual')
'''
    write(HERE/'geometry.py',header+'\n'+encoder+'\n\n'+(HERE/'geometry_tail.py.inc').read_text(encoding='utf-8'))
    runner=(OLD/'experiment.py').read_text(encoding='utf-8')
    replacements={
        "('baseline', 'fine_only', 'grid_only', 'fine_grid')":repr(ARMS),
        'fine_geometry_v1':'fine_residual_geometry_v1',
        'GeometryDEIM':'ResidualGeometryDEIM',
        'len(plans) != 8':'len(plans) != 6',
        'Expected all four arms and both seeds':'Expected baseline, same-structure control, candidate, and both seeds',
        'fine_geometry_identity':'fine_residual_geometry_identity'}
    for old,new in replacements.items():
        assert old in runner,old
        runner=runner.replace(old,new)
    write(HERE/'experiment.py',runner)
    runtime=(OLD/'runtime.py').read_text(encoding='utf-8')
    runtime=runtime[:runtime.index('def load_common_initialization')]
    runtime=runtime.replace('GeometryDEIM','ResidualGeometryDEIM').replace(
        "('geometry.py', 'runtime.py', 'experiment.py')","('geometry.py', 'residual.py', 'runtime.py', 'experiment.py')")
    runtime+='''def load_common_initialization(model, path):
    import torch
    if sha(path) != COMMON_INIT_SHA:
        raise RuntimeError('Expected the bound common initialization')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['ema']['module'] if 'ema' in payload else payload['model']
    current = model.state_dict()
    expected_extra = set() if model.geometry_arm == 'baseline' else {
        'decoder.geometry_residual.' + x for x in
        ('kernel','mix_weight','mix_bias','output_weight','output_bias')}
    if set(current) - set(state) != expected_extra or set(state) - set(current):
        raise RuntimeError('Native keys or extra residual keys differ')
    for key in state:
        if state[key].shape != current[key].shape or state[key].dtype != current[key].dtype:
            raise RuntimeError('Native initialization tensor mismatch: ' + key)
    if expected_extra:
        for suffix in ('output_weight','output_bias'):
            if torch.count_nonzero(current['decoder.geometry_residual.'+suffix]).item():
                raise RuntimeError('Residual output must start at zero')
    result = model.load_state_dict(state, strict=not expected_extra)
    if set(result.missing_keys) != expected_extra or result.unexpected_keys:
        raise RuntimeError('Unexpected common initialization load result')
    return {'common_init_sha': COMMON_INIT_SHA, 'native_tensor_count': len(state),
            'geometry_arm': model.geometry_arm, 'new_parameter_count': 35396 if expected_extra else 0,
            'initial_output_equality': 'Zero residual retains the native forward'}
'''
    write(HERE/'runtime.py',runtime)
    for name in ('source_binding.json','training_source_binding.json'):
        write(HERE/name,(OLD/name).read_text(encoding='utf-8'))
    (HERE/'configs').mkdir(exist_ok=False)
    binding={}
    for seed in (1,2):
        for arm in ARMS:
            plan=read(OLD/f'configs/baseline_seed{seed}.json')
            plan.update(schema='fine_residual_geometry_v1',arm=arm)
            plan.pop('factors')
            plan['residual_readout']=dict(source={'baseline':None,'fused_residual':'fused','fine_residual':'fine'}[arm],
                                         window=3,hidden=64,initialization_seed=0,
                                         main_candidate=arm=='fine_residual')
            key=f'{arm}_seed{seed}'
            writej(HERE/f'configs/{key}.json',plan)
            binding[key]=hashlib.sha256(json.dumps(plan,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    writej(HERE/'plan_binding.json',binding)
    writej(HERE/'runner_provenance.json',dict(source=str(OLD),source_files={n:sha(OLD/n) for n in ('geometry.py','runtime.py','experiment.py')},
        changes='Preserved native encoder bundle; local residual model registration; native criterion; strict extra-key initialization; three arms; isolated checkpoint identity.'))
    print(json.dumps({'status':'GENERATED_NOT_RUN','plans':len(binding),'model_calls':0,'optimizer_calls':0}))

if __name__=='__main__': main()
