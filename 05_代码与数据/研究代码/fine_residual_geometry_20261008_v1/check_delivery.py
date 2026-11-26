"""Read-only configuration checks; imports no torch or native training engine."""
from copy import deepcopy
import json
import sys
import time
from runtime import HERE, sha
from experiment import ARMS, check_code, read_json, validate_plan, resolved_config

def main():
    old=HERE.parent/'fine_geometry_20261008_v1'
    old_bind=read_json(old/'plan_binding.json')
    sources={}; checks=[]
    for seed in (1,2):
        original=read_json(old/f'configs/baseline_seed{seed}.json')
        import hashlib
        canonical=json.dumps(original,sort_keys=True,ensure_ascii=False).encode('utf-8')
        assert hashlib.sha256(canonical).hexdigest()==old_bind[f'baseline_seed{seed}']
        sources[f'baseline_seed{seed}']=sha(old/f'configs/baseline_seed{seed}.json')
        for arm in ARMS:
            plan=read_json(HERE/f'configs/{arm}_seed{seed}.json')
            validate_plan(plan)
            for key in ('native_baseline_config','model_recipe','criterion_recipe','native_args_sha256'):
                assert plan[key]==original[key],(arm,seed,key)
            expected_source={'baseline':None,'fine_residual':'fine','fused_residual':'fused'}[arm]
            assert plan['residual_readout']==dict(source=expected_source,window=3,hidden=64,initialization_seed=0,main_candidate=arm=='fine_residual')
            paths={k:'/review/'+k for k in ('common_init','output','train_images','val_images','train_ann','val_ann')}
            cfg=resolved_config(plan,paths)
            assert cfg['criterion']=='DEIMCriterion' and cfg['DEIMCriterion']==original['criterion_recipe']
            assert cfg['ResidualGeometryDEIM']==dict(recipe=original['model_recipe'],arm=arm)
            changed=deepcopy(plan);changed['native_baseline_config']['optimizer']['lr']*=2
            try:validate_plan(changed)
            except ValueError:pass
            else:raise AssertionError('Unapproved learning-rate change accepted')
            changed=deepcopy(plan);changed['residual_readout']['window']=5
            try:validate_plan(changed)
            except ValueError:pass
            else:raise AssertionError('Unreviewed neighborhood change accepted')
            checks.append(f'{arm}_seed{seed}: native recipe unchanged, source fixed, drift rejected')
    path=old/'prepared_local/baseline_seed1/receipt.json'
    native=read_json(path)['paths']['native_root']
    static=check_code(native)
    assert 'torch' not in sys.modules
    result=dict(status='PASS',checks=checks,original_plan_sha256=sources,
                static=static,torch_imported=False,training_started=False)
    dest=HERE/('delivery_check_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
