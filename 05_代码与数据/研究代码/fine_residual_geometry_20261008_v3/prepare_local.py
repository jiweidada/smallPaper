"""Create all six local configuration bundles. Never starts training."""
from types import SimpleNamespace
from pathlib import Path
import json
import sys
import time
from experiment import ARMS, HERE, read_json, prepare_run, verify_bundle

def main():
    old=HERE.parent/'fine_geometry_20261008_v1'
    previous=read_json(old/'prepared_local/baseline_seed1/receipt.json')['paths']
    results=[]
    for seed in (1,2):
        for arm in ARMS:
            key=f'{arm}_seed{seed}'
            args={k:v for k,v in previous.items() if k!='output'}
            args.update(plan=str(HERE/f'configs/{key}.json'),
                        output=str(HERE/f'runs/{key}'),bundle=str(HERE/f'prepared_local/{key}'))
            result=prepare_run(SimpleNamespace(**args))
            verify_bundle(args['bundle'])
            assert not Path(args['output']).exists()
            results.append(dict(arm=arm,seed=seed,status=result['status'],bundle=args['bundle']))
    assert 'torch' not in sys.modules
    report=dict(status='SIX_BUNDLES_PREPARED_NOT_TRAINED',results=results,
                torch_imported=False,optimizer_calls=0,model_calls=0)
    dest=HERE/('preparation_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    with dest.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
