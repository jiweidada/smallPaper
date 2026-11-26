"""Exploratory fixed-assignment choice AP; reuse all previously computed action predictions."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json,multiprocessing as mp,sys
ROOT=Path('/root/c10_region_restore_20261029_v1');OUT=ROOT/'restoration_probe_v1'
sys.path.insert(0,str(OUT));from run_probe import NAMES,evaluate_one
def main():
    assert (OUT/'completed.json').exists() and not (OUT/'fixed_choice_metrics.json').exists()
    plan=dict(stage='Exploratory after primary AP results; clarify whether assignment-based surrogate choices change AP',
              detector_forwards=0,training_updates=0,policy_fitting=0,selection_requires_GT=True)
    (OUT/'fixed_choice_plan.json').write_text(json.dumps(plan,indent=2))
    jobs=[]
    for arm in ['D0','DA']:
        rows=json.loads((OUT/(arm+'_loss_diagnostics.json')).read_text());assert len(rows)==100
        by_action={}
        for name in NAMES:
            mapping=defaultdict(list)
            for p in json.loads((OUT/'predictions'/arm/(name+'.json')).read_text()):mapping[p['image_id']].append(p)
            by_action[name]=mapping
        for method in ['independent','quadratic','finite_loss_best','per_image_alpha_loss_best']:
            predictions=[]
            for row in rows:predictions.extend(by_action[NAMES[row['fixed_assignment']['choices'][method]]][row['image_id']])
            name='fixed_assignment_'+method
            (OUT/'predictions'/arm/(name+'.json')).write_text(json.dumps(predictions))
            jobs.append((arm,name))
    result=defaultdict(dict)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for arm,name,metrics in pool.map(evaluate_one,jobs):result[arm][name]=metrics
    (OUT/'fixed_choice_metrics.json').write_text(json.dumps(dict(status='COMPLETED',plan=plan,metrics=result),indent=2))
    print(json.dumps({a:{n:100*m['AP_small_input'] for n,m in v.items()} for a,v in result.items()}),flush=True)
if __name__=='__main__':main()
