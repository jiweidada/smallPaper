"""Paired image bootstrap from saved predictions only; no model execution."""
from pathlib import Path
from collections import defaultdict
import json, os, sys, time, multiprocessing as mp
import numpy as np
ROOT=Path('/root/c10_region_restore_20261029_v1')
OUT=ROOT/'simple_codec_gate_v1'
sys.path.insert(0,str(OUT));sys.path.insert(0,str(ROOT/'code'))
from run_gate import metrics
from evaluate_fixed import ground_truth_input,bootstrap_ap
from study import json_write
CONTEXT={}
COMPARISONS={'source_minus_webp':('source','webp'),'source_minus_half_jpeg':('source','half_jpeg'),
             'webp_minus_q20':('webp','q20'),'half_jpeg_minus_q20':('half_jpeg','q20'),
             'half_jpeg_minus_webp':('half_jpeg','webp')}

def chunk(job):
    offset,draws=job;values=defaultdict(list)
    for draw in draws:
        for arm in ['D0','DA']:
            ap={c:bootstrap_ap(CONTEXT[(arm,c)],draw) for c in ['source','q20','webp','half_jpeg']}
            for key,(a,b) in COMPARISONS.items():values[arm+'/'+key].append(100*(ap[a]-ap[b]))
    return offset,dict(values)

def main():
    start=time.monotonic()
    assert (OUT/'completed.json').exists() and not (OUT/'bootstrap_completed.json').exists()
    with (OUT/'bootstrap_one_shot.lock').open('x') as f:f.write(str(os.getpid()))
    plan=dict(status='STARTED',unit='source image; paired across conditions',replicates=2000,seed=20261029,workers=8,
              detector_forwards=0,training_updates=0,comparisons=COMPARISONS,primary_max_det=500,
              correction='exploratory pointwise 95% percentile intervals; not simultaneous/familywise guarantees',
              limitations='single trained seed/checkpoint; related frames and scene non-independence not corrected')
    json_write(OUT/'bootstrap_plan.json',plan)
    complete=json.loads((OUT/'completed.json').read_text());original=json.loads((ROOT/'evaluation/initial_diagnostic_result.json').read_text())
    gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text())
    obs=json.loads((OUT/'observations.json').read_text());inp_gt=ground_truth_input(gt,obs['images'])
    n=len(gt['images']);point={}
    for arm in ['D0','DA']:
        point[arm]={}
        for cond in ['source','q20','webp','half_jpeg']:
            path=(ROOT/'evaluation'/arm/(cond+'_predictions_input.json')) if cond in ['source','q20'] else (OUT/'evaluation'/arm/(cond+'_predictions_input.json'))
            ev,ms=metrics(inp_gt,json.loads(path.read_text()),500)
            expected=original[arm][cond]['AP_small_input_500'] if cond in ['source','q20'] else complete['metrics'][arm][cond]['AP_small_input']
            assert abs(ms['AP_small']-expected)<1e-12
            assert abs(bootstrap_ap(ev,np.arange(n))-ms['AP_small'])<1e-12
            CONTEXT[(arm,cond)]=ev;point[arm][cond]=ms['AP_small']
            json_write(OUT/'bootstrap_live.json',dict(stage='MATCH_CACHE',arm=arm,condition=cond,elapsed_seconds=time.monotonic()-start))
    draws=np.random.default_rng(plan['seed']).integers(0,n,size=(plan['replicates'],n))
    collected=defaultdict(list);finished=0
    with mp.get_context('fork').Pool(processes=plan['workers']) as pool:
        for offset,results in pool.imap(chunk,[(i,draws[i:i+50]) for i in range(0,len(draws),50)]):
            for key,vals in results.items():collected[key].extend(vals)
            finished+=len(next(iter(results.values())))
            info=dict(stage='BOOTSTRAP',replicates=finished,total=plan['replicates'],elapsed_seconds=time.monotonic()-start)
            json_write(OUT/'bootstrap_live.json',info)
            if finished%200==0:print(json.dumps(info),flush=True)
    out={}
    for arm in ['D0','DA']:
        out[arm]={}
        for name,(a,b) in COMPARISONS.items():
            values=np.asarray(collected[arm+'/'+name]);assert len(values)==plan['replicates'] and np.isfinite(values).all()
            out[arm][name]=dict(point_pp=100*(point[arm][a]-point[arm][b]),paired_image_bootstrap95_pp=np.quantile(values,[.025,.975]).tolist())
    json_write(OUT/'bootstrap_completed.json',dict(status='COMPLETED',plan=plan,comparisons=out,elapsed_seconds=time.monotonic()-start,full_draw_exact_checks=True))
    print(json.dumps(dict(status='COMPLETED',DA=out['DA'])),flush=True)

if __name__=='__main__':main()
