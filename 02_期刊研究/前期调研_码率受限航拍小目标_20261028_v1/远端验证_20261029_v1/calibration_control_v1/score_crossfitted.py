"""Score frozen local CPU cross-fit decisions on saved detector outputs; no model fitting or forward."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,multiprocessing as mp,sys,time
ROOT=Path('/root/c10_region_restore_20261029_v1');OUT=ROOT/'calibration_control_v1';REG=ROOT/'restoration_probe_v1'
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'simple_codec_gate_v1'))
from study import json_write
from evaluate_fixed import ground_truth_input
from run_gate import metrics,matches
METHODS=['regional_threshold','regional_ridge','regional_hgb','global_ridge']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def grouped(path):
    d=defaultdict(list)
    for p in json.loads(path.read_text()):d[p['image_id']].append(p)
    return d
def score(job):
    arm,method=job;gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text());obs=json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())['images'];inp=ground_truth_input(gt,obs)
    pred=json.loads((OUT/'predictions'/arm/(method+'.json')).read_text());ev,mi=metrics(inp,pred,500)
    original=[];mapping={m['image_id']:m for m in obs}
    for p in pred:
        m=mapping[p['image_id']];r=m['geometry']['ratio'];l,t,_,_=m['geometry']['pad_ltrb'];x,y,w,h=p['bbox']
        original.append(dict(image_id=p['image_id'],category_id=p['category_id'],score=p['score'],bbox=[(x-l)/r,(y-t)/r,w/r,h/r]))
    _,mo=metrics(gt,original,500);assert abs(mi['AP']-mo['AP'])<1e-10
    result=dict(AP=mo['AP'],AP50=mo['AP50'],AP_small_original=mo['AP_small'],AP_small_input=mi['AP_small'],
                matches=[matches(ev,inp,c,k) for c in [.05,.1,.25,.5] for k in [.5,.75]])
    json_write(OUT/'metrics'/arm/(method+'.json'),result);return arm,method,result
def main():
    start=time.monotonic()
    with (OUT/'score_one_shot.lock').open('x') as f:f.write('score frozen local CPU cross-fit outputs')
    frozen=json.loads((OUT/'crossfit_frozen.json').read_text());assert sha(OUT/'crossfitted_decisions.json')==frozen['decision_sha256']
    assert sha(OUT/'protocol.json')==frozen['protocol_sha256'] and frozen['fitted_utility_policies']==40
    decisions=json.loads((OUT/'crossfitted_decisions.json').read_text());folds=json.loads((OUT/'fold_manifest.json').read_text())['source_folds']
    for arm in ['D0','DA']:
        rows=decisions[arm];assert len(rows)==100 and len({r['image_id'] for r in rows})==100
        for r in rows:assert r['fold']==folds[r['file_name']]
        by_action={i:grouped(REG/'predictions'/arm/('subset_%02d.json'%i)) for i in range(16)}
        for method in METHODS:
            pred=[]
            for row in rows:pred.extend(by_action[row['choices'][method]][row['image_id']])
            json_write(OUT/'predictions'/arm/(method+'.json'),pred)
    result=defaultdict(dict);jobs=[(a,m) for a in ['D0','DA'] for m in METHODS]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for a,m,scores in pool.map(score,jobs):
            result[a][m]=scores;json_write(OUT/'live.json',dict(stage='SCORING',arm=a,method=m,elapsed_seconds=time.monotonic()-start))
    assert sha(OUT/'crossfitted_decisions.json')==frozen['decision_sha256']
    prior=json.loads((REG/'completed.json').read_text());interpret={}
    for arm in ['D0','DA']:
        oracle=prior['metrics'][arm]['independent']['AP_small_input'];glob=prior['metrics'][arm]['alpha_075']['AP_small_input']
        interpret[arm]={m:dict(small_AP_pct=100*result[arm][m]['AP_small_input'],gap_from_GT_independent_pp=100*(oracle-result[arm][m]['AP_small_input']),
                    gain_over_observed_global075_pp=100*(result[arm][m]['AP_small_input']-glob)) for m in METHODS}
    json_write(OUT/'completed.json',dict(status='COMPLETED',metrics=result,interpretation=interpret,frozen_local_fit=frozen,
         elapsed_seconds=time.monotonic()-start,detector_forwards=0,restorer_forwards=0,detector_training_updates=0,utility_policy_fits=40,
         exploratory_cross_fitting_only=True,independent_confirmation_completed=False))
    print(json.dumps(dict(status='COMPLETED',interpretation=interpret)),flush=True)
if __name__=='__main__':main()
