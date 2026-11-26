"""Post-diagnostic descriptive/uncertainty analysis from saved DA predictions only."""
from pathlib import Path
from collections import defaultdict
import json,multiprocessing as mp,sys,time
import numpy as np
ROOT=Path('/root/c10_region_restore_20261029_v1');OUT=ROOT/'restoration_probe_v1'
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'simple_codec_gate_v1'))
from run_gate import metrics
from evaluate_fixed import bootstrap_ap,ground_truth_input
from study import json_write
CTX={}
CONDITIONS=['subset_00','subset_15','alpha_075','independent','quadratic','finite_loss_best']
CONTRASTS={'quadratic_minus_independent':('quadratic','independent'),'independent_minus_global075':('independent','alpha_075'),
 'quadratic_minus_global075':('quadratic','alpha_075'),'independent_minus_JPEG':('independent','subset_00'),
 'global075_minus_JPEG':('alpha_075','subset_00'),'full_minus_JPEG':('subset_15','subset_00'),
 'finite_loss_best_minus_quadratic':('finite_loss_best','quadratic')}
def work(job):
    offset,draws=job;out=defaultdict(list)
    for draw in draws:
        ap={name:bootstrap_ap(CTX[name],draw) for name in CONDITIONS}
        for name,(a,b) in CONTRASTS.items():out[name].append(100*(ap[a]-ap[b]))
    return offset,dict(out)
def main():
    start=time.monotonic();assert (OUT/'completed.json').exists() and not (OUT/'bootstrap_completed.json').exists()
    with (OUT/'bootstrap_one_shot.lock').open('x') as f:f.write('DA saved-prediction analysis')
    json_write(OUT/'bootstrap_plan.json',dict(stage='after primary regional diagnostics; exploratory',replicates=2000,seed=20261029,workers=8,
        contrasts=CONTRASTS,training_updates=0,detector_forwards=0,
        warning='Actions use GT. Image bootstrap conditions on those privileged actions; no deployment/learning uncertainty. Global alpha075 was strongest observed fixed alpha and intervals are conditional/pointwise, not corrected simultaneous guarantees.'))
    gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text());obs=json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())['images'];input_gt=ground_truth_input(gt,obs)
    complete=json.loads((OUT/'completed.json').read_text());point={};classes={}
    for name in CONDITIONS:
        preds=json.loads((OUT/'predictions/DA'/(name+'.json')).read_text());ev,ms=metrics(input_gt,preds,500)
        assert abs(ms['AP_small']-complete['metrics']['DA'][name]['AP_small_input'])<1e-12
        assert abs(bootstrap_ap(ev,np.arange(len(obs)))-ms['AP_small'])<1e-12
        CTX[name]=ev;point[name]=ms['AP_small'];classes[name]={}
        for k,cat in enumerate(ev.params.catIds):
            precision=ev.eval['precision'][:,:,k,1,2];v=precision[precision>=0]
            count=sum(a['category_id']==cat and not a['iscrowd'] and a['area']<1024 for a in input_gt['annotations'])
            classes[name][str(cat)]=dict(AP_small=float(v.mean()) if len(v) else None,GT_small=count)
    draws=np.random.default_rng(20261029).integers(0,len(obs),size=(2000,len(obs)))
    vals=defaultdict(list)
    with mp.get_context('fork').Pool(processes=8) as pool:
        for offset,result in pool.imap(work,[(i,draws[i:i+50]) for i in range(0,2000,50)]):
            for k,v in result.items():vals[k].extend(v)
            done=len(next(iter(vals.values())));info=dict(replicates=done,total=2000,elapsed_seconds=time.monotonic()-start)
            json_write(OUT/'bootstrap_live.json',info)
            if done%400==0:print(json.dumps(info),flush=True)
    results={name:dict(point_pp=100*(point[a]-point[b]),paired_image_bootstrap95_pp=np.quantile(vals[name],[.025,.975]).tolist()) for name,(a,b) in CONTRASTS.items()}
    def matched(m):return set(next(x for x in m['matches'] if x['confidence']==.25 and x['iou']==.5)['matched_small_uids'])
    jpeg=matched(complete['metrics']['DA']['subset_00']);ref=matched(json.loads((ROOT/'evaluation/DA/source_metrics.json').read_text()));original_compression_misses=ref-jpeg
    transitions={}
    for name in CONDITIONS:
        selected=matched(complete['metrics']['DA'][name]);transitions[name]=dict(jpeg_hits_lost=len(jpeg-selected),new_hits_vs_jpeg=len(selected-jpeg),
                recovered_source_hit_jpeg_missed=len(original_compression_misses&selected),source_hit_jpeg_missed_total=len(original_compression_misses))
    json_write(OUT/'bootstrap_completed.json',dict(status='COMPLETED',arm='DA',contrasts=results,per_class_input_small_AP=classes,
              transitions_conf025_iou05=transitions,replicates=2000,elapsed_seconds=time.monotonic()-start,detector_forwards=0,training_updates=0))
    print(json.dumps(dict(status='COMPLETED',contrasts=results,transitions=transitions)),flush=True)
if __name__=='__main__':main()
