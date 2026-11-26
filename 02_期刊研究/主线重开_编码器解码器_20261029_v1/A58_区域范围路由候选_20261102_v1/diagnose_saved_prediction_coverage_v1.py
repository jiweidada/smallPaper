"""GT-centered necessary coverage conditions on saved fixed12 predictions.

No one-to-one assignment, detector, optimizer or new COCO evaluation. A box
can cover multiple GTs; coverage is an upper bound, not detection recall.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import sys
import time
import traceback
import numpy as np

sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
TYPES=['no_rough_overlap','rough_overlap_only','tight_other_label_only','tight_correct_label']

def load(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def save_new(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)

def bbox_iou(gt,pred):
    if not len(pred):return np.zeros((len(gt),0),dtype=np.float64)
    g=np.asarray(gt,dtype=np.float64);p=np.asarray(pred,dtype=np.float64)
    lo=np.maximum(g[:,None,:2],p[None,:,:2]);hi=np.minimum((g[:,:2]+g[:,2:])[:,None],(p[:,:2]+p[:,2:])[None])
    side=np.maximum(hi-lo,0.);inter=side[:,:,0]*side[:,:,1]
    union=(g[:,2]*g[:,3])[:,None]+(p[:,2]*p[:,3])[None,:]-inter
    assert np.all(union>0)
    return inter/union

def blank():
    return dict(objects=0,partition={k:0 for k in TYPES},any_at={str(t):0 for t in [.1,.5,.75]},
        same_at={str(t):0 for t in [.1,.5,.75]},best_rank_at_most={str(k):0 for k in [10,50,100,300]},
        score_at_least={str(t):0 for t in [.05,.1,.25,.5]})

def update(s,row):
    s['objects']+=1;s['partition'][row['partition']]+=1
    for t in [.1,.5,.75]:
        s['any_at'][str(t)]+=int(row['max_iou_any_label']>=t-1e-12)
        s['same_at'][str(t)]+=int(row['max_iou_correct_label']>=t-1e-12)
    if row['best_compatible_filtered_rank_iou05'] is not None:
        for k in [10,50,100,300]:s['best_rank_at_most'][str(k)]+=int(row['best_compatible_filtered_rank_iou05']<=k)
        for t in [.05,.1,.25,.5]:s['score_at_least'][str(t)]+=int(row['best_compatible_score_iou05']>=t)

def main(arm):
    scope=load(P/'saved_coverage_scope_FROZEN_v1.json')
    assert sha(Path(__file__))==scope['script_sha256']
    bundle=load(P/('fixed12_'+arm+'_bundle_local_verification_v1.json'))
    assert bundle['status']=='PASS_FIXED12_BUNDLE_COUNTERS_INPUT_FLOW_AND_CPU_MATCHING'
    d=P/('fixed12_'+arm+'_remote_v1')/('arm_'+arm)
    manifest=load(d.parent/'bundle_manifest.json')
    inputs=['epoch12_EMA_GT.json','epoch12_EMA_predictions.json','epoch12_EMA_precision_recall.npz']
    for n in inputs:assert sha(d/n)==manifest['files']['arm_'+arm+'/'+n]['sha256']
    assert 'torch' not in sys.modules
    gt=load(d/inputs[0]);pred=load(d/inputs[1])
    assert len(gt['images'])==548 and len(gt['annotations'])==38602 and len(pred)<=164400
    small=[g for g in gt['annotations'] if g['area']<=1024];assert len(small)==26444
    out=P/'saved_prediction_coverage_v1'/arm;out.mkdir(parents=True,exist_ok=False)
    save_new(out/'run_once.json',dict(arm=arm,maximum_saved_output_passes=1,model_forwards=0,checkpoint_loads=0,COCO_evaluations=0))
    start=time.monotonic()
    try:
        image={g['id']:g for g in gt['images']};bygt=collections.defaultdict(list);bypred=collections.defaultdict(list)
        for g in small:bygt[g['image_id']].append(g)
        for item in pred:bypred[item['image_id']].append(item)
        total=blank();classes={str(c['id']):blank() for c in gt['categories']}
        bins={k:blank() for k in ['<=4','(4,8]','(8,16]','>16']}
        with (out/'per_small_GT.jsonl').open('x',encoding='utf-8') as rows:
            for iid in sorted(image):
                assert time.monotonic()-start<=scope['maximum_seconds_per_arm']
                gs=bygt[iid];ps=bypred[iid]
                if not gs:continue
                matrix=bbox_iou([g['bbox'] for g in gs],[p['bbox'] for p in ps])
                if ps:
                    labels=np.asarray([p['category_id'] for p in ps]);scores=np.asarray([p['score'] for p in ps])
                    order=np.argsort(-scores,kind='mergesort');rank=np.empty(len(ps),dtype=np.int64);rank[order]=np.arange(1,len(ps)+1)
                else:labels=np.empty(0,dtype=np.int64);scores=np.empty(0);rank=np.empty(0,dtype=np.int64)
                for j,g in enumerate(gs):
                    overlaps=matrix[j];same=labels==g['category_id']
                    any_best=float(overlaps.max()) if len(ps) else 0.
                    same_best=float(overlaps[same].max()) if same.any() else 0.
                    if any_best<.1-1e-12:kind=TYPES[0]
                    elif any_best<.5-1e-12:kind=TYPES[1]
                    elif same_best<.5-1e-12:kind=TYPES[2]
                    else:kind=TYPES[3]
                    ok=same & (overlaps>=.5-1e-12)
                    w,h=g['bbox'][2:];im=image[iid]
                    short=min(w*640/im['width'],h*640/im['height'])
                    size='<=4' if short<=4 else '(4,8]' if short<=8 else '(8,16]' if short<=16 else '>16'
                    row=dict(GT_id=g['id'],image_id=iid,category_id=g['category_id'],original_bbox=g['bbox'],
                        original_area=g['area'],input_short_side=short,input_bin=size,partition=kind,
                        max_iou_any_label=any_best,max_iou_correct_label=same_best,
                        best_compatible_filtered_rank_iou05=int(rank[ok].min()) if ok.any() else None,
                        best_compatible_score_iou05=float(scores[ok].max()) if ok.any() else None)
                    update(total,row);update(classes[str(g['category_id'])],row);update(bins[size],row)
                    rows.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
        assert total['objects']==sum(total['partition'].values())==26444
        assert sum(s['objects'] for s in bins.values())==sum(s['objects'] for s in classes.values())==26444
        for s in [total,*bins.values(),*classes.values()]:
            assert sum(s['partition'].values())==s['objects']
            assert all(s['same_at'][str(t)]<=s['any_at'][str(t)]<=s['objects'] for t in [.1,.5,.75])
        arrays=np.load(d/'epoch12_EMA_precision_recall.npz',allow_pickle=False)
        recall=arrays['recall'];ious=arrays['iou_thresholds'];assert recall.shape==(10,10,4,3)
        comparison={}
        for t in [.5,.75]:
            idx=int(np.flatnonzero(np.isclose(ious,t))[0]);native=recall[idx,:,1,-1];native=native[native>=0]
            macro=float(np.mean([s['same_at'][str(t)]/s['objects'] for s in classes.values() if s['objects']])*100)
            actual=float(native.mean()*100)
            assert macro+1e-8>=actual,(t,macro,actual)
            comparison[str(t)]=dict(individual_GT_oracle_macro_coverage_percent=macro,
                cached_COCO_macro_AR_percent=actual,gap_pp=macro-actual,
                interpretation='Coverage allows shared predictions across GT; gap includes one-to-one matching, score order and competition, not just classification')
        assert 'torch' not in sys.modules
        result=dict(status='PASS_ONE_SAVED_OUTPUT_COVERAGE_PASS',arm=arm,epoch=12,
            small_GT=26444,total=total,by_input_short_side=bins,by_category_id=classes,
            native_recall_comparison=comparison,input_hashes={n:sha(d/n) for n in inputs},
            script_sha256=sha(Path(__file__)),scope_sha256=sha(P/'saved_coverage_scope_FROZEN_v1.json'),
            model_constructors=0,model_forwards=0,optimizer_updates_added=0,checkpoint_loads=0,
            COCO_matching_calls_added=0,saved_output_passes=1,monotonic_seconds=time.monotonic()-start,
            per_GT_rows_sha256=sha(out/'per_small_GT.jsonl'),
            limits=scope['limits'])
        save_new(out/'result.json',result)
        print(json.dumps(dict(status=result['status'],arm=arm,total=total,by_input_short_side=bins,
            native_recall_comparison=comparison,monotonic_seconds=result['monotonic_seconds'],model_forwards=0)),flush=True)
    except BaseException:
        save_new(out/'failure.json',dict(status='FAILED_SAVED_DATA_DIAGNOSTIC_NO_MODEL_RETRY',arm=arm,
            model_forwards=0,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B','M','Q'],required=True)
    main(parser.parse_args().arm)
