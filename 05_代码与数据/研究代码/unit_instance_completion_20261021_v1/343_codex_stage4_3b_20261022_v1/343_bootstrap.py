"""Paired dataset bootstrap with unique replica image IDs and COCO reaccumulation.

Per-image IoUs/matches are reused; precision/recall are recomputed on each
resampled dataset. This is checked against independent full cloned evaluations.
"""
import importlib,copy,contextlib,io,hashlib
C=importlib.import_module('343_context');E=importlib.import_module('343_evaluator')
import numpy as np

def sampled_evaluator(source,indices,ap_only=False):
    old=list(source._paramsEval.imgIds);n=len(old);indices=np.asarray(indices,dtype=np.int64)
    assert indices.ndim==1 and len(indices)>0 and indices.min()>=0 and indices.max()<n
    result=copy.copy(source);result.params=copy.deepcopy(source.params)
    result._paramsEval=copy.deepcopy(source._paramsEval)
    ids=list(range(1,len(indices)+1))
    result.params.imgIds=ids;result._paramsEval.imgIds=ids
    ng=max(source.cocoGt.anns,default=0)+1;nd=max(source.cocoDt.anns,default=0)+1
    # Original annotation id zero would be indistinguishable from an unmatched
    # sentinel in COCO dtMatches. Bootstrap inputs must be reindexed positively.
    assert all(aid>0 for aid in source.cocoGt.anns),'reindex source GT positively before precomputing matches'
    a=len(source._paramsEval.areaRng);k=len(source._paramsEval.catIds)
    result.evalImgs=[]
    for cat in range(k):
        for area in range(a):
            for occurrence,src in enumerate(indices):
                record=source.evalImgs[(cat*a+area)*n+int(src)]
                if record is None:result.evalImgs.append(None);continue
                r=dict(record);r['image_id']=occurrence+1
                r['gtIds']=[occurrence*ng+v for v in record['gtIds']]
                r['dtIds']=[occurrence*nd+v for v in record['dtIds']]
                r['dtMatches']=np.where(record['dtMatches']>0,record['dtMatches']+occurrence*ng,0)
                r['gtMatches']=np.where(record['gtMatches']>0,record['gtMatches']+occurrence*nd,0)
                result.evalImgs.append(r)
    if ap_only:
        assert result.params.areaRngLbl[:2]==['all','small']
        result.params.areaRng=result.params.areaRng[:2];result.params.areaRngLbl=['all','small']
        result.params.maxDets=[100]
    with contextlib.redirect_stdout(io.StringIO()):result.accumulate()
    return result

def ap_aps(ev):
    p=ev.eval['precision'];last=ev.params.maxDets.index(100)
    values=[]
    for label in ('all','small'):
        v=p[:,:,:,ev.params.areaRngLbl.index(label),last];v=v[v>-1]
        values.append(float(v.mean()) if len(v) else -1.)
    return values

def positive_gt_ids(dataset):
    result=copy.deepcopy(dataset)
    for aid,ann in enumerate(result['annotations'],1):ann['id']=aid
    return result

def paired(evaluators,b_counts=None,repetitions=1000,seed=2026102103):
    assert repetitions==1000 and seed==2026102103
    names=list(evaluators);imageids=list(evaluators[names[0]]._paramsEval.imgIds)
    assert len(imageids)==548
    assert all(list(ev._paramsEval.imgIds)==imageids for ev in evaluators.values())
    if b_counts is not None:assert all(len(b_counts[name])==548 for name in names)
    # One sample matrix is shared by every arm and both seeds.
    indices=np.random.default_rng(seed).integers(0,548,size=(repetitions,548),dtype=np.int64)
    outputs={name:[] for name in names};calls=0
    for r,sample in enumerate(indices):
        memo={}
        for name,ev in evaluators.items():
            if id(ev) not in memo:
                memo[id(ev)]=ap_aps(sampled_evaluator(ev,sample,ap_only=True));calls+=1
            row=list(memo[id(ev)])
            if b_counts is not None:row.append(int(np.asarray(b_counts[name])[sample].sum()))
            outputs[name].append(row)
        if (r+1)%100==0:print(f'bootstrap {r+1}/1000',flush=True)
    return indices,outputs,calls

def intervals(outputs):
    return {name:np.percentile(np.asarray(rows),[2.5,97.5],axis=0).tolist() for name,rows in outputs.items()}

def paired_differences(outputs):
    result={}
    for seed in (1,2):
        for control in ('baseline','N','S','G'):
            delta=np.asarray(outputs[f'seed{seed}_R'])-np.asarray(outputs[f'seed{seed}_{control}'])
            result[f'seed{seed}_R_minus_{control}']={'percentile95':np.percentile(delta,[2.5,97.5],axis=0).tolist()}
    return result
