"""Prespecified A4 cached-feature ridge diagnostic. No torch or fetched-code imports."""
from pathlib import Path
from datetime import datetime, timezone
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import hashlib
import json
import time
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def group_ci(rows,key,seed=20261030,iterations=2000):
    kept=[r for r in rows if r.get(key) is not None]
    if not kept:
        return dict(mean=None,low=None,high=None,images=0,groups=0)
    groups=sorted({r['group'] for r in kept})
    values=[[r[key] for r in kept if r['group']==g] for g in groups]
    rng=np.random.default_rng(seed)
    boot=[np.mean([v for i in rng.integers(len(groups),size=len(groups)) for v in values[i]]) for _ in range(iterations)]
    return dict(mean=float(np.mean([r[key] for r in kept])),low=float(np.quantile(boot,.025)),
                high=float(np.quantile(boot,.975)),images=len(kept),groups=len(groups))

def fit(rows,view,ridge,fold):
    active=[r for r in rows if len(r['y'])]
    x=np.concatenate([r[view] for r in active]).astype(np.float64)
    y=np.concatenate([r['y'] for r in active]).astype(np.float64)
    weights=np.concatenate([np.full(len(r['y']),1/len(r['y'])/len(active)) for r in active])
    shuffled=[]
    for row in active:
        seed=int(hashlib.sha256(('A4-offset-null:'+str(fold)+':'+row['file_name']).encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed)
        shuffled.append(row['y'][rng.permutation(len(row['y']))])
    targets=np.column_stack([y,np.concatenate(shuffled)])
    mean=(x*weights[:,None]).sum(0)
    std=np.sqrt(((x-mean)**2*weights[:,None]).sum(0))
    std=np.where(std>1e-6,std,1.)
    z=np.column_stack([(x-mean)/std,np.ones(len(x))])
    penalty=np.eye(z.shape[1])*ridge
    penalty[-1,-1]=0
    coef=np.linalg.solve(z.T@(z*weights[:,None])+penalty,z.T@(targets*weights[:,None]))
    assert coef.shape==(193,4) and np.isfinite(coef).all()
    return dict(mean=mean,std=std,coef=coef,constant=(y*weights[:,None]).sum(0))

def main():
    manifest=json.loads((ROOT/'freeze_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    assert p['status']=='FROZEN_BEFORE_A4_FITS'
    (ROOT/'started.json').open('x',encoding='utf-8').write(json.dumps(dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat())))
    began=time.monotonic()
    rows=[]
    for spec in p['source_files']:
        file=Path(spec['path'])
        assert sha(file)==spec['sha256']
        with np.load(file,allow_pickle=False) as z:
            ci=z['gt_center_index']
            eligible=z['gt_probe_eligible'].astype(bool)&(z['gt_sqrt_area']<16)&(ci>=0)
            _,inv,count=np.unique(ci[ci>=0],return_inverse=True,return_counts=True)
            unique=np.zeros(len(ci),bool)
            unique[np.where(ci>=0)[0]]=count[inv]==1
            selected=np.where(eligible&unique)[0]
            centers=(z['gt_boxes'][selected,:2]+z['gt_boxes'][selected,2:])*.5
            y=centers-z['xy'][ci[selected]]
            assert np.all(np.abs(y)<=2.0001)
            row=dict(file_name=str(z['file_name']),fold=int(z['fold']),group=str(z['group']),y=y,
                     uid=[str(v) for v in z['gt_uids'][selected]],size=z['gt_sqrt_area'][selected],
                     encoder_miss=z['gt_all_encoder_max_iou'][selected]<.5)
            for view in p['views']:
                row[view]=z['features_'+view][ci[selected]].astype(np.float64)
                assert row[view].shape==(len(selected),192)
            assert (row['file_name'],row['fold'],row['group'])==(spec['file_name'],spec['fold'],spec['group'])
            rows.append(row)
    for group in {r['group'] for r in rows}:
        assert len({r['fold'] for r in rows if r['group']==group})==1
    predictions={r['file_name']:{} for r in rows}
    models=ROOT/'models';models.mkdir(exist_ok=True)
    receipts=[]
    fits=0
    for fold in range(4):
        train=[r for r in rows if r['fold']!=fold]
        test=[r for r in rows if r['fold']==fold]
        assert not ({r['group'] for r in train}&{r['group'] for r in test})
        for view in p['views']:
            state=fit(train,view,p['ridge'],fold)
            fits+=2
            path=models/(f'fold{fold}_{view}.npz')
            np.savez_compressed(path,**state)
            receipts.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            for row in test:
                z=np.column_stack([(row[view]-state['mean'])/state['std'],np.ones(len(row['y']))])
                prediction=np.clip(z@state['coef'],-2.,2.)
                error=np.linalg.norm(prediction[:,:2]-row['y'],axis=1)
                null_error=np.linalg.norm(prediction[:,2:]-row['y'],axis=1)
                constant_error=np.linalg.norm(np.clip(state['constant'],-2.,2.)-row['y'],axis=1)
                predictions[row['file_name']][view]=dict(error=error,null_error=null_error,constant_error=constant_error)
        print(json.dumps(dict(status='A4_FOLD_COMPLETE',fold=fold,two_response_models_fitted=fits)),flush=True)
    assert fits==32
    per_image=[];per_gt=[]
    for row in rows:
        pred=predictions[row['file_name']]
        image=dict(file_name=row['file_name'],group=row['group'],fold=row['fold'])
        masks=dict(lt16=np.ones(len(row['y']),bool),lt8=row['size']<8,from8to16=row['size']>=8,
                   encoder_miss50=row['encoder_miss'])
        for cohort,mask in masks.items():
            image[cohort+'_n']=int(mask.sum())
            if not mask.any():
                continue
            image[cohort+'_P3minusP2']=float(np.mean(pred['P3_after']['error'][mask]-pred['P2']['error'][mask]))
            image[cohort+'_zero_error']=float(np.mean(np.linalg.norm(row['y'][mask],axis=1)))
            for view in p['views']:
                data=pred[view]
                for metric in ('error','null_error','constant_error'):
                    image[cohort+'_'+view+'_'+metric]=float(np.mean(data[metric][mask]))
                for control in ('null','constant'):
                    image[cohort+'_'+view+'_advantage_'+control]=float(np.mean(data[control+'_error'][mask]-data['error'][mask]))
        per_image.append(image)
        for i,uid in enumerate(row['uid']):
            rec=dict(uid=uid,file_name=row['file_name'],group=row['group'],fold=row['fold'],size=float(row['size'][i]),
                     target_offset=row['y'][i].tolist())
            for view in p['views']:
                rec[view]={k:float(v[i]) for k,v in pred[view].items()}
            per_gt.append(rec)
    metrics=sorted({k for r in per_image for k in r if k not in ('file_name','group','fold') and not k.endswith('_n')})
    estimates={k:group_ci(per_image,k,**{k:p['inference'][k] for k in ('seed','iterations')}) for k in metrics}
    def strong(key,minimum):
        item=estimates[key]
        return item['mean']>=minimum and item['low']>0
    counts={cohort:sum(r.get(cohort+'_n',0) for r in per_image) for cohort in ('lt16','lt8','from8to16','encoder_miss50')}
    enough=counts['lt16']>=200 and counts['lt8']>=50 and counts['from8to16']>=50 and estimates['lt16_P3minusP2']['groups']>=8
    gap=bool(enough and strong('lt16_P3minusP2',.2) and all(estimates[c+'_P3minusP2']['mean']>0 for c in ('lt8','from8to16'))
             and all(strong('lt16_P2_advantage_'+c,.15) for c in ('constant','null')))
    fused=bool(enough and not gap and all(strong('lt16_P3_after_advantage_'+c,.15) for c in ('constant','null')))
    decision=('SUPPORTS_CONDITIONAL_LINEAR_ENCODER_PRECISION_GAP_NOT_NEW_ARCHITECTURE' if gap else
              'FUSED_FINE_CENTER_OFFSETS_LINEARY_READABLE_NATIVE_CAUSALITY_UNPROVEN' if fused else
              'NO_SUFFICIENT_LINEAR_PRECISION_SUPPORT_NO_ARCHITECTURE_MODULE_APPROVED')
    summary=dict(status='A4_CACHED_PRECISION_PROBE_COMPLETE',images=len(rows),counts=counts,estimates=estimates,
                 enough_primary_evidence=bool(enough),encoder_gap_gate=gap,fused_readable_gate=fused,decision=decision,
                 two_response_linear_models_fitted=32,scalar_coordinate_heads=64,parameters_per_two_response_model=386,
                 new_model_forwards=0,new_main_detector_updates=0,global_full_forwards_unchanged=185,
                 numpy_version=np.__version__,python_version=sys.version,elapsed_seconds=time.monotonic()-began,
                 model_files=receipts,scope=p['target'],limitations=p['limits'],scientific_mainline_validated=False)
    dump(ROOT/'summary.json',summary);dump(ROOT/'per_image.json',per_image)
    with (ROOT/'per_gt.jsonl').open('w',encoding='utf-8') as handle:
        for row in per_gt:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    dump(ROOT/'completed.json',dict(status='TERMINAL_SUCCESS',summary_sha256=sha(ROOT/'summary.json'),
                                    new_forwards=0,new_main_updates=0,two_response_models=32,scalar_coordinate_heads=64))
    print(json.dumps(dict(status=summary['status'],decision=decision,counts=counts,
                         primary=estimates['lt16_P3minusP2'],elapsed_seconds=summary['elapsed_seconds'])),flush=True)

if __name__=='__main__':
    main()
