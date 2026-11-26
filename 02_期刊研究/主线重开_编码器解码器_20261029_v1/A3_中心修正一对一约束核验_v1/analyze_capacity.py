"""A3 conditional GT-oracle capacity audit;no model import,forward,or fitting."""
from pathlib import Path
import hashlib
import json
import os
import time
import tarfile
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent

def match(edges,iou,protected=None):
    n,q=edges.shape
    protected=np.zeros(n,bool) if protected is None else protected
    weight=np.ones(n)+(n+1)*protected.astype(float)
    utility=np.zeros((n,q+n),np.float64)
    utility[:,:q]=np.where(edges,weight[:,None]+1e-6*iou,-1e6)
    rr,cc=linear_sum_assignment(-utility)
    real=cc<q
    rr,cc=rr[real],cc[real]
    ok=edges[rr,cc]
    rr,cc=rr[ok],cc[ok]
    assert len(set(cc.tolist()))==len(cc)
    return rr,cc

def audit(original,oracle,iou,primary):
    base_gt,base_q=match(original,iou)
    protected=np.zeros(len(primary),bool); protected[base_gt]=True
    assert not np.any(protected&primary)
    free=np.ones(original.shape[1],bool); free[base_q]=False
    oracle=oracle&primary[:,None]
    free_gt,free_q=match(oracle&free[None,:],iou)
    flex_edges=(original&protected[:,None])|oracle
    flex_gt,flex_q=match(flex_edges,iou,protected)
    retained=np.zeros(len(primary),bool); retained[flex_gt]=True
    assert retained[protected].all()
    flex_recovered=flex_gt[primary[flex_gt]]
    assert len(free_gt)<=len(flex_recovered)<=int(oracle.any(1).sum())
    return dict(base_gt=base_gt,base_q=base_q,free_gt=free_gt,free_q=free_q,
                flex_gt=flex_recovered,unconstrained_gt=np.where(oracle.any(1))[0],
                protected_retained=int(protected.sum()),free_queries=int(free.sum()))

def selfcheck():
    original=np.array([[1,0],[0,0]],bool)
    oracle=np.array([[0,0],[1,0]],bool)
    primary=np.array([False,True])
    a=audit(original,oracle,np.array([[.9,0.],[0.,0.]]),primary)
    assert len(a['unconstrained_gt'])==1 and len(a['free_gt'])==0 and len(a['flex_gt'])==0
    original=np.array([[1,1],[0,0]],bool)
    b=audit(original,oracle,np.array([[.9,.8],[0.,0.]]),primary)
    assert len(b['free_gt'])==0 and len(b['flex_gt'])==1
    c=audit(np.array([[1,0],[0,0]],bool),np.array([[0,0],[0,1]],bool),np.array([[.9,0.],[0.,0.]]),primary)
    assert len(c['free_gt'])==1 and len(c['flex_gt'])==1
    return 'PASS same-query conflict,coverage-preserving reassignment,and truly spare-query recovery'

def matrices(gt,pred):
    size=gt[:,2:]-gt[:,:2]
    area=np.prod(size,1)
    pa=np.prod(pred[:,2:],1)
    lo=pred[:,:2]-pred[:,2:]*.5
    hi=pred[:,:2]+pred[:,2:]*.5
    inter=np.prod(np.maximum(np.minimum(gt[:,None,2:],hi[None])-np.maximum(gt[:,None,:2],lo[None]),0),-1)
    iou=inter/np.maximum(area[:,None]+pa[None]-inter,1e-12)
    center=(gt[:,:2]+gt[:,2:])*.5
    clo=center[:,None,:]-pred[None,:,2:]*.5
    chi=center[:,None,:]+pred[None,:,2:]*.5
    ci=np.prod(np.maximum(np.minimum(gt[:,None,2:],chi)-np.maximum(gt[:,None,:2],clo),0),-1)
    center_iou=ci/np.maximum(area[:,None]+pa[None]-ci,1e-12)
    return iou,center_iou

def ratio_interval(rows,numerator):
    rows=[r for r in rows if r['unconstrained_rescuable']>0]
    groups=sorted({r['group'] for r in rows})
    if not rows:
        return dict(ratio=None,low=None,high=None,images=0,groups=0,numerator=0,denominator=0)
    grouped=[[(r[numerator],r['unconstrained_rescuable']) for r in rows if r['group']==g] for g in groups]
    rng=np.random.default_rng(20261030); samples=[]
    for _ in range(2000):
        draw=rng.integers(0,len(groups),len(groups))
        selected=[v for j in draw for v in grouped[j]]
        samples.append(sum(v[0] for v in selected)/sum(v[1] for v in selected))
    num=sum(r[numerator] for r in rows); den=sum(r['unconstrained_rescuable'] for r in rows)
    return dict(ratio=num/den,low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),
                images=len(rows),groups=len(groups),numerator=num,denominator=den)

def main():
    freeze=json.loads((ROOT/'freeze_manifest.json').read_text())
    for row in freeze['files']:
        assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['sha256']
    protocol=json.loads((ROOT/'protocol.json').read_text())
    inputs=json.loads((ROOT/'input_manifest.json').read_text())
    assert len(inputs)==100
    print(json.dumps(dict(status='A3_SELF_CHECKS',result=selfcheck(),pid=os.getpid())),flush=True)
    with (ROOT/'started.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),started_unix=time.time()),f)
    started=time.monotonic()
    primary_ids=set(protocol['primary_uids']); seen_primary=set()
    per_image=[]; per_gt=[]; total_gt=0; hashes=[]
    for index,item in enumerate(inputs):
        file=BASE/item['path']
        assert file.resolve().is_relative_to(BASE.resolve())
        assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
        with np.load(file,allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files}
        gt=a['gt_boxes']; total_gt+=len(gt)
        pred=a['decoder_3_boxes'].astype(np.float64)*640.
        original_iou,center_iou=matrices(gt,pred)
        primary=np.array([str(uid) in primary_ids for uid in a['gt_uids']],dtype=bool)
        seen_primary.update(str(uid) for uid,p in zip(a['gt_uids'],primary) if p)
        assert np.all(original_iou.max(1)[primary]<.5)
        token=a['topk_token_ids']; c=(gt[:,:2]+gt[:,2:])*.5
        cell=np.floor(c/8).astype(int)
        local=(token[None,:]<6400)&(np.abs(token[None,:]//80-cell[:,1,None])<=1)&(np.abs(token[None,:]%80-cell[:,0,None])<=1)
        geometry=np.ones_like(original_iou,dtype=bool)
        argmax=a['decoder_3_logits'].argmax(-1)[None,:]==a['gt_labels'][:,None]
        pp=np.zeros((300,10),bool)
        pp[a['native_pp_query_ids'],a['native_pp_labels']]=True
        native_post=pp[:,a['gt_labels']].T
        for mode,eligible in [('native_post300',native_post),('class_argmax',argmax),('geometry',geometry)]:
            original=(original_iou>=.5)&eligible
            oracle=(center_iou>=.5)&local&eligible
            result=audit(original,oracle,original_iou,primary)
            name=file.stem+'.jpg'
            rec=dict(file_name=name,group=file.stem.split('_')[0],mode=mode,primary_gt=int(primary.sum()),
                     baseline_one_to_one_gt=len(result['base_gt']),unconstrained_rescuable=len(result['unconstrained_gt']),
                     rescued_with_baseline_pairs_fixed=len(result['free_gt']),
                     rescued_with_baseline_coverage_fixed=len(result['flex_gt']),
                     baseline_gt_retained=result['protected_retained'],free_queries=result['free_queries'])
            assert rec['baseline_gt_retained']==rec['baseline_one_to_one_gt']
            per_image.append(rec)
            sets={key:set(result[key].tolist()) for key in ('base_gt','unconstrained_gt','free_gt','flex_gt')}
            for g in np.where(primary)[0]:
                per_gt.append(dict(uid=str(a['gt_uids'][g]),file_name=name,mode=mode,
                    sqrt_area=float(a['gt_sqrt_area'][g]),local_query_count=int(local[g].sum()),
                    unconstrained_rescuable=g in sets['unconstrained_gt'],rescued_with_pairs_fixed=g in sets['free_gt'],
                    rescued_with_coverage_fixed=g in sets['flex_gt']))
        hashes.append(dict(path=item['path'],sha256=item['sha256']))
        if (index+1)%25==0:
            print(json.dumps(dict(status='A3_RUNNING',images=index+1)),flush=True)
    assert total_gt==4491 and seen_primary==primary_ids and len(primary_ids)==470
    estimates={}
    for mode in ('native_post300','class_argmax','geometry'):
        rows=[r for r in per_image if r['mode']==mode]
        estimates[mode]=dict(baseline_one_to_one_gt=sum(r['baseline_one_to_one_gt'] for r in rows),
            pairs_fixed=ratio_interval(rows,'rescued_with_baseline_pairs_fixed'),
            coverage_fixed=ratio_interval(rows,'rescued_with_baseline_coverage_fixed'))
    p=estimates['native_post300']['pairs_fixed']
    enough=p['denominator']>=100 and p['groups']>=15
    if enough and p['ratio']>=.8 and p['low']>=.65:
        decision='MOST_ORACLE_RESCUE_USES_SPARE_QUERIES_DO_NOT_JUSTIFY_COMPLEX_OWNERSHIP_MODULE_FROM_THIS_EVIDENCE'
    elif enough and p['ratio']<=.5 and p['high']<=.7:
        decision='ORACLE_QUERY_RESOURCE_CONFLICT_IS_MATERIAL_REQUIRES_CORRESPONDENCE_DIAGNOSIS_NOT_YET_A_NEW_METHOD'
    else:
        decision='INCONCLUSIVE_FOR_QUERY_RESOURCE_CONFLICT_NO_ROUTING_MODULE_APPROVED'
    summary=dict(status='A3_ORACLE_CAPACITY_AUDIT_COMPLETE',images=100,gt=4491,primary_gt=470,
                 estimates=estimates,primary_mode='native_post300',enough_primary_evidence=enough,decision=decision,
                 elapsed_seconds=time.monotonic()-started,new_model_forwards=0,new_fits=0,main_model_updates=0,
                 global_model_forwards_unchanged=185,
                 scope='GT-oracle audit at finaldecoder IoU0.5,original score/class selection fixed,conditional A2 cohort;not AP or a deployable intervention',
                 limitations=['A canonical maximum-cardinality baseline matched GT set is protected;this does not mean every any-box-covered GT can be protected simultaneously.',
                              'Free-query identities are a coverage audit construct,not true instance ownership or the original training assignment.',
                              'GT-specific box center corrections remain oracles;holding scores fixed does not prove AP improvement.',
                              'Author checkpoint saw these training images;filename groups are a correlation-reduction heuristic.'])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (ROOT/'per_image.json').write_text(json.dumps(per_image,indent=2,allow_nan=False)+'\n')
    with (ROOT/'per_gt.jsonl').open('w') as f:
        for row in per_gt: f.write(json.dumps(row,allow_nan=False)+'\n')
    (ROOT/'input_verification.json').write_text(json.dumps(hashes,indent=2)+'\n')
    paths=[ROOT/name for name in ['protocol.json','input_manifest.json','freeze_manifest.json','analyze_capacity.py',
        'started.json','summary.json','per_image.json','per_gt.jsonl','input_verification.json']]
    manifest=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (ROOT/'result_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'A3_capacity_audit.tar.gz'
    with tarfile.open(archive,'w:gz') as target:
        for path in paths+[ROOT/'result_manifest.json']: target.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='A3_COMPLETE_ARCHIVED',bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 files=len(paths),primary=p,decision=decision,new_model_forwards=0,new_fits=0)
    (ROOT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,allow_nan=False),flush=True)

if __name__=='__main__': main()
