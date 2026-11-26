"""One sparse local log-field Taylor control. NumPy only;no learned models fitted."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
from datetime import datetime,timezone
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def taylor(field):
    field=np.asarray(field,dtype=np.float64).reshape(3,3)
    assert np.isfinite(field).all()
    if field[1,1]<=1e-12:
        return np.zeros(2),'nonpositive_center'
    if field[1,1]<np.max(field)-1e-12:
        return np.zeros(2),'not_local_peak'
    f=np.log(np.maximum(field,1e-12));h=8.
    grad=np.array([(f[1,2]-f[1,0])/(2*h),(f[2,1]-f[0,1])/(2*h)])
    mixed=(f[2,2]-f[2,0]-f[0,2]+f[0,0])/(4*h*h)
    H=np.array([[(f[1,2]-2*f[1,1]+f[1,0])/(h*h),mixed],
                [mixed,(f[2,1]-2*f[1,1]+f[0,1])/(h*h)]])
    if np.linalg.eigvalsh(H).max()>=-1e-6:
        return np.zeros(2),'nonconcave_or_flat'
    raw=-np.linalg.solve(H,grad)
    clipped=np.clip(raw,-2.,2.)
    return clipped,'accepted_clipped' if np.any(raw!=clipped) else 'accepted'

def selfcheck():
    yy,xx=np.meshgrid([-8.,0.,8.],[-8.,0.,8.],indexing='ij')
    grid=np.stack([xx,yy],axis=-1)
    for dx in (-1.9,0.,1.9):
        for dy in (-1.9,0.,1.9):
            target=np.array([dx,dy])
            field=np.exp(-np.sum((grid-target)**2,axis=-1)/8.)
            estimate,reason=taylor(field)
            assert reason=='accepted'
            np.testing.assert_allclose(estimate,target,rtol=0,atol=1e-11)
    assert taylor(np.ones((3,3)))[1]=='nonconcave_or_flat'
    assert taylor(-np.ones((3,3)))[1]=='nonpositive_center'
    corner=np.zeros((3,3));corner[0,0]=1.;corner[1,1]=.5
    assert taylor(corner)[1]=='not_local_peak'
    return 'PASS9_known_Gaussian_offsets_xy_units_flat_negative_and_nonpeak_fallbacks'

def group_ci(rows,key,iterations=2000,seed=20261030):
    kept=[r for r in rows if r.get(key) is not None]
    if not kept:
        return dict(mean=None,low=None,high=None,images=0,groups=0)
    groups=sorted({r['group'] for r in kept});values=[[r[key] for r in kept if r['group']==g] for g in groups]
    rng=np.random.default_rng(seed)
    boot=[np.mean([v for i in rng.integers(len(groups),size=len(groups)) for v in values[i]]) for _ in range(iterations)]
    return dict(mean=float(np.mean([r[key] for r in kept])),low=float(np.quantile(boot,.025)),
                high=float(np.quantile(boot,.975)),images=len(kept),groups=len(groups))

def main():
    frozen=json.loads((ROOT/'freeze_manifest.json').read_text(encoding='utf-8'))
    for item in frozen['files']:
        assert sha(ROOT/item['file'])==item['sha256']
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    assert p['status']=='FROZEN_BEFORE_A5_FIELD_EVALUATION'
    (ROOT/'started.json').open('x',encoding='utf-8').write(json.dumps(dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat())))
    checks=selfcheck();began=time.monotonic()
    assert sha(Path(p['source_cohort']))==p['source_cohort_sha256']
    heads={}
    for item in p['source_A1_heads']:
        path=Path(item['path']);assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:
            heads[path.stem]={key:z[key].copy() for key in z.files}
    image_specs={r['file_name']:r for r in p['fixed_images']}
    rows=[];pure_max_error=0.
    for item in p['source_features']:
        path=Path(item['path']);assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:
            name=str(z['file_name']);spec=image_specs[name];uids=set(spec['uids'])
            assert str(z['group'])==spec['group'] and int(z['fold'])==spec['fold']
            inverse={int(flat):i for i,flat in enumerate(z['fine_flat_ids'])}
            for g,uid in enumerate(z['gt_uids']):
                if str(uid) not in uids:
                    continue
                c=int(z['gt_center_index'][g]);rr,cc=divmod(int(z['fine_flat_ids'][c]),160)
                ids=[inverse[(rr+dr)*160+cc+dc] for dr in (-2,0,2) for dc in (-2,0,2)]
                center=(z['gt_boxes'][g,:2].astype(np.float64)+z['gt_boxes'][g,2:].astype(np.float64))*.5
                base=z['xy'][c].astype(np.float64);target=center-base
                coordinates=z['xy'][ids].astype(np.float64)
                pure=np.exp(-np.sum((coordinates-center)**2,axis=1)/8.)
                exact,_=taylor(pure)
                pure_error=float(np.linalg.norm(exact-target));pure_max_error=max(pure_max_error,pure_error)
                assert pure_error<2e-10
                oracle,oracle_reason=taylor(z['target'][ids])
                row=dict(uid=str(uid),file_name=name,group=spec['group'],fold=spec['fold'],size=float(z['gt_sqrt_area'][g]),
                         target_offset=target.tolist(),zero_error=float(np.linalg.norm(target)),
                         oracle_field_error=float(np.linalg.norm(oracle-target)),oracle_reason=oracle_reason,pure_error=pure_error)
                for view in p['views']:
                    state=heads[f'head_fold{spec["fold"]}_{view}']
                    features=z['features_'+view][ids].astype(np.float64)
                    scores=np.column_stack([(features-state['mean'])/state['std'],np.ones(9)])@state['coef']
                    values={}
                    for index,label in enumerate(('real','null')):
                        delta,reason=taylor(scores[:,index])
                        values[label+'_error']=float(np.linalg.norm(delta-target))
                        values[label+'_offset']=delta.tolist();values[label+'_reason']=reason
                    row[view]=values
                rows.append(row)
    assert len(rows)==sum(r['eligible'] for r in p['fixed_images'])
    per_image=[]
    for spec in p['fixed_images']:
        image=dict(file_name=spec['file_name'],group=spec['group'],fold=spec['fold'])
        these=[r for r in rows if r['file_name']==spec['file_name']]
        for cohort in ('lt16','lt8','from8to16'):
            selected=[r for r in these if cohort=='lt16' or (r['size']<8 if cohort=='lt8' else r['size']>=8)]
            image[cohort+'_n']=len(selected)
            if not selected:
                continue
            image[cohort+'_zero_error']=float(np.mean([r['zero_error'] for r in selected]))
            image[cohort+'_oracle_error']=float(np.mean([r['oracle_field_error'] for r in selected]))
            for view in p['views']:
                image[cohort+'_'+view+'_real_error']=float(np.mean([r[view]['real_error'] for r in selected]))
                image[cohort+'_'+view+'_null_error']=float(np.mean([r[view]['null_error'] for r in selected]))
                image[cohort+'_'+view+'_advantage_zero']=float(np.mean([r['zero_error']-r[view]['real_error'] for r in selected]))
                image[cohort+'_'+view+'_advantage_null']=float(np.mean([r[view]['null_error']-r[view]['real_error'] for r in selected]))
        per_image.append(image)
    keys=sorted({key for r in per_image for key in r if key not in ('file_name','group','fold') and not key.endswith('_n')})
    estimates={key:group_ci(per_image,key) for key in keys}
    counts={key:sum(r[key+'_n'] for r in per_image) for key in ('lt16','lt8','from8to16')}
    enough=counts['lt16']>=200 and min(counts['lt8'],counts['from8to16'])>=50 and estimates['lt16_P3_after_advantage_zero']['groups']>=8
    positive=bool(enough and all(estimates['lt16_P3_after_advantage_'+key]['mean']>=.15 and estimates['lt16_P3_after_advantage_'+key]['low']>0 for key in ('zero','null'))
                  and all(estimates[cohort+'_P3_after_advantage_zero']['mean']>0 for cohort in ('lt8','from8to16')))
    reasons={view:{label:{reason:sum(r[view][label+'_reason']==reason for r in rows) for reason in ('accepted','accepted_clipped','nonpositive_center','not_local_peak','nonconcave_or_flat')}
                   for label in ('real','null')} for view in p['views']}
    result=dict(status='A5_SPARSE_LOCAL_TAYLOR_CONTROL_COMPLETE',decision='EXISTING_ANALYTIC_CONTROL_RECOVERS_CONDITIONAL_PRECISION_NOT_A_NEW_METHOD' if positive else
                'NO_POSITIVE_LOCAL_TAYLOR_CONTROL_DO_NOT_BUILD_GENERIC_CENTER_REFINEMENT',
                counts=counts,estimates=estimates,enough_primary_evidence=bool(enough),positive_gate=positive,reason_counts=reasons,
                numerical_selfcheck=checks,pure_Gaussian_max_error=pure_max_error,new_fits=0,new_forwards=0,new_detector_updates=0,
                global_full_forwards_unchanged=185,existing_architecture_diagnostic_models_unchanged=64,existing_scalar_heads_unchanged=96,
                numpy_version=np.__version__,elapsed_seconds=time.monotonic()-began,limitations=p['limitations'],
                not_full_DARK_reproduction=True,scientific_mainline_validated=False)
    write(ROOT/'summary.json',result);write(ROOT/'per_image.json',per_image)
    with (ROOT/'per_gt.jsonl').open('w',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write(ROOT/'completed.json',dict(status='TERMINAL_SUCCESS',summary_sha256=sha(ROOT/'summary.json'),new_fits=0,new_forwards=0))
    print(json.dumps(dict(status=result['status'],decision=result['decision'],counts=counts,
                         primary_zero_advantage=estimates['lt16_P3_after_advantage_zero'],primary_null_advantage=estimates['lt16_P3_after_advantage_null'],
                         pure_Gaussian_max_error=pure_max_error,new_fits=0,new_forwards=0)))

if __name__=='__main__':
    main()
