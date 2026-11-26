"""Transfer integrity and independent moment/pair arithmetic from raw cached predictions."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib
import json
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent
A11=ROOT/'A11_坐标插值相位与分布扩散审计_v1'
A12=ROOT/'A12_原生边界分布诊断_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
done11=read(A11/'completed.json')
assert sha(ROOT/'a11_coordinate_phase_math.py')==done11['script_sha256']
assert sha(A11/'protocol.json')==done11['protocol_sha256']
for r in done11['files']:assert sha(A11/r['path'])==r['sha256']
receipt=read(A12/'backup'/'receipt.json')
archive=A12/'backup'/'A12_native_distribution_results.tar.gz'
assert sha(archive)==receipt['sha256']
manifest=read(A12/'backup'/'file_manifest.json')['files']
with tarfile.open(archive) as tar:
    assert {m.name for m in tar.getmembers()}=={r['path'] for r in manifest}
    for r in manifest:
        rel=PurePosixPath(r['path']);member=tar.getmember(r['path'])
        assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
        raw=tar.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==r['sha256']
        dest=A12.joinpath(*rel.parts)
        if dest.exists():assert dest.read_bytes()==raw
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(raw)
done=read(A12/'completed.json')
for r in done['files']:assert sha(A12/r['path'])==r['sha256']
edges=read(A12/'edge_rows.json');pairs=read(A12/'pair_rows.json')
by_image={i:[r for r in edges if r['image']==i] for i in range(12)}
index={};max_mean_error=0.;moment_checks=0
for i in range(12):
    with np.load(A12/'predictions'/('case%02d.npz'%i),allow_pickle=False) as z:
        logits=z['corner_logits'].astype(np.float64);grid=z['bin_grid'].astype(np.float64)
        p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        mean=np.einsum('lqsk,k->lqs',p,grid)
        max_mean_error=max(max_mean_error,float(np.max(np.abs(mean-z['source_means']))))
        assert max_mean_error<=2e-6
        ref=z['ref_initial'].astype(float)
        second=np.einsum('qsk,k->qs',p[-1],grid**2)
        raw_var=second-mean[-1]**2
        b=z['raw_boxes'].astype(float)
        pixel=np.concatenate((b[:,:2]-b[:,2:]*.5,b[:,:2]+b[:,2:]*.5),axis=-1)*800
        gt=z['gt_boxes800'].astype(float)
        for r in by_image[i]:
            q=r['query'];side=r['side'];left=r['interval']
            axis=side%2;physical=ref[q,2+axis]*200.
            m=mean[-1,q,side]
            # Equivalent lower-convex-envelope formula,independent of V-minus-floor implementation.
            ex=second[q,side]-((grid[left]+grid[left+1])*m-grid[left]*grid[left+1])
            values={'raw_variance_px2':raw_var[q,side]*physical**2,'excess_variance_px2':max(ex,0.)*physical**2}
            for name,value in values.items():assert abs(value-r[name])<=1e-7
            assert abs(abs(pixel[q,side]-gt[r['gt_index'],side])-r['edge_error_px'])<=1e-7
            row=dict(r);row.update(values);index[(i,r['gt_index'],side)]=row;moment_checks+=2
expected_ids=set()
for i,rows in by_image.items():
    for good in rows:
        if good['edge_error_px']>.5:continue
        for bad in rows:
            if bad['edge_error_px']<1:continue
            if (good['side'],good['interval'],good['class_id'])!=(bad['side'],bad['interval'],bad['class_id']):continue
            if max(good['bin_width_px'],bad['bin_width_px'])>2*min(good['bin_width_px'],bad['bin_width_px']):continue
            expected_ids.add((i,good['gt_index'],bad['gt_index'],good['side'],good['interval'],good['class_id']))
actual_ids={(r['image'],r['good_gt'],r['bad_gt'],r['side'],r['interval'],r['class_id']) for r in pairs}
assert expected_ids==actual_ids and len(pairs)==270
ranking_differences=0
for r in pairs:
    good=index[(r['image'],r['good_gt'],r['side'])];bad=index[(r['image'],r['bad_gt'],r['side'])]
    for key in ('raw_variance_px2','excess_variance_px2'):
        v=.5 if bad[key]==good[key] else float(bad[key]>good[key])
        assert v==r[key]
    ranking_differences+=r['raw_variance_px2']!=r['excess_variance_px2']
assert ranking_differences==0
s=read(A12/'summary.json');a=read(A12/'analysis_summary.json')
assert s['new_full_model_forwards']==13 and s['cumulative_full_model_forwards']==213
assert s['state_digest_before']==s['state_digest_after'] and s['identity_max_difference']==0
assert a['primary_excess_minus_raw_variance']['mean']==0 and len(edges)==1023
result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_RAW_ARCHIVE_SOURCE_MOMENTS_AND_ALL_PRIMARY_PAIR_FLAGS','archive_members_verified':len(manifest),'completion_manifest_files_verified':len(done['files']),'raw_prediction_files':12,'source_mean_max_abs_error':max_mean_error,'physical_variance_moment_values_checked':moment_checks,'eligible_edge_rows':len(edges),'all_pair_eligibility_and_scores_checked':len(pairs),'different_raw_vs_corrected_rankings':ranking_differences,'model_forwards_in_verifier':0,'fits_in_verifier':0,'scope':'Integrity,raw-logit source mean reconstruction,independent second-moment/convex-envelope formulas,all pair eligibility andprimary flags. Not a new algorithm,AP,causal effect or independent dataset replication.'}
with (A12/'independent_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
