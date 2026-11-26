"""Read-only negative replay artifact verification and query-order diagnostics."""
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile, datetime, collections
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/p2_seed1_replay_mismatch_collection12_20261011_v1'
OLD=ROOT/'outputs/user_backbone_primary_collection12_20261011_v2/seed1_p2_skip_v1/extracted/run'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
inp=read(ROOT/'outputs/p2_seed1_replay_mismatch_collection12_20261011_v1_inputs.json')
raw=inp['payload'].encode(); assert len(raw)==inp['transport']['bytes'] and sha(raw)==inp['transport']['sha256']
meta=json.loads(raw); archive=OUT/'failed_replay_artifacts.zip'
assert archive.stat().st_size==meta['bytes'] and sha(archive.read_bytes())==meta['sha256']
EX=OUT/'extracted'; EX.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==meta['entries'] and z.testzip() is None
    mraw=z.read('manifest.json'); assert sha(mraw)==meta['manifest_sha256']; m=json.loads(mraw)
    assert m['status']=='FAILED_REPLAY_ARTIFACTS_PRESERVED_NOT_RECLASSIFIED_AS_PASS'
    assert set(z.namelist())==set(m['files'])|{'manifest.json'}
    for n in z.namelist():
        p=PurePosixPath(n); assert not p.is_absolute() and '..' not in p.parts and '\\' not in n
        data=z.read(n)
        if n!='manifest.json': assert len(data)==m['files'][n]['bytes'] and sha(data)==m['files'][n]['sha256']
        dest=EX/n; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
NEW=EX/'candidate_reload_v1'; proof=read(NEW/'verification.json')
assert proof['status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY' and proof['strict_loaded_tensors']==801
assert proof['checkpoint_sha256']==read(OLD/'training_completion.json')['last_checkpoint_sha256']
a=read(OLD/'endpoint_epoch12/summary.json'); b=read(NEW/'endpoint_epoch12/summary.json')
assert a['identity']==b['identity']==proof['identity']
assert a['native_stats']==proof['original_stats'] and b['native_stats']==proof['reloaded_stats']
fields=['encoder_initial_boxes','pre_after_first_layer_boxes','pre_after_first_layer_logits','final_boxes','final_logits']
same_slots=collections.Counter(); same_unordered=collections.Counter(); unordered_joint=0
errors={f:[] for f in fields}; geometry_errors=[]; mutual_counts=[]; exactboxrows=[]; summaries=[]
for iid in range(548):
    name=f'image_{iid:06d}.npz'
    with np.load(OLD/'stages_epoch12'/name,allow_pickle=False) as az,np.load(NEW/'stages_epoch12'/name,allow_pickle=False) as bz:
        assert set(az.files)==set(bz.files)==set(fields)|{'original_hw'}
        av={f:az[f].copy() for f in fields}; bv={f:bz[f].copy() for f in fields}
        assert np.array_equal(az['original_hw'],bz['original_hw'])
    for f in fields:
        assert av[f].shape==bv[f].shape and av[f].dtype==bv[f].dtype and len(av[f])==300
        assert np.isfinite(av[f]).all() and np.isfinite(bv[f]).all()
        same_slots[f]+=int(np.array_equal(av[f],bv[f]))
        rows=lambda x:collections.Counter(row.tobytes() for row in x)
        same_unordered[f]+=int(rows(av[f])==rows(bv[f]))
    unordered_joint+=int(rows(np.concatenate(list(av.values()),axis=1))==rows(np.concatenate(list(bv.values()),axis=1)))
    aa=av[fields[0]].astype('float64'); bb=bv[fields[0]].astype('float64')
    # Mutual nearest geometry is a diagnostic association, not query identity.
    distances=np.max(np.abs(aa[:,None,:]-bb[None,:,:]),axis=2)
    nearest=distances.argmin(axis=1); reverse=distances.argmin(axis=0)
    ii=np.arange(300); valid=reverse[nearest]==ii; indices=ii[valid]; jj=nearest[valid]
    ties=(distances==distances.min(axis=1,keepdims=True)).sum(axis=1)
    mutual_counts.append(int(valid.sum())); exactboxrows.append(int((distances.min(axis=1)==0).sum()))
    geo=distances[indices,jj]; geometry_errors.extend(geo.tolist())
    for f in fields:errors[f].extend(np.max(np.abs(av[f][indices].astype('float64')-bv[f][jj].astype('float64')),axis=1).tolist())
    summaries.append(dict(image_id=iid,mutual_pairs=int(valid.sum()),nearest_initial_box_exact_rows=exactboxrows[-1],nearest_initial_box_tie_rows=int((ties>1).sum()),changed_position_pairs=int((indices!=jj).sum()),max_mutual_initial_box_error=float(geo.max()) if len(geo) else None))
def quant(x):
    return dict(count=len(x),max=float(np.max(x)),quantiles=dict(zip(['median','p90','p99','p999'],map(float,np.quantile(x,[.5,.9,.99,.999])))))
ga=np.load(OLD/'endpoint_epoch12/GT_hits.npz',allow_pickle=False); gb=np.load(NEW/'endpoint_epoch12/GT_hits.npz',allow_pickle=False)
assert np.array_equal(ga['annotation_id'],gb['annotation_id']) and np.array_equal(ga['category_id'],gb['category_id'])
gains=(~ga['hits'])&gb['hits']; losses=ga['hits']&~gb['hits']
r=dict(status='NEGATIVE_REPLAY_ARTIFACTS_VERIFIED_AND_QUERY_ORDER_DIAGNOSTIC_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 archive_sha256=meta['sha256'],archive_files_verified=len(m['files']),source_checkpoint_sha256=proof['checkpoint_sha256'],original_replay_failure_preserved=True,
 original_percent=dict(AP=100*a['native_stats'][0],APs=100*a['native_stats'][3]),reloaded_percent=dict(AP=100*b['native_stats'][0],APs=100*b['native_stats'][3]),
 delta_points=dict(AP=100*(b['native_stats'][0]-a['native_stats'][0]),APs=100*(b['native_stats'][3]-a['native_stats'][3])),
 images=548,queries=548*300,exact_same_slot_images=dict(same_slots),exact_unordered_field_images=dict(same_unordered),exact_unordered_joint_images=unordered_joint,
 mutual_nearest_initial_geometry_pairs=sum(mutual_counts),all300_mutual_pair_images=sum(v==300 for v in mutual_counts),nearest_initial_box_exact_rows=sum(exactboxrows),
 changed_position_mutual_pairs=sum(v['changed_position_pairs'] for v in summaries),mutual_geometry_errors=quant(geometry_errors),mutual_pairs_field_max_errors={f:quant(v) for f,v in errors.items()},
 native_GT_gains_by_IoU=gains.sum(axis=0).tolist(),native_GT_losses_by_IoU=losses.sum(axis=0).tolist(),
 root_cause_proven=False,exact_replay_passed=False,matching_is_query_identity=False,new_model_forwards=0,new_native_evaluator_calls=0,new_training_epochs=0,new_BASE_forwards=0)
(OUT/'query_order_diagnostic.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
(OUT/'query_order_per_image.json').write_text(json.dumps(summaries,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps(r))
