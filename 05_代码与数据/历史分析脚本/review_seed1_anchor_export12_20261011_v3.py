"""Correct copied audit metadata using immutable downloaded actual arrays; no model calls."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent.parent;D=R/'outputs/native_selection_anchor_audit12_20261011_v2/seed1';O=R/'outputs/native_selection_anchor_audit12_20261011_v3/seed1'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=read(R/'outputs/reviewed_remote_tests_actual12_20261011_v4_collection_inputs.json')['turn3_A_anchor_export_metadata_seed1'];raw=c['payload'].encode();assert hashlib.sha256(raw).hexdigest()==c['transport']['sha256'] and len(raw)==c['transport']['bytes']
meta=json.loads(raw);assert {p.name for p in D.iterdir()}==set(meta['files'])
for n,x in meta['files'].items():assert sha(D/n)==x['sha256'] and (D/n).stat().st_size==x['bytes']
old=read(D/'report.json');O.mkdir(parents=True,exist_ok=False);cases={}
nominal=[];sizes=[]
for h,w in [(80,80),(40,40),(20,20)]:
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');nominal.append(np.stack([(xx+.5)/w,(yy+.5)/h],-1).reshape(-1,2));sizes.append(np.tile([w,h],(h*w,1)))
nominal=np.concatenate(nominal);sizes=np.concatenate(sizes)
for arm,item in old['cases'].items():
    with np.load(D/(arm+'.npz'),allow_pickle=False) as z:a,m,g=[z[n].copy() for n in ['EMA_anchors','model_anchors','generated_anchors']]
    assert all(v.shape==(8400,4) and v.dtype==np.float32 for v in [a,m,g])
    assert hashlib.sha256(a.tobytes()).hexdigest()==item['array_sha256']
    assert all(np.array_equal(np.isfinite(a),np.isfinite(v)) for v in [m,g])
    good=np.isfinite(a).all(1)
    for v in [a,m,g]:
        xy=1/(1+np.exp(-v[good,:2].astype('float64')))
        assert np.array_equal(np.floor(xy*sizes[good]),np.floor(nominal[good]*sizes[good]))
    same=np.array_equal(m,g);difference=float(np.max(np.abs(m[good].astype('float64')-g[good].astype('float64'))))
    assert same==item['model_equal_generated'] and difference==item['model_generated_max_abs_difference']
    corrected={k:v for k,v in item.items() if k!='EMA_equal_to_original_export'}
    corrected.update(original_v1_export_crosscheck_performed=False,nominal_feature_cells_exact_for_EMA_model_and_generated=True,export_file=str(D/(arm+'.npz')))
    cases[arm]=corrected
r=dict(status='VERIFIED_ACTUAL_EMA_EXPORT_AND_FEATURE_CELLS_CPU_MODEL_GENERATOR_DIFFERENCE_RECORDED',seed=1,cases=cases,original_audit_report_sha256=sha(D/'report.json'),
    unsupported_literal_field_removed='EMA_equal_to_original_export was copied as True after removing a missing seed1 prior-export comparison; that comparison was not performed.',
    original_source_preserved=True,original_CPU_report_preserved=True,model_and_generated_arrays_exact=False,all_EMA_model_generated_nominal_feature_cells_exact=True,
    no_tolerance_increased=True,model_forwards=0,evaluator_calls=0,training_epochs=0,original_checkpoint_reload_pass_not_reclassified=True,
    source_sha256=sha(__file__),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
(O/'report.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(dict(status=r['status'],exact_CPU_model_generator=False,nominal_feature_cells_exact=True,source_preserved=True)))
