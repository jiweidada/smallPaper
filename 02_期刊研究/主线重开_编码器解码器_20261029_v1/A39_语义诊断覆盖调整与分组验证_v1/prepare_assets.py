from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys,zipfile
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent;OLD=B/'A38_细尺度与查询语义成对诊断_v1'
REMOTE='/root/c10_region_restore_20261029_v1/architecture_probe_v1/a39_semantic_coverage_20261031_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    with (D/name).open('x',encoding='utf-8') as output:json.dump(value,output,ensure_ascii=False,indent=2);output.write('\n')
prior=read(B/'87_A38冻结采集与独立覆盖核验回执.json')
for p,h in prior['files_sha256'].items():assert sha(p)==h,p
plan=read(OLD/'coverage_followup_sample_DRAFT.json')
assert plan['case_counts']=={'train':119,'val':105} and plan['prospective_new_unique_images']==112
assert read(OLD/'analysis/semantic_analysis_result.json')['probe_fit_entries']==0
for name in ['semantic_math.py','pool_tensor.py','npz_export.py','pinned_dome_loader.py']:
    with (D/name).open('xb') as f:f.write((OLD/name).read_bytes())
cases=[];new_cases=[]
z=D/'new_val_inputs_bundle.zip'
with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED) as archive:
    for r in plan['cases']:
        assert sha(r['image_path'])==r['image_sha256'] and sha(r['annotation_path'])==r['annotation_sha256']
        row=dict(r);row['index']=r['prospective_index']
        if r['reuse_A38_capture']:
            row.update(capture_phase='A38',capture_index=r['A38_index'],capture_path=r['A38_capture'],capture_sha256=sha(r['A38_capture']))
        else:
            row.update(capture_phase='A39',capture_index=row['index'],capture_relative='captures/case_%03d.npz'%row['index'])
            base='/root/c10_region_restore_20261029_v1/dataset/train' if r['split']=='train' else REMOTE+'/inputs/val'
            rr={k:row[k] for k in ['index','split','prefix','file','width','height','image_sha256','annotation_sha256']}
            rr.update(image_path=base+'/images/'+r['file'],annotation_path=base+'/annotations/'+Path(r['annotation_path']).name)
            new_cases.append(rr)
            if r['split']=='val':
                archive.write(r['image_path'],'inputs/val/images/'+r['file'])
                archive.write(r['annotation_path'],'inputs/val/annotations/'+Path(r['annotation_path']).name)
        cases.append(row)
assert len(cases)==224 and len(new_cases)==112
assert sum(r['split']=='train' for r in new_cases)==55 and sum(r['split']=='val' for r in new_cases)==57
save('combined_cases_FROZEN.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),cases=cases,
    counts={'train':119,'val':105},prefix_counts={'train':64,'val':64},reused=112,new_unique=112,
    selection_source=str(OLD/'coverage_followup_sample_DRAFT.json'),selection_source_sha256=sha(OLD/'coverage_followup_sample_DRAFT.json'),
    adjustment_basis='A38count-onlysupportfailure78<100beforeanydataprobefitorsemanticAUC;notperformance-guided.',
    model_call_allocation=0,data_probe_fit_allocation=0))
save('collector_inputs.json',dict(remote_root=REMOTE,cases=new_cases,archive_sha256=sha(z),archive_bytes=z.stat().st_size,
    unique_new_cases=112,repeat_case_index=new_cases[0]['index']))
groups=sorted({r['prefix'] for r in cases if r['split']=='val'});assert len(groups)==64
save('bootstrap_groups_FROZEN.json',dict(groups=groups,unit='Filename-prefixproxygroups,allselectedframesandpairsofadrawngroupareduplicated.'))
draws=np.random.default_rng(381031).integers(0,64,size=(1000,64),dtype=np.int64)
with (D/'bootstrap_draws_FROZEN.npz').open('xb') as f:np.savez_compressed(f,draws=draws)
save('handoff_and_sample_receipt.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
    previous_goal_turn_classification='PROGRESS_A38native114entries,112imagecaches,andindependentlyverifiedsupportcount78;noeffectestimate.',
    previous_A38_files_verified=227,old_A38_cap_closed=True,old_A38_new_model_calls=0,
    old_semantic_data_probe_fits=0,all_cases=224,additional_cases=112,
    model_identity={'commit':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861','checkpoint_sha256':'42874302fd97e17c2a23a46a8ec35d2ea4d2ffe479f0d632c1327fafbf910eb7',
        'loader_sha256':'8dc66cf817d387733a895501dc160dad703eedad9d061ce8f4ffd95296b7c1ab',
        'state_digest':'82c3ba49d69370a3d0105e47fc0ab5d715a9e7b1ad15feb275b1b497b5b0ccce'},
    unchanged_math_sha256=sha(D/'semantic_math.py'),unchanged_pool_tensor_sha256=sha(D/'pool_tensor.py'),
    standard_npz_writer_sha256=sha(D/'npz_export.py'),native_model_allocation_so_far=0))
print(json.dumps(dict(status='A39_SAMPLES_AND_REUSE_PINNED_NO_NEW_MODEL',cases=224,reused=112,new_train=55,new_val=57,
    val_bundle_bytes=z.stat().st_size,val_bundle_sha256=sha(z)),ensure_ascii=False))
