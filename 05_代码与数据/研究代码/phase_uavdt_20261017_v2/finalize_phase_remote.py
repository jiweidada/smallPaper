"""Freeze the accepted new experiment, using measured capacity and runtime contract proofs."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

root=Path(CONFIG['root']);assert root.is_absolute()
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
assert not (root/'final_acceptance.json').exists() and not (root/'launch_receipt.json').exists()
prepared=read(root/'preparation_receipt.json');assert prepared['state']=='CPU_verified_pending_capacity_and_finalization'
assert prepared['official_reference']['author_reference_full_curve_passed'] and prepared['C_Cmean_initial_parameters_equal']
runtime=read(root/'runtime_contract_receipt.json');assert runtime['state']=='passed' and runtime['formal_training_updates']==0
assert read(root/'runtime_contract_worker_result.json')['returncode']==0
capacity=read(root/'capacity_receipt.json');assert capacity['state']=='passed'
assert read(root/'capacity_worker_result_v2.json')['returncode']==0
patch=read(root/'entry_path_guard_patch_receipt.json')
for name,expected in prepared['overlay_sha256'].items():
 actual=sha(root/'code'/name)
 if name=='train_uavdt.py':assert expected==patch['entry_before_sha256'] and actual==patch['entry_after_sha256']
 else:assert actual==expected,'Accepted overlay changed: '+name
assert sha(root/'code/check_runtime_contract.py')==patch['runtime_test_sha256']
plan=read(root/'plan.json');assert set(plan['jobs'])=={'baseline','c','c_mean'} and len(plan['jobs'])==3 and plan['epoches']==96
data=Path(plan['data_root']);ready=read(data/'data_ready.json');assert ready['state']=='complete_data_view_verified'
assert ready['data_metadata_sha256']==plan['data_metadata_sha256']
assert sha(Path(plan['initialization']))==plan['initialization_sha256']==prepared['initialization_sha256']
for split,expected in plan['annotations_sha256'].items():assert sha(data/'annotations'/(split+'.json'))==expected
selected=capacity['selected_concurrency'];assert 1<=selected<=3
case=next(c for c in capacity['cases'] if c['concurrency']==selected);assert case['passed'] and case['minimum_GPU_free_MiB']>=2048
slowest=max(w['seconds_per_step_mean'] for w in case['workers'].values())
estimated_train=slowest*(18990//4)*96
timeout=max(96*3600,math.ceil(estimated_train*2.5+12*3600))
plan['maximum_concurrent_trainings']=selected;plan['training_timeout_seconds']=timeout
plan['measured_capacity_choice']={'concurrency':selected,'aggregate_images_per_second':case['aggregate_images_per_second'],'case_peak_GPU_used_MiB':case['peak_GPU_used_MiB']}
with (root/'artifacts/plan_before_capacity_finalization.json').open('x') as f:json.dump(read(root/'plan.json'),f,indent=2)
temp=root/'plan.next';temp.write_text(json.dumps(plan,indent=2));os.replace(temp,root/'plan.json')
source=Path(CONFIG['old'])/'artifacts/round2_20261014_v1/execution_v3/code';code=root/'code';files=[];engine_python=[]
for path in sorted(code.rglob('*')):
 if not path.is_file() or path.is_symlink() or path.suffix not in ['.py','.yml','.yaml'] or '__pycache__' in path.parts:continue
 rel=str(path.relative_to(code));digest=sha(path)
 original=source/rel
 if original.is_file():assert sha(original)==digest,'A frozen inherited source changed: '+rel
 files.append({'relative':str(path.relative_to(root)),'sha256':digest})
 if rel.startswith('engine/') and path.suffix=='.py':engine_python.append((rel,digest))
for path in [root/'plan.json',root/'preparation_receipt.json',root/'capacity_receipt.json',root/'runtime_contract_receipt.json',root/'entry_path_guard_patch_receipt.json',Path(plan['initialization'])]:
 files.append({'relative':str(path.relative_to(root)),'sha256':sha(path)})
manifest={'files':files,'training_entry':'train_uavdt.py','controller':'run_uavdt_queue.py','source_model_files_unchanged':True}
with (root/'frozen_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
accepted={'state':'ready','account':CONFIG['account'],'root':str(root),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'plan_sha256':sha(root/'plan.json'),'frozen_manifest_sha256':sha(root/'frozen_manifest.json'),
          'initialization_sha256':plan['initialization_sha256'],'data_metadata_sha256':plan['data_metadata_sha256'],
          'annotations_sha256':plan['annotations_sha256'],'maximum_concurrent_trainings':selected,
          'training_timeout_seconds':timeout,'CPU_protocol_tests':10,'native_stage2_cases':4,'runtime_save_resume_contract_passed':True,
          'model_engine_python_fingerprint':hashlib.sha256(json.dumps(engine_python).encode()).hexdigest(),
          'GPU_capacity_cases_passed':[c['concurrency'] for c in capacity['cases'] if c['passed']],
          'per_arm_step_projection_train_hours':estimated_train/3600,'formal_training_started':False}
with (root/'final_acceptance.json').open('x') as f:json.dump(accepted,f,indent=2)
print(json.dumps(accepted),flush=True)
