"""Freeze six capacity calls only; preserve prior state before updating it."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import shutil
import zipfile

P=Path(__file__).resolve().parent
B=P.parent
R54=B/'A54_场景尺度约束候选_20261101_v1'

def load(p): return json.loads(Path(p).read_text('utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def replace(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,p)

r=load(R54/'A58_CPU_constructor_poll_v6.json')
assert r['exit']==0
x=json.loads(r['stdout'])
cpu=x['CPU_constructor_result_v1.json']
assert cpu['status']=='PASS_CPU_CONSTRUCTOR_NO_FORWARD'
assert cpu['counts']==dict(constructors_attempted=1,constructors_completed=1,whole_detector_forwards=0,
                          backwards=0,optimizer_steps=0,AP_evaluations=0)
assert x['CPU_constructor_exit_v1.json']['exit']==0 and len(x['process'].splitlines())==1
for n,h in cpu['bindings'].items():assert sha(P/n)==h,n
save(P/'CPU_constructor_observed_result_v1.json',cpu)
for n in ['runtime_A58.py','capacity_A58.py','supervise_capacity_A58.py','train_A58.py','supervise_A58.py']:
    ast.parse((P/n).read_text('utf-8'))
names=['runtime_A58.py','capacity_A58.py','supervise_capacity_A58.py','regional_range_attention_v1.py',
       'training_utils_A58.py','asset_expectations.json']
plan=dict(phase='A58',protocol_type='REAL_BATCH8_CAPACITY_ONLY',arms=['B','M','Q'],
    maximum_steps_per_arm=2,maximum_real_detector_forwards=6,maximum_discarded_optimizer_updates=6,
    maximum_model_constructors=3,maximum_AP_evaluations=0,weights_discarded=True,
    formal_training_authorized_by_this_file=False,formal_training_queue=[],
    actual_batch=8,image_size=[640,640],seed=20261101,epoch=0,epoch_step=809,
    data='VisDrone2019-DET native valid10class train6471/343204 boxes',
    precision='BF16 autocast for detector; criterion FP32; routing and fine attention FP32',
    backend=dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True),
    augmented_initialization_sha256=cpu['common_init_augmented']['sha256'],
    maximum_seconds_per_arm=600,no_retry=True,no_batch_reduction=True,
    gates=['No OOM, finite native outputs/loss/gradients','B/M/Q same two inputs and pre-forward RNG',
           'B context inactive; M/Q nonzero context gradients by second step'],
    files_sha256={n:sha(P/n) for n in names},
    global_allocation=dict(previous_closed_calls=246151,new_capacity_calls=6,total_call_cap=246157,
                          previous_updates_including_capacity=204470,new_discarded_updates=6,
                          formal_training_calls_allocated=0))
save(P/'capacity_protocol_FROZEN_v1.json',plan)
payloadnames=names+['capacity_protocol_FROZEN_v1.json','evaluate_A58.py','train_A58.py','supervise_A58.py']
zip_path=P/'pilot_payload_v1.zip'
with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED) as z:
    for n in payloadnames:z.write(P/n,n)
target=R54/'A58_uploads/pilot_payload_v1.zip'
assert not target.exists();shutil.copyfile(zip_path,target)
save(P/'capacity_payload_preparation_v1.json',dict(payload_bytes=zip_path.stat().st_size,
    payload_sha256=sha(zip_path),members=payloadnames,capacity_protocol_sha256=sha(P/'capacity_protocol_FROZEN_v1.json')))

priorpath=P/'A58_operator_preservation_receipt_v1.json'
prior=load(priorpath)
mutable=[B/'01_候选判别表.json',B/'15_当前状态与结果校验.json',P/'CURRENT_A58.json']
snap=P/'before_A58_capacity_allocation_v1';snap.mkdir(exist_ok=False)
mapping={}
for f in mutable:
    assert sha(f)==prior['files'][str(f)]['sha256']
    target=snap/f.name;shutil.copyfile(f,target)
    mapping[str(f)]=dict(snapshot=str(target),sha256=sha(target))
save(P/'prior_A58_operator_resolution_v1.json',dict(prior_receipt=str(priorpath),prior_receipt_sha256=sha(priorpath),files=mapping))
for n,row in prior['files'].items():
    f=Path(mapping[n]['snapshot']) if n in mapping else Path(n)
    assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],n
save(P/'prior_A58_operator_resolved_reverification_v1.json',dict(status='PASS',files=len(prior['files'])))
now=datetime.now(timezone.utc).isoformat()
current=load(P/'CURRENT_A58.json')
current.update(status='A58_CPU_CONSTRUCTOR_PASS_CAPACITY6_ALLOCATED_NOT_STARTED',observed_utc=now,
    closest_review='Four selected AdaCluster files; direct equality not identified, active inner kernel not fully read; novelty not cleared',
    closest_review_report=str(P/'04_AdaCluster有界源码核查.txt'),
    CPU_real_detector_constructors=1,CPU_constructor_verified=True,no_new_real_detector_instance_yet=False,
    augmented_parameters=9951721,augmented_initialization_sha256=cpu['common_init_augmented']['sha256'],
    capacity_calls_allocated=6,capacity_calls_observed=0,capacity_updates_observed=0,
    capacity_protocol=str(P/'capacity_protocol_FROZEN_v1.json'),formal_calls_allocated=0,
    next='Execute once B/M/Q two real batch8 steps each, discarded. If capacity passes, freeze fresh12x3 formal pilot before launch.',
    limits='Synthetic checks and one CPU constructor passed. No real detector forward, AP, candidate benefit or novelty clearance yet.')
replace(P/'CURRENT_A58.json',current)
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    d=load(B/name)
    d.update(status=current['status'],A58_execution=current,latest_update_utc=now,
             pilot_validation_status='CPU constructor passed; only6 capacity calls allocated, formal training not allocated',
             current_host_candidate='ECDet-S matched initialization, 9951721 registered parameters; capacity pending')
    if name.startswith('01_'):
        row=next(r for r in d['candidates'] if r['id']==current['candidate'])
        row.update(status=current['status'],capacity_allocation=6,selected_for_training=False,
                   evidence_scope='Synthetic CPU/CUDA checks plus one real CPU constructor; 0 whole-detector forwards')
    else:
        assert d['model_forwards']==d['cap']==246151 and d['training_updates']==204470
        d.update(cap=246157,full_forward_cap_closed=False,counts_terminal=True,
                 latest_authoritative_phase='A58',current_phase='A58_CAPACITY6_ALLOCATED',
                 current_native_job_terminal=True,training_counts_terminal=True,
                 current_goal_turn='progress: bounded selected-source overlap review and single CPU constructor passed; six capacity calls allocated, no formal training',
                 current_user_task_status='Goal active; mainline not validated; capacity only next')
        d.setdefault('authoritative_phase_history',[]).append(dict(phase='A58',utc=now,classification='progress',
                 capacity_calls_allocated=6,formal_calls_allocated=0,new_detector_forwards=0))
    replace(B/name,d)
files=[f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts]+mutable[:2]
save(P/'A58_capacity_allocation_preservation_receipt_v1.json',dict(status='CAPACITY6_ONLY_ALLOCATED_PRESERVED',
    goal_status='active',mainline_validated=False,files={str(f):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(set(files))},
    new_real_detector_forwards=0,new_optimizer_updates=0,capacity_calls_allocated=6,formal_calls_allocated=0))
print(json.dumps(dict(status=current['status'],payload_sha256=sha(zip_path),payload_bytes=zip_path.stat().st_size,
    initialization_sha256=cpu['common_init_augmented']['sha256'],global_call_cap=246157),ensure_ascii=False))
