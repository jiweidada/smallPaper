"""Verify completed capacity locally, then freeze the 3x12 formal protocol."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import json
import os
import shutil
import zipfile
from preservation_boundary_A58_v1 import P,B,sha,save_new,begin,finish

R54=B/'A54_场景尺度约束候选_20261101_v1'
def load(p):return json.loads(Path(p).read_text('utf-8'))
def replace(p,x):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,p)

rpc=load(R54/'A58_capacity_poll_export_v2.json');assert rpc['exit']==0
remote=json.loads(rpc['stdout'])
assert remote['supervisor']['status']=='CAPACITY_COMPLETED' and not remote['GPU'].strip()
download=load(R54/'A58_capacity_download_v1.json');assert download['status']=='TRANSFERRED'
z=R54/'A57_downloads/A58_capacity_terminal_v1.zip'
assert sha(z)==remote['export']['sha256'] and z.stat().st_size==remote['export']['bytes']
assert not (P/'capacity_terminal_v1.zip').exists();shutil.copyfile(z,P/'capacity_terminal_v1.zip')
d=P/'capacity_remote_v1';d.mkdir(exist_ok=False)
with zipfile.ZipFile(z) as f:
    for n in f.namelist():
        target=(d/n).resolve();assert target.is_relative_to(d.resolve())
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.read(n))
result=load(d/'capacity_result.json')
assert result==remote['result'] and result['status']=='PASS'
assert load(d/'capacity_supervisor_exit.json')['exit']==0
assert result['real_detector_forwards']==result['discarded_optimizer_updates']==6
assert result['protocol_sha256']==sha(P/'capacity_protocol_FROZEN_v1.json')
for arm in ['B','M','Q']:
    row=result['arms'][arm]
    assert row['status']=='PASS' and row['counts']==dict(full_model_forwards_started=2,full_model_forwards_completed=2,optimizer_updates=2,native_counter=2)
    events=[json.loads(s) for s in (d/('capacity_'+arm)/'actual_events.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['forward_start','forward_complete','optimizer_returned']*2
    assert all(s['input_sha256']==b['input_sha256'] and s['pre_forward_rng_sha256']==b['pre_forward_rng_sha256'] for s,b in zip(row['steps'],result['arms']['B']['steps']))
    if arm!='B':
        grads=row['steps'][1]['context_gradient_max']
        assert grads['gate']>0 and any(v is not None and v>0 for k,v in grads.items() if k!='gate')
cap=load(P/'capacity_protocol_FROZEN_v1.json')
for n,h in cap['files_sha256'].items():assert sha(P/n)==sha(d/n)==h,n
names=['runtime_A58.py','train_A58.py','supervise_A58.py','regional_range_attention_v1.py',
       'training_utils_A58.py','evaluate_A58.py','asset_expectations.json','capacity_A58.py','supervise_capacity_A58.py']
for n in names:
    assert sha(P/n)==sha(d/n),n
    if n.endswith('.py'):ast.parse((P/n).read_text('utf-8'))
save_new(P/'capacity_local_verification_v1.json',dict(status='PASS_BYTES_EVENTS_INPUT_RNG_AND_GRADIENTS',
    actual_detector_forwards=6,discarded_optimizer_updates=6,AP_evaluations=0,
    arms={a:dict(peak_allocated=max(s['allocated_peak'] for s in result['arms'][a]['steps']),
                 peak_reserved=max(s['reserved_peak'] for s in result['arms'][a]['steps'])) for a in ['B','M','Q']},
    source_files_verified=len(names),new_audit_detector_forwards=0))
plan=dict(phase='A58',protocol_type='FRESH_MATCHED_FIXED12_PILOT',arms=['B','M','Q'],epochs_per_arm=12,
    seed=20261101,training_batch=8,gradient_accumulation=1,train_images=6471,val_images=548,
    train_boxes=343204,val_boxes_before_ignore=38759,scored_val_boxes=38602,
    image_transform='Native deterministic 640x640 bilinear stretch, deterministic epoch/file horizontal flip, ImageNet normalization',
    training_batches_per_epoch=809,maximum_optimizer_updates_per_arm=9708,
    evaluation_epochs=[4,8,12],evaluation_batch=4,evaluation_forwards_per_arm=411,
    latency_forwards_per_arm=40,maximum_detector_forwards_per_arm=10159,
    maximum_detector_forwards_all_arms=30477,maximum_optimizer_updates_all_arms=29124,
    fixed_endpoint='epoch12 EMA only; no best epoch or best checkpoint selection',
    baseline_learning_gate=dict(AP=8,APs=3,on_failure='Cancel unstarted M/Q; routing hypothesis remains untested'),
    candidate_gates=dict(Q_minus_B_APs_min_pp=.5,Q_minus_M_APs_min_pp=.3,
                         Q_minus_B_AP_min_pp=-.2,Q_over_M_median_model_latency_max=1.2),
    gate_meaning='Practical continuation gates for this short single-seed pilot; not statistical significance, convergence or novelty proof',
    optimizer='Same native AdamW: backbone2.5e-5/rest5e-4; decay0 for norm/bn/bias else1e-4; betas(.9,.999)',
    schedule='500-update linear warmup then cosine to0.1 over9708 updates; grad norm clip0.1; EMA.9999 ramp1000',
    precision=cap['precision'],backend=cap['backend'],
    operator=dict(site='Native fused P3 output only; all query positions updated',channels=192,heads=3,
                  spatial_region_side=4,topk_key_regions_per_query_region_per_head=4,residual_gate_initial=0,
                  extra_parameters=148609,registered_parameters_all_arms=9951721,
                  B='Registered context disabled; original ECDet-S path',
                  M='Mean query against per-channel K interval, Quest-style point-to-box score',
                  Q='Q and K per-channel intervals, max of four endpoint products summed over channels',
                  shared='Same QKV/projection/LayerNorm/residual gate/fine-token attention, fixed budget; no new loss or query labels'),
    initialization='Strict common augmented state, fresh for every arm; capacity weights discarded',
    augmented_initialization_sha256=cap['augmented_initialization_sha256'],
    capacity_protocol_sha256=sha(P/'capacity_protocol_FROZEN_v1.json'),
    capacity_result_sha256=sha(d/'capacity_result.json'),
    files_sha256={n:sha(P/n) for n in names},
    matching_checks=['Every epoch image order','First8 actual image/target tensor hashes','First8 pre-forward CPU/CUDA RNG hashes','All-batch target flow hashes'],
    evaluator='Frozen A49 COCO-style10class IoU.5:.95, maxDets300, clipped original pixel area; class0 rectangle union IoA>=.5 removes GT/predictions; not official VisDrone leaderboard score',
    latency='Same3090, batch1 native eager BF16 model only, 10warm+30timed calls; excludes decode/postprocess and is not onboard latency',
    maximum_seconds_per_arm=21600,maximum_seconds_total=64800,
    stopping=['Nonfinite or OOM stops queue','Baseline gate failure cancels M/Q','Any source/input/RNG mismatch stops queue','No restart, extra seed, extra epoch, coefficient/block/topk tuning or kernel rescue'],
    source_review_scope='Selected primary sections and four AdaCluster files only. No direct equality identified within reading; novelty not cleared.',
    hypothesis_not_fact='Minority target query relations may be hidden by regional query means. Q only bounds maximum unnormalized dot product, not attention mass or AP.',
    global_allocation=dict(previous_closed_model_calls=246151,capacity_model_calls_completed=6,
        new_formal_calls_allocated=30477,total_model_call_cap=276634,
        previous_optimizer_updates_including_capacity=204470,capacity_updates_completed=6,
        new_formal_updates_allocated=29124,maximum_all_optimizer_updates=233600))
save_new(P/'protocol_FROZEN_v1.json',plan)
stage=R54/'A58_uploads/protocol_FROZEN_v1.json'
assert not stage.exists();shutil.copyfile(P/'protocol_FROZEN_v1.json',stage)
begin('A58_formal_allocation_v1','A58_capacity_allocation_preservation_receipt_v1.json')
now=datetime.now(timezone.utc).isoformat()
current=load(P/'CURRENT_A58.json')
current.update(status='A58_CAPACITY_PASS_FORMAL_BMQ_ALLOCATED_NOT_STARTED',observed_utc=now,
    selected_for_formal_training=True,new_training_queued=True,new_training_queue=['B12','M12','Q12'],
    new_detector_forwards=6,new_optimizer_updates=6,capacity_calls_observed=6,capacity_updates_observed=6,
    capacity_passed=True,formal_calls_allocated=30477,formal_calls_observed=0,formal_updates_observed=0,
    formal_protocol=str(P/'protocol_FROZEN_v1.json'),formal_protocol_sha256=sha(P/'protocol_FROZEN_v1.json'),
    next='Launch frozen sequential freshB/M/Q. FixedB12 learning gate then fixedQ12 vsB/M gates; no automatic rescue.',
    limits='Only technical feasibility passed. Candidate efficacy, minority-target mechanism and novelty remain unproven.')
replace(P/'CURRENT_A58.json',current)
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    x=load(B/name)
    x.update(status=current['status'],A58_execution=current,latest_update_utc=now,candidate_selected_for_pilot=True,
             mainline_selected_for_pilot=False,mainline_validated=False,
             pilot_validation_status='Capacity6 passed; fresh3x12 formal protocol frozen and allocated, not launched yet')
    if name.startswith('01_'):
        row=next(r for r in x['candidates'] if r['id']==current['candidate'])
        row.update(status=current['status'],selected_for_training=True,training_allocation=30477,
                   evidence_scope='Capacity6 forwards/6 discarded updates passed; no AP yet',formal_protocol=current['formal_protocol'])
    else:
        assert x['model_forwards']==246151 and x['cap']==246157 and x['training_updates']==204470
        x.update(model_forwards=246157,cap=276634,training_updates=204476,
                 current_phase='A58_FIXED12_FORMAL_ALLOCATED',current_native_job_terminal=True,
                 current_goal_turn='progress: realbatch8 capacity B/M/Q all passed, matched3x12 formal recipe frozen and allocated',
                 full_forward_cap_closed=False,current_user_task_status='Goal active; formal candidate pilot next, mainline unvalidated')
        x.setdefault('authoritative_phase_history',[]).append(dict(phase='A58',utc=now,classification='progress',
                 actual_capacity_model_calls=6,actual_capacity_updates=6,formal_model_calls_allocated=30477,formal_observed=0))
    replace(B/name,x)
finish('A58_formal_allocation_preservation_receipt_v1.json','CAPACITY_PASS_FORMAL3X12_ALLOCATED',
       actual_capacity_model_calls=6,actual_capacity_updates=6,formal_model_calls_allocated=30477)
print(json.dumps(dict(status=current['status'],formal_protocol_sha256=sha(P/'protocol_FROZEN_v1.json'),
    global_calls_completed=246157,global_call_cap=276634,capacity_status='PASS'),ensure_ascii=False))
