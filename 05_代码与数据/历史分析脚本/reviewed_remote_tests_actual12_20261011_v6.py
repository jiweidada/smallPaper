"""Record actual ongoing training, a new reviewed first-attempt diagnostic, and stage evidence."""
from pathlib import Path
import datetime,hashlib,json
R=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
evidence={}
def record(p):
    p=Path(p);evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return read(p)
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def capture(p):
    j=record(p);result={}
    for k,v in j.items():
        raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256'];result[k]=json.loads(raw)
    return result
prior=R/'outputs/reviewed_remote_tests_actual12_20261011_v5';old=record(prior/'report.json');m=record(prior/'manifest.json')
assert sha(prior/'report.json')==m['report_sha256']
for n,v in m['files'].items():assert sha(R/n)==v['sha256'],n
c=capture(R/'outputs/reviewed_remote_tests_actual12_20261011_v6_inputs.json')
earlier=capture(R/'outputs/turn6_reviewed_remote_actual_inputs.json')
training=[]
for seed in (1,2):
    j=c[f'turn6_final_rank16_native_same_epoch_seed{seed}'];previous=old['new_lowrank_actual_training'][seed-1]
    assert j['candidate_same_live'] and j['queue_same_live'] and j['controller_same_live']
    assert j['failure'] is None and j['queue_failure'] is None and j['first100']['all100_rows_exact']
    assert j['completed_epochs'] in (3,4) and j['completed_epochs']>previous['completed_epochs']
    assert j['audit']['actual_optimizer_steps']>previous['audit']['actual_optimizer_steps']
    assert j['candidate_handle']['automatic_retry'] is False
    for x in j['same_epoch_metrics']:assert x['decision_scope']=='descriptive_only_fixed12_required'
    training.append(dict(previous,observed_remote_UTC=j['UTC'],completed_epochs=j['completed_epochs'],audit=j['audit'],
        actual_handles=dict(queue=j['queue_handle'],supervisor=j['controller_handle'],trainer=j['candidate_handle']),
        exact_owner_handles_live=True,source_log_paths=j['source_log_paths'],source_log_sha256=j['source_log_sha256'],same_epoch_metrics=j['same_epoch_metrics']))
source=record(R/'outputs/p2_lowrank12_20261011_v3/source_review.json')
assert {p.name:sha(p) for p in (R/'英文小论文/code/p2_lowrank12_20261011_v3').iterdir()}==source['files']
direction=R/'outputs/p2_direction_diagnostic12_20261011_v2';review=record(direction/'source_review.json')
code=R/'英文小论文/code/p2_direction_diagnostic12_20261011_v2';assert {p.name:sha(p) for p in code.iterdir()}==review['files']
engineering=c['turn6_direction_v2_actual_remote_engineering_seed1'];dispatch=c['turn6_direction_v2_actual_queue_dispatch_seed1'];observed=c['turn6_direction_v2_exact_live_observation_seed1']
assert engineering['source_files']==review['files'] and engineering['failed_replay_status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
assert engineering['model_forwards']==engineering['BASE_forwards']==engineering['parent_forwards']==engineering['training_epochs']==0
assert engineering['old_GPU_controls_unattempted'] and len(engineering['negative_cases'])==6 and not engineering['torch_imported']
assert observed['handle']==dispatch and observed['same_live'] and observed['failure'] is None and not observed['GPU_control_started']
assert dispatch['source_sha256']==sha(code/'wait.py') and dispatch['protocol_sha256']==sha(code/'protocol.json')
assert observed['actual_start']['pid']==dispatch['pid'] and observed['actual_start']['start_ticks']==dispatch['start_ticks']
assert observed['phase']['phase']=='waiting_exact_rank16_and_first_SiLU_attention_closeouts_GPU_idle'
protocol=record(code/'protocol.json')
assert protocol['primary_contrast']=='svd16_norm_minus_dc16_norm' and protocol['reference_replay_status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
for key in ('failed_replay_verification_sha256','CPU_report_sha256','CPU_weights_sha256'):assert protocol[key]==engineering[key]
for p in code.iterdir():evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
stagefolder=R/'outputs/Stem_stage_hypotheses12_20261011_v1';stages=record(stagefolder/'report.json');sm=record(stagefolder/'manifest.json')
assert sha(stagefolder/'report.json')==sm['report_sha256'] and sha(stagefolder/'问题解释检验.txt')==sm['text_sha256']
for n,v in sm['files'].items():assert sha(R/n)==v['sha256'],n
assert stages['new_model_forwards']==stages['new_evaluator_calls']==stages['new_training_epochs']==0
out=R/'outputs/reviewed_remote_tests_actual12_20261011_v6';out.mkdir(exist_ok=False)
r=dict(old)
r.update(status='ACTUAL_RANK16_P2_EPOCH3_4_AND_TWO_UNATTEMPTED_DIRECTION_CONTROLS_REVIEWED_QUEUED',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_immutable_report_sha256=sha(prior/'report.json'),
    new_lowrank_actual_training=training,two_seed_Stem_literal_stage_explanations=stages,
    A_previously_unattempted_P2_direction_controls=dict(source_review=review,actual_remote_engineering=engineering,actual_dispatch=dispatch,exact_live_observation=observed,
        native_GPU_results_available=False,new_GPU_namespace=True,old_failed_parent_replay_preserved=True,old_failed_parent_replay_retried=False,
        primary_contrast='svd16_norm_minus_dc16_norm',parent_reference_deltas_descriptive_only=True,formal_new12_training_efficacy=False),
    A_remaining_SiLU_attention_unattempted_formal_closeout_queue=dict(old['A_remaining_SiLU_attention_unattempted_formal_closeout_queue'],
        latest_actual_live_observation=c['turn6_final_rank16_native_same_epoch_seed1']['remaining_formal_queue']),
    current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    pending=['Finish both actually running new rank16 configurations through fixed12 EMA and first independent saved-weight reload; preserve all logs/weights/stages, no extension or failed-training retry.',
        'Download and verify both new fixed12 packets; report same-seed BASE12 and originalP2 changes. Apply both-seed APs strict improvement and AP non-decrease without best-epoch substitution.',
        'A first SiLU/attention formal-closeout queue waits exact rank16 owners; each previously unattempted closeout runs once after GPU becomes idle.',
        'A new direction-control queue waits exact rank16 owners and the SiLU/attention queue plus recorded future finishers. Then DC16 and SVD16 are evaluated once. No original/reloaded parent or BASE forward, no new training.',
        'Compare SVD16 versus DC16 as the primary conditional direction diagnostic. Keep A failed parent replay pinned; descriptive parent differences cannot serve as formal exact-parent efficacy.',
        'Use the verified Stem stage cohorts and the future actual direction outputs to choose the next falsifiable mechanism if new fixed12 gates fail; do not attribute all gains to selection, decoder weights, frequency, or removed redundancy.'])
r['current_scientific_conclusions']=[x for x in old['current_scientific_conclusions'] if not x.startswith('Rank16 actual epoch2')]
r['current_scientific_conclusions'].append('Actual latest rank16: seed1 epoch3 AP14.608432/APs8.279524; seed2 epoch4 AP16.833413/APs9.669493. Both latest same-epoch AP/APs exceed BASE; neither is a fixed12 result and the seeds are at different epochs.')
r['current_scientific_conclusions'].append('Highpass-versus-RGB initial50 support gains include shared selected sites in 36.056%/60.888%. Among gained native all-area small-GT TP75, 826/937 and1053/1239 lacked initial75 support and reached75 later. This excludes literal selection-only or all-initial75 descriptions, not unique parameter causality or APs attribution.')
write(out/'report.json',r)
lines=['当前真实远程进度与新诊断：','']
for x in training:
    row=x['same_epoch_metrics'][-1];a,b,p=row['candidate'],row['BASE'],row['original_P2']
    lines.append(f'种子{x["seed"]}，完成第{x["completed_epochs"]}轮：AP{a["AP"]:.6f} / APs{a["APs"]:.6f}；同轮BASE AP{b["AP"]:.6f} / APs{b["APs"]:.6f}；原P2 AP{p["AP"]:.6f} / APs{p["APs"]:.6f}。')
    lines.append(f'同轮BASE差值：AP{a["AP"]-b["AP"]:+.6f} / APs{a["APs"]-b["APs"]:+.6f}；实际优化更新{x["audit"]["actual_optimizer_steps"]}次，进程PID/启动标识核实存活。')
lines+=['两个种子当前轮次不同，不能直接互比；最终仍使用各自固定第12轮EMA与同种子BASE12比较。',
    '没有新增基线训练/前向；新配置最多12轮，不延长、不用最好轮替代终点。','',
    '阶段诊断：高频输入相比同参数RGB，初始50支持新获益中共同入选位置占36.1%/60.9%，说明变化不只发生在选中位置。',
    '新增最终75命中中88.2%/85.0%没有初始75支持，必须在后续几何细化中达到阈值；但这不隔离decoder参数因果贡献。',
    '这些命中人口不是APs贡献；类别分解和实际小面积PR口径继续保留。','',
    'A此前未进入GPU的两个P2方向对照已完成新协议和真实远程权重检查，并部署单次等待队列。',
    '主比较：SVD16减DC16，两者秩16、权重范数相同，其余原始权重严格相等。只作训练后模型的方向敏感性诊断。',
    '原P2严格重放失败继续保留；不重试失败重放，不增加父模型/基线前向，不重新训练。',
    '等待当前训练和SiLU/注意力首次核验全部结束且GPU空闲后执行两个对照；尚无新的GPU对照结果。',
    '算子逼近误差、特征输出能量未匹配；不能从SVD比DC好推出低秩去冗余是唯一原因，也不能代替rank16新训练的第12轮门槛。']
(out/'真实进度与问题检验.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(out/'manifest.json',dict(files=evidence,report_sha256=sha(out/'report.json'),text_sha256=sha(out/'真实进度与问题检验.txt'),producer_sha256=sha(__file__)))
state=dict(status=r['status'],canonical_report='../reviewed_remote_tests_actual12_20261011_v6/report.json',canonical_report_sha256=sha(out/'report.json'),
    current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,new_lowrank_actual_training=training,
    two_seed_Stem_stage_hypotheses_checked=True,A_direction_controls_source_and_actual_CPU_reviewed=True,A_direction_controls_exact_queue_live=True,
    A_original_P2_exact_replay_failure_preserved=True,A_original_P2_failed_replay_retried=False,
    new_BASE_forwards=0,new_BASE_training_epochs=0,new_parent_P2_forwards=0,failed_training_retried=False,
    two_seed_formal_efficacy_proven=False,root_or_novelty_proven=False,pending=r['pending'])
for family,num,previous in [('stem_component12_20261011_v1',47,46),('original_stem12_20261011_v1',50,49),
    ('user_backbone12_20261011_v1',24,23),('user_backbone_primary_collection12_20261011_v2',9,8),
    ('p2_subspace_diagnostic12_20261011_v1',7,6),('p2_lowrank12_20261011_v3',3,2),
    ('native_selection_analysis12_20261011_v5',2,1),('native_selection_common_location12_20261011_v3',2,1),('six_Stem_category_APs12_20261011_v1',2,1)]:
    write(R/f'outputs/{family}/execution_state_v{num}.json',dict(state,supersedes=f'execution_state_v{previous}.json'))
for family in ('Stem_stage_hypotheses12_20261011_v1','p2_direction_diagnostic12_20261011_v2'):
    write(R/f'outputs/{family}/execution_state_v1.json',state)
print(json.dumps(dict(status=r['status'],epochs=[x['completed_epochs'] for x in training],A_direction_queue_PID=dispatch['pid'],direction_GPU_started=False,new_BASE_forwards=0,goal_complete=False),ensure_ascii=False))
