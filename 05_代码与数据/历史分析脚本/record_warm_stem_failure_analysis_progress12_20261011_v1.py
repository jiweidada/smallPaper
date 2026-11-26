"""Advance current goal ledgers from verified packets, analyses and live process observations."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
inputs=read('outputs/warm_stem_failure_analysis_progress12_20261011_v1_inputs.json')
obs={}
for family,seeds in inputs['observations'].items():
 obs[family]={}
 for s,v in seeds.items():
  raw=v['payload'].encode();t=v['transport'];assert len(raw)==t['bytes'] and hashlib.sha256(raw).hexdigest()==t['sha256']
  obs[family][s]=json.loads(raw)
for s in ('1','2'):
 w=obs['warm'][s];u=obs['user'][s];d=obs['diagnostic'][s]
 assert w['source_SHA_verified'] and w['failure'] is None and w['seed']==int(s)
 assert u['code_SHA_verified'] and u['queue']['same_live'] and not u['failures']
 assert d['source_verified'] and d['queue']['same_live'] and d['failure'] is None and not d['actual_diagnostic_children']
 assert d['live']['phase']=='waiting_actual_predecessor_handles_and_GPU_idle'
 if s=='1':assert w['trainer']['same_live'] and w['completed_epochs']==11 and w['native_training_completion'] is None
 else:
  assert not w['trainer']['same_live'] and w['completed_epochs']==12
  assert w['native_training_completion']['identity_equals_receipt'] and w['controller_completion_status']=='COMPLETE_FIXED12'
 for a,j in u['jobs'].items():
  assert 0<=j['completed_epochs']<=12 and j['failure'] is None
  if j['trainer']:
   assert j['trainer']['same_live'] and j['controller']['same_live'] and j['controller']['identity_equals_receipt']
   assert j['full_preflight']['identity_matches_receipt'] and j['first100']['all100_rows_exact']
   assert j['runtime_start']['identity_equals_receipt'] and j['runtime_start']['optimizer_coverage']['all_parameters_exactly_once']
  else:assert j['completed_epochs']==0
collection=ROOT/'outputs/warm_stem_primary_collection12_20261011_v1'
v=read(collection/'seed2_warm_stem_v1/primary_verification_v1.json')
assert v['status']=='PASS_WARM_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS' and v['files_verified']==615 and v['actual_stage_images']==548
assert v['verification_script_sha256']==sha(ROOT/'英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py')
assert not v['saved_EMA_independent_replay_verified'] and not v['formal_terminal_reader_passed']
packet=Path(v['downloaded_packet']);manifest=read(packet/'primary_manifest.json')
assert sha(packet/'primary_manifest.json')==v['primary_manifest_sha256']
assert all((packet/n).stat().st_size==i['bytes'] and sha(packet/n)==i['sha256'] for n,i in manifest['files'].items())
analyses={}
for name,path in [('primary','outputs/warm_stem_primary_analysis12_20261011_v1/seed2'),
                  ('trajectory_and_AP_arithmetic','outputs/warm_stem_failure_review12_20261011_v1'),
                  ('clarified_AP_interpretation','outputs/warm_stem_failure_review12_20261011_v2')]:
 folder=ROOT/path
 for n,h in read(folder/'manifest.json')['files'].items():assert sha(folder/n)==h,n
 analyses[name]=dict(path='../'+path[len('outputs/'):]+('/report.json'),sha256=sha(folder/'report.json'))
clarify=read(ROOT/'outputs/warm_stem_failure_review12_20261011_v2/report.json')
assert clarify['status']=='WARM_SEED2_LOW_IOU_NATIVE_PR_LOSS_CLARIFIED_WITH_SAME_GT_SUPPORT_STRATA' and not clarify['native_gate_passed']
assert clarify['native_small_area_TP50_delta']==51 and clarify['native_small_area_AP50_delta']<0 and clarify['native_small_area_AP75_delta']>0
root=ROOT/'outputs/warm_stem_failure_analysis_progress12_20261011_v1';root.mkdir(exist_ok=False)
for f,seeds in inputs['observations'].items():
 for s,i in seeds.items():(root/f'{f}_seed{s}.json').write_bytes(i['payload'].encode())
write(root/'observation_transports.json',{f:{s:i['transport'] for s,i in seeds.items()} for f,seeds in inputs['observations'].items()})
counts={'warm_stem':{s:w['completed_epochs'] for s,w in obs['warm'].items()}}
jobs={}
for s,q in obs['user'].items():
 jobs[s]={}
 for a,j in q['jobs'].items():
  counts.setdefault(a,{})[s]=j['completed_epochs'];stats=j['latest_stats']
  jobs[s][a]=dict(j,state='training' if j['trainer'] else 'awaiting_capacity',
    AP_points=stats[0]*100 if stats else None,APs_points=stats[3]*100 if stats else None,
    compared_to_cached_BASE12=False,fixed12_efficacy_not_claimed=True)
report=dict(status='WARM_SEED2_PRIMARY615_FILES_VERIFIED_NATIVE_PR_FAILURE_LOCALIZED_ORIGINAL_USER_TRAINING_CONTINUES',
 local_saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 remote_observation_UTC={f:{s:o['UTC'] for s,o in seeds.items()} for f,seeds in obs.items()},
 previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
 actual_completed_epochs=counts,user_jobs=jobs,source26_and28_frozen_files_verified_both_servers=True,
 downloaded_Warm_primary_packets=1,seed=2,files_verified=615,ZIP_entries=616,stage_images=548,all_stage_arrays_reloaded_exact=True,
 saved_EMA_independent_replay_pending=True,analyses=analyses,
 warm_seed2_fixed12=dict(BASE12=clarify['BASE12'],warm12=clarify['warm12'],delta_points=clarify['delta_points'],native_gate_passed=False),
 low_vs_high_IoU_APs_contributions=clarify['exploratory_IoU_range_contributions_to_total_APs_points'],
 native_small_area_TP50_delta=51,native_small_area_AP50_delta=clarify['native_small_area_AP50_delta'],
 native_small_area_AP75_delta=clarify['native_small_area_AP75_delta'],
 same_GT_new_initial_support_at75=clarify['same_GT_initial_support_strata']['0.75']['warm_new_initial_support'],
 changed_next_action=clarify['changed_next_action'],
 small_area_native_matching_score_prefix_records_missing=True,high_IoU_geometry_loss_as_overall_APs_root_rejected=True,
 local_reader_initial_failure_and_fix='../warm_stem_primary_collection12_20261011_v1/local_reader_failure_v1.json',
 local_reader_fix_scope='Interpret remote Linux paths as PurePosixPath on Windows and re-hash complete existing extraction; training/experiment/code and results unchanged.',
 actual_diagnostic_waiters_verified_live_both_servers=True,diagnostic_model_children_started=0,
 root_proven=False,novelty_proven=False,new_BASE_forwards=0,new_BASE_training_epochs=0,
 new_candidate_detector_forwards_for_collection_and_analysis=0,new_evaluator_calls_for_collection_and_analysis=0,
 new_optimizer_updates_for_collection_and_analysis=0,new_training_epochs_for_collection_and_analysis=0,
 existing_process_signals_this_turn=0,configuration_training_retry=False)
write(root/'report.json',report)
text=['本轮实际工作与训练进度','',
 '完成种子2 Warm固定12轮产物下载、615份文件校验、548张阶段数组重载、逐目标边界和同查询传播分析。基线全部使用已有缓存。',
 'Warm种子2：AP %.4f / APs %.4f；基线 %.4f / %.4f；差值 %.4f / %.4f。'%(clarify['warm12']['AP'],clarify['warm12']['APs'],clarify['BASE12']['AP'],clarify['BASE12']['APs'],clarify['delta_points']['AP'],clarify['delta_points']['APs']),
 '原生小面积IoU0.50：TP增加51但AP下降0.5391。IoU0.75：TP减少2但AP提高0.0435。整体APs下降主要来自0.50–0.70段的PR曲线，不能从框覆盖数量直接认定失败原因。',
 'IoU0.75新增初始参考框覆盖1066个GT，其中702个本来就有基线最终框覆盖，因此“初始新增覆盖”不等于“新找到了1066个目标”。该阈值还丢掉719个原始初始覆盖，净增加347。',
 '总体APs最大负贡献是people，最大正贡献是van；下一步同时检查类别条件的精度—召回和最终真类别保留。','',
 '实际训练完成轮数：']
for a,seeds in counts.items():text.append(a+'：'+', '.join('种子%s %d/12'%(s,e) for s,e in seeds.items()))
text.extend(['','Warm种子1训练仍活跃，种子2已正完成12轮。SiLU/P2两台和注意力种子2均在训练；已启动项完整模型AMP/损失反传和前100输入一致性通过。另一台注意力等待容量。',
 '两个选点诊断队列的PID+启动时间已重新确认存活，仍等待已有训练与GPU空闲，没有启动诊断前向。独立EMA/正式终点证据仍待完成。',
 '本轮本地读取器遇到Windows把Linux远程路径改为反斜杠的问题，已保留失败记录并修正路径解析后通过。没有改远程冻结代码、重跑训练或扩大12轮上限。',
 '目标继续进行；不把单种子的描述或AP算术分解认作根因与创新成立。'])
(root/'本轮分析与实验进度.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
write(root/'manifest.json',dict(files={p.name:sha(p) for p in root.iterdir() if p.is_file()},source_sha256=sha(__file__),input_sha256=sha('outputs/warm_stem_failure_analysis_progress12_20261011_v1_inputs.json')))
evidence='../warm_stem_failure_analysis_progress12_20261011_v1/report.json'
def update(directory,version,changes):
 folder=ROOT/'outputs'/directory;old=folder/f'execution_state_v{version}.json';v=read(old)
 v.update(changes);v.update(supersedes=old.name,goal_complete=False);write(folder/f'execution_state_v{version+1}.json',v)
write(collection/'execution_state_v1.json',dict(status='SEED2_WARM_NATIVE_PRIMARY12_PACKET615_FILES_VERIFIED_AND_ANALYZED',
 evidence=evidence,completed_seeds=[2],seed1_pending=True,independent_saved_EMA_pending=True,formal_terminal_reader_pending=True,
 primary_verification='seed2_warm_stem_v1/primary_verification_v1.json',files_verified=615,stage_images=548,goal_complete=False))
update('warm_stem12_20261011_v1',15,dict(status='SEED2_PRIMARY_FIXED12_NEGATIVE_ANALYZED_SEED1_AT11_REPLAY_PENDING',evidence=evidence,
 actual_new_training_epochs=counts['warm_stem'],trainers={s:w['trainer'] for s,w in obs['warm'].items()},
 phases={s:w['live']['phase'] for s,w in obs['warm'].items()},last_remote_observation_UTC={s:w['UTC'] for s,w in obs['warm'].items()},
 native_training_completions={s:w['native_training_completion'] for s,w in obs['warm'].items()},
 seed2_primary_packet='../warm_stem_primary_collection12_20261011_v1/execution_state_v1.json',
 seed2_failure_review='../warm_stem_failure_review12_20261011_v2/report.json',seed2_analysis=analyses,
 pending='Finish seed1 fixed12, await owned independent EMA closeouts, preserve negative endpoint; no extension/retry.'))
update('user_backbone12_20261011_v1',11,dict(status='ORIGINAL_USER_SILU_P2_TRAIN_BOTH_SERVERS_ATTENTION_SEED2_TRAINING',evidence=evidence,jobs=jobs,
 completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
 last_remote_observation_UTC={s:q['UTC'] for s,q in obs['user'].items()},queue_processes={s:q['queue'] for s,q in obs['user'].items()},
 queue_phases={s:q['live']['phase'] for s,q in obs['user'].items()},GPU_resource={s:q['live'].get('resource') for s,q in obs['user'].items()},
 full_GPU_preflight_passed_by_seed={s:{a:bool(j['full_preflight']) for a,j in q['jobs'].items()} for s,q in obs['user'].items()},
 remaining_GPU_preflights_and_trainings_pending={'1':['backbone_attention'],'2':[]},fixed12_efficacy_not_yet_claimed=True))
update('native_selection_diagnostic12_20261011_v2',4,dict(evidence=evidence,last_actual_process_observation_UTC={s:d['UTC'] for s,d in obs['diagnostic'].items()},
 actual_live_waiters={s:d['queue'] for s,d in obs['diagnostic'].items()},actual_diagnostic_children_started=0,
 phase='waiting_actual_predecessor_handles_and_GPU_idle',source6_verified_both_servers=True))
update('stem_component12_20261011_v1',34,dict(latest_compact_status=evidence,current_goal_turn='progress',no_progress_streak=0,
 warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v16.json',standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v12.json',
 standalone_user_completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
 standalone_user_queue_phases={s:q['live']['phase'] for s,q in obs['user'].items()},
 actual_native_selection_diagnostic_state='../native_selection_diagnostic12_20261011_v2/execution_state_v5.json',
 remote_result_observers_verified_live_both_seeds=True,remote_result_observer_liveness_scope='Exact native selection waiter PID/start ticks re-observed live now on both hosts; no new assertion on separate collection waiters.',
 warm_seed2_primary_failure_analysis='../warm_stem_failure_review12_20261011_v2/report.json',
 warm_seed2_primary_collection='../warm_stem_primary_collection12_20261011_v1/execution_state_v1.json',
 small_area_raw_native_score_prefix_causal_explanation_not_available=True,
 high_IoU_geometry_counts_alone_as_failure_root_rejected=True,
 existing_process_signals_this_turn=0,old_training_process_signals_this_turn=0,
 interpretation='Warm seed2 fixed12 primary packet and saved arrays verified. Small early coverage improves, but APs decreases through low-IoU native PR metrics despite more TP; high-IoU AP segment is positive. Class-conditioned PR and selection evidence now take priority over a uniform geometry-loss explanation. Seed1 Warm at11; original SiLU/P2 at4/7, attention seed2 at2. Independent EMA and initial-location diagnostics still pending.'))
update('original_stem12_20261011_v1',37,dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v35.json'))
print(json.dumps(dict(status=report['status'],counts=counts,verified_packet_files=615,goal_complete=False),ensure_ascii=False))
