"""Persist verified remote records and advance immutable progress ledgers."""
from pathlib import Path
import base64,datetime,hashlib,json,sys
ROOT=Path.cwd()
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
inputs=read('outputs/p2_phase_mean_remote_review_progress12_20261011_v1_inputs.json')
def decode(item):
 raw=item['payload'].encode('utf8');t=item['transport']
 assert len(raw)==t['bytes'] and hashlib.sha256(raw).hexdigest()==t['sha256']
 return json.loads(raw)
obs={family:{s:decode(i) for s,i in seeds.items()} for family,seeds in inputs['observations'].items()}
integrity=decode(inputs['warm_seed2_primary_integrity'])
assert integrity['status']=='PASS_WARM_SEED2_EXACT12_PRIMARY_INTEGRITY_AND_NATIVE_AP_ARITHMETIC_ONLY'
assert integrity['completed_epochs']==12 and integrity['stage_images_each_SHA_verified']==548
assert not integrity['actual_trainer']['same_live'] and not integrity['actual_controller']['same_live']
assert not integrity['native_gate_passed'] and not integrity['independent_saved_EMA_replay_verified']
assert integrity['new_model_forwards']==integrity['new_BASE_forwards']==integrity['new_training_epochs']==integrity['new_evaluator_calls']==0
prototype_code=ROOT/'英文小论文/code/p2_phase_mean_prototype_20261011_v1'
protocol=read(prototype_code/'protocol.json')
source_map={**protocol['prototype_sources'],'protocol.json':sha(prototype_code/'protocol.json')}
assert len(protocol['frozen_user_files'])==28 and not protocol['training_allowed']
for name,digest in source_map.items():assert sha(prototype_code/name)==digest
reports={}
for s,o in obs['prototype'].items():
 assert int(s)==o['seed'] and not o['worker']['same_live'] and not o['failure_exists'] and o['source_files']==source_map
 actual={}
 for n,item in o['records'].items():
  raw=base64.b64decode(item['base64'],validate=True)
  assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
  actual[n]=json.loads(raw)
 r,c=actual['report.json'],actual['completion.json']
 assert r['seed']==int(s) and r['status']==c['status']=='PASS_P2_PHASE_MEAN_CPU_OPERATOR_AND_BRANCH_ENGINEERING_ONLY'
 assert c['exit_code']==0 and c['report_sha256']==o['records']['report.json']['sha256']
 assert c['receipt']['pid']==o['worker']['pid'] and c['receipt']['start_ticks']==o['worker']['start_ticks']
 assert c['receipt']['frozen_user_files_verified']==28 and not c['receipt']['automatic_retry']
 assert r['protocol_sha256']==source_map['protocol.json'] and r['source_sha256']==source_map['cpu_validate.py']
 assert r['frozen_user_source_SHA']==protocol['frozen_user_files']['user_p2.py'] and r['frozen_user_files_before_after_exact']
 assert len(r['cases'])==6 and r['CPU_branch_forwards']==8
 assert r['candidate_detector_forwards']==r['BASE_detector_forwards']==r['training_epochs']==r['optimizer_updates']==r['new_accuracy_experiments']==0
 branch=[c for c in r['cases'] if c['name']=='actual_zero_initialized_branch_shape_and_gradient']
 assert len(branch)==4 and all(c['parameters_each']==17440 and c['identical7_initial_states'] and c['initial_output_exact_P3'] and c['restore_grad_nonzero'] and c['reduce_initial_grad_zero'] and c['mean_restore_phase_gradient_max_abs_difference']==0 for c in branch)
 reports[s]=r
for s,w in obs['warm'].items():
 assert w['seed']==int(s) and w['source_SHA_verified'] and w['failure'] is None and 0<=w['completed_epochs']<=12
 assert w['detector_preflight']['identity_equals_current_receipt']
 if w['completed_epochs']==12:
  assert not w['trainer']['same_live'] and w['native_training_completion']['status']=='NATIVE_FIT_FINISHED_EXACT12'
  assert w['candidate_completion']['exit_code']==0 and w['controller_completion_status']=='COMPLETE_FIXED12'
 else:assert w['trainer']['same_live']
assert obs['warm']['2']['latest_native_stats'][0]*100==integrity['AP']
assert obs['warm']['2']['latest_native_stats'][3]*100==integrity['APs']
jobs={}
for s,q in obs['user'].items():
 assert q['code_SHA_verified'] and q['queue']['same_live'] and not q['failures']
 jobs[s]={}
 for a,j in q['jobs'].items():
  assert 0<=j['completed_epochs']<=12 and j['failure'] is None
  if j['trainer']:
   assert j['trainer']['same_live'] and j['controller']['identity_equals_receipt']
   assert j['full_preflight']['status']=='PASS_USER_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
   assert j['full_preflight']['identity_matches_receipt'] and j['first100']['all100_rows_exact']
   assert j['runtime_start']['identity_equals_receipt'] and j['runtime_start']['optimizer_coverage']['all_parameters_exactly_once']
  else:assert j['completed_epochs']==0
  stats=j['latest_stats']
  jobs[s][a]=dict(j,state='training' if j['trainer'] else 'awaiting_capacity',AP_points=stats[0]*100 if stats else None,
                 APs_points=stats[3]*100 if stats else None,compared_to_cached_BASE12=False,fixed12_efficacy_not_claimed=True)
root=ROOT/'outputs/p2_phase_mean_remote_review_progress12_20261011_v1';root.mkdir(exist_ok=False)
proto=root/'CPU_prototype';proto.mkdir()
for s,o in obs['prototype'].items():
 folder=proto/f'seed{s}';folder.mkdir()
 for n,item in o['records'].items():(folder/n).write_bytes(base64.b64decode(item['base64'],validate=True))
for family,seeds in inputs['observations'].items():
 for s,item in seeds.items():(root/f'{family}_seed{s}.json').write_bytes(item['payload'].encode('utf8'))
(root/'warm_seed2_primary_integrity.json').write_bytes(inputs['warm_seed2_primary_integrity']['payload'].encode('utf8'))
write(root/'transports.json',{**{f:{s:i['transport'] for s,i in ss.items()} for f,ss in inputs['observations'].items()},
                            'warm_seed2_primary_integrity':inputs['warm_seed2_primary_integrity']['transport']})
paper=ROOT/'英文小论文/文献_相关原始论文_20261004/167_MessDet_ICCV2025_retry.pdf'
assert sha(paper)=='4b1803f6e522a2447468fffd6fa1f9fed0411ae2fea06cfda1d185601e9cd5ed'
paper_lesson=dict(title='Measuring the Impact of Rotation Equivariance on Aerial Object Detection',venue='ICCV 2025 main',
 source='https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_Measuring_the_Impact_of_Rotation_Equivariance_on_Aerial_Object_Detection_ICCV_2025_paper.pdf',
 local_pdf=str(paper),sha256=sha(paper),read_pages_one_based=[1,3,4,5,6,7,8],already_local_reused=True,
 lesson='The paper measures a structural property across layers and contrasts sampling designs. Its strict rotation-equivariant sampling improved the equivariant RE-Net variant, while the ordinary CNN comparison did not improve. Parameter counts differed, and DOTA/DIOR-R oriented detection with group channels differs from our ordinary VisDrone backbone. We borrow the property-and-counterfactual validation approach; the results do not establish a benefit for replacing our sampler or attention.',
 prototype_connection_is_our_inference=True,direct_attention_or_equivariance_module_transplanted=False,
 P2_phase_mean_is_not_rotation_equivariance_or_Fourier_phase_attention=True)
scope=dict(P2_original_branch_parameters=17440,P2_original_extra_state_tensors=7,
 P2_inserted_after_backbone_stage1_and_before_stage2_3=True,P2_adds_fourth_encoder_level=False,
 native_encoder_locations_at640=8400,native_queries=300,
 identical_parameter_count_is_not_equal_functional_capacity=True,
 reduced_feature_2x2_permutation_reversible=True,entire64_to16_branch_lossless=False,
 within_cell_mean_preserved_and_zero_sum_contrast_removed=True,
 zero_restore_initial_output_exact_P3=True,zero_restore_initial_reduce_gradient_zero=True,
 CUDA_tensor_operations_and_context_initialization=0,CUDA_availability_and_initialized_queries_were_used=True,
 exact_initial_common794_full_detector_assertion_performed=False,full_detector_AMP_or_AP_proven=False,
 failed_old_FineED_degraded_route_related_but_not_relaunched=True,
 old_FineED_scope='Old GeometryField P2 memory and dual-state decoder with raw-P2 pool-then-bilinear degradation; current branch-only mean control removes within-cell contrast after reduced-feature folding. Related information-removal question; no novelty or success claim.',
 next_action='Review original P2 fixed12 both seeds and error evidence. Only then, if scientifically warranted, bind a separate full-detector phase-mean control, preflight actual loss/AMP and preserve native states. No automatic training, BASE repetition, failed-route retry or epoch extension.')
warm12=dict(seed=2,AP=integrity['AP'],APs=integrity['APs'],BASE12=integrity['BASE12'],
 delta_points=integrity['delta_points'],native_gate_passed=False,primary_integrity_verified=True,
 saved_EMA_independent_replay_pending=True,formal_terminal_reader_pending=True,both_seed_primary_training_complete=False,
 efficacy_gate_already_not_satisfied_on_seed2=True,training_extension_or_retry_allowed=False)
counts={'warm_stem':{s:w['completed_epochs'] for s,w in obs['warm'].items()}}
for s,js in jobs.items():
 for a,j in js.items():counts.setdefault(a,{})[s]=j['completed_epochs']
report=dict(status='BOTH_REMOTE_P2_CPU_PROTOTYPES_VERIFIED_WARM_SEED2_PRIMARY12_BELOW_BASE_USER_TRAINING_CONTINUES',
 local_saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 remote_observation_UTC={f:{s:o['UTC'] for s,o in ss.items()} for f,ss in obs.items()},
 warm_primary_integrity_UTC=integrity['UTC'],actual_completed_epochs=counts,user_jobs=jobs,warm_fixed12_seed2=warm12,
 P2_CPU_prototype=dict(seeds=[1,2],cases_each=6,branch_forwards_each=8,parameters_each=17440,initial_states_each=7,
                      actual_workers_stopped=True,frozen_user28_sources_unchanged=True,training_epochs=0,new_accuracy_experiments=0,scope=scope),
 paper_lesson=paper_lesson,
 prior_both_seed_stem_control_review='../stem_both_controls_primary_review12_20261011_v1/report.json',
 prior_random_native_review='../stem_random_primary_collection12_20261011_v1/analysis_summary.json',
 source_frozen_training_files_changed=False,new_BASE_forwards=0,new_BASE_training_epochs=0,
 existing_process_signals_this_turn=0,configuration_training_retry=False,
 new_CPU_prototype_forwards_total=16,new_CPU_prototype_training_epochs=0,
 read_only_integrity_extra_detector_forwards=0,read_only_integrity_extra_evaluator_calls=0,
 new_training_configuration_started_by_this_turn=0,
 queued_original_user_attention_seed2_started_by_existing_dispatcher=True,
 root_proven=False,novelty_proven=False,goal_complete=False,current_goal_turn='progress',no_progress_streak=0)
write(root/'report.json',report)
text=[
 '本轮审查与两台远程实验进度',
 '',
 '保留预训练原始 Stem、另加增强残差支路：种子2已固定12轮，原生AP %.4f、APs %.4f；同种子已有基线AP %.4f、APs %.4f，差值 %.4f / %.4f。两项都低于基线。'%(warm12['AP'],warm12['APs'],warm12['BASE12']['AP'],warm12['BASE12']['APs'],warm12['delta_points']['AP'],warm12['delta_points']['APs']),
 '训练正完成记录、最后权重SHA、548张阶段文件SHA、38759个GT标识和保存的精度数组计算均通过；独立EMA前向及正式终点读取仍待GPU空闲。种子1本次观察10/12轮。这条配置已不满足两种子都提升的要求，不延长或重跑。',
 '',
 '你原来的独立改动仍按每配置最多12轮继续：',
]
for s,js in jobs.items():
 text.append('服务器/种子%s：'%s+'；'.join('%s %d/12（AP %s，APs %s）'%(a,j['completed_epochs'],'%.4f'%j['AP_points'] if j['AP_points'] is not None else '未出数','%.4f'%j['APs_points'] if j['APs_points'] is not None else '未出数') for a,j in js.items()))
text.extend([
 '以上中途数字尚未与固定BASE12作提升判定。原有28份源码一致；已训练的每项通过640/832完整检测器AMP、真实损失反传和前100输入一致性检查。',
 '',
 'P2机制对照：两台服务器CPU各6项检查通过，各8次实际支路前向；没有新训练和AP实验。',
 'P2接在stage1之后、stage2/3之前，会继续影响P3/P4/P5；没有增加第四层Encoder输入。支路先64→16通道压缩，再把2×2四个位置折入通道，零初始化投影后加到P3。',
 '对照保持17,440参数及7份初始状态，只把每个2×2单元的四位置特征替换为均值，删除单元内零和差异。160²、208²及两种奇数边长形状、初始P3恒等、梯度和BN更新均通过。',
 '四位置重排本身可逆，64→16压缩并不无损。参数量相同仍会改变特征空间秩和优化条件，因此不能把未来AP差异直接认定为某一种信息的独立因果贡献。',
 'CUDA_calls=0字段仅指没有CUDA张量计算和上下文初始化；检查程序确实调用了CUDA可用性查询。没有完整检测器、AMP或AP的有效性证明。',
 '旧FineED也做过信息退化对照，已保留负结果。本对照是原始P2到P3支路的不同位置与结构，当前只做CPU工程检查，不重跑旧失败路线，也不宣称创新。',
 '',
 '论文方法学习：阅读本地已有MessDet（ICCV 2025）第1、3、4、5、6、7、8页。借鉴逐层度量结构性质、用反事实检查解释的做法。其严格下采样收益依赖旋转等变网络，普通CNN比较未改善且参数量不同，不能直接移植为当前模型的收益依据。',
 '原始P2双种子固定12轮完成后，结合错误证据再判断是否值得独立绑定上述对照，先做完整模型AMP/真实损失检查。当前原始P2继续训练；CPU对照没有自动启动12轮训练。',
 '',
 '两台远程观察UTC：'+json.dumps(report['remote_observation_UTC'],ensure_ascii=False),
 '目标仍进行中；没有新增基线训练/前向、没有训练重启或超出12轮。',
])
(root/'审查与远程实验进度.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
write(root/'manifest.json',dict(schema='p2_phase_mean_remote_review_progress12_v1',
 source_sha256=sha(__file__),input_sha256=sha('outputs/p2_phase_mean_remote_review_progress12_20261011_v1_inputs.json'),
 files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}))
evidence='../p2_phase_mean_remote_review_progress12_20261011_v1/report.json'
def update(directory,version,changes):
 base=ROOT/'outputs'/directory;old=base/f'execution_state_v{version}.json';v=read(old)
 v.update(changes);v.update(supersedes=old.name,goal_complete=False)
 write(base/f'execution_state_v{version+1}.json',v)
update('warm_stem12_20261011_v1',14,dict(status='SEED2_NATIVE_FIXED12_BELOW_BASE_SEED1_TRAINING_REPLAY_PENDING',
 evidence=evidence,actual_new_training_epochs=counts['warm_stem'],trainers={s:w['trainer'] for s,w in obs['warm'].items()},
 phases={s:w['live']['phase'] for s,w in obs['warm'].items()},last_remote_observation_UTC={s:w['UTC'] for s,w in obs['warm'].items()},
 native_training_completions={s:w['native_training_completion'] for s,w in obs['warm'].items()},
 seed2_primary_integrity='../p2_phase_mean_remote_review_progress12_20261011_v1/warm_seed2_primary_integrity.json',
 seed2_fixed12_native=warm12,AP_improvement_proven=False,
 resource_rule='Seed2 own trainer stopped after positive12 closeout; saved-EMA replay waits existing predecessors and actual GPU idle. Seed1 own trainer continues. Original user configs retain their capacity guards.',
 pending='Finish seed1 fixed12 and both owned saved-EMA closeouts; preserve seed2 negative endpoint, no extension.'))
update('user_backbone12_20261011_v1',10,dict(evidence=evidence,jobs=jobs,
 completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
 last_remote_observation_UTC={s:q['UTC'] for s,q in obs['user'].items()},
 queue_processes={s:q['queue'] for s,q in obs['user'].items()},queue_phases={s:q['live']['phase'] for s,q in obs['user'].items()},
 GPU_resource={s:q['live'].get('resource') for s,q in obs['user'].items()},
 fixed12_efficacy_not_yet_claimed=True,standalone_attention_seed2_training_started=True))
update('stem_component12_20261011_v1',33,dict(latest_compact_status=evidence,current_goal_turn='progress',no_progress_streak=0,
 warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v15.json',
 standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v11.json',
 standalone_user_completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
 standalone_user_training_started={s:[a for a,j in js.items() if j['trainer']] for s,js in jobs.items()},
 standalone_user_queue_phases={s:q['live']['phase'] for s,q in obs['user'].items()},
 P2_phase_mean_CPU_prototype=evidence,CPU_prototype_training_epochs=0,warm_seed2_fixed12_native=warm12,
 remote_result_observers_verified_live_both_seeds=False,remote_result_observer_liveness_scope='No new diagnostic/collection waiter liveness assertion in this report; earlier observations retained in their own states.',
 existing_process_signals_this_turn=0,old_training_process_signals_this_turn=0,
 interpretation='Both remote P2 phase-mean engineering checks passed, no AP experiment. Warm seed2 exact12 primary metrics below cached BASE; seed1 continues. Original SiLU and P2 configurations continue on both servers; seed2 attention now trains after Warm freed capacity. Formal EMA and selection diagnostics still pending.'))
update('original_stem12_20261011_v1',36,dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v34.json'))
assert 'torch' not in sys.modules
print(json.dumps(dict(status=report['status'],counts=counts,warm_seed2=warm12,prototype_cases_each=6,goal_complete=False),ensure_ascii=False))
