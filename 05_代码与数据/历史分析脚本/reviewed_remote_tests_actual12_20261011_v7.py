"""Record verified local endpoint-reader preparation and actual matched-error CPU checks."""
from pathlib import Path
import datetime,hashlib,json
R=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence={}
def record(p):
    p=Path(p);evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return read(p)
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def captured(p):
    j=record(p);o={}
    for k,v in j.items():
        raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256'];o[k]=json.loads(raw)
    return o
prior=R/'outputs/reviewed_remote_tests_actual12_20261011_v6';old=record(prior/'report.json');m=record(prior/'manifest.json')
assert sha(prior/'report.json')==m['report_sha256']
for n,v in m['files'].items():assert sha(R/n)==v['sha256'],n
c=captured(R/'outputs/reviewed_remote_tests_actual12_20261011_v7_inputs.json')
training=[]
for seed in (1,2):
    x=c[f'turn7_final_rank16_native_actual_seed{seed}'];previous=old['new_lowrank_actual_training'][seed-1]
    assert x['candidate_same_live'] and x['queue_same_live'] and x['controller_same_live']
    assert x['failure'] is None and x['queue_failure'] is None and x['first100']['all100_rows_exact']
    assert x['completed_epochs']>previous['completed_epochs'] and x['completed_epochs']<12
    assert x['audit']['actual_optimizer_steps']>previous['audit']['actual_optimizer_steps']
    training.append(dict(previous,observed_remote_UTC=x['UTC'],completed_epochs=x['completed_epochs'],audit=x['audit'],
        actual_handles=dict(queue=x['queue_handle'],supervisor=x['controller_handle'],trainer=x['candidate_handle']),
        exact_owner_handles_live=True,source_log_paths=x['source_log_paths'],source_log_sha256=x['source_log_sha256'],same_epoch_metrics=x['same_epoch_metrics']))
source=record(R/'outputs/p2_lowrank12_20261011_v3/source_review.json')
assert {p.name:sha(p) for p in (R/'英文小论文/code/p2_lowrank12_20261011_v3').iterdir()}==source['files']
readerfolder=R/'outputs/p2_lowrank12_local_reader_20261011_v1';readerreview=record(readerfolder/'source_review.json');readervalidation=record(readerfolder/'verification.json')
assert readervalidation['status']=='PASS_REAL_TWO_SEED_RUNTIME_AND_CACHED_BASE_LOCAL_READER_PREPARATION'
assert readervalidation['new_terminal_positive_records']==0 and len(readervalidation['negative_cases'])==23
assert {p.name:sha(p) for p in (R/'英文小论文/code/p2_lowrank12_local_reader_20261011_v1').iterdir() if p.is_file()}==readerreview['files']
decision=R/'英文小论文/code/p2_lowrank12_decision_20261011_v1';ds=record(decision/'protocol.json');assert sha(decision/'decide_fixed12.py')==ds['decision_sha256']
for p in decision.iterdir():evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
energyfolder=R/'outputs/P2_weight_direction_energy12_20261011_v1';energy=record(energyfolder/'report.json');em=record(energyfolder/'manifest.json')
assert sha(energyfolder/'report.json')==em['report_sha256']
for n,v in em['files'].items():assert sha(R/n)==v['sha256'],n
controls=R/'outputs/p2_coefficient_error_control12_20261011_v1';proposal=record(controls/'report.json');repair=record(controls/'remote_CPU_v2_source_review.json')
assert repair['v1_failure_preserved'] and repair['proposal_matrices_and_norm_error_constraints_unchanged'] and not repair['failed_training_retried']
actual_CPU=[];failures=[]
for seed in (1,2):
    f=c[f'turn7_error_control_actual_remote_CPU_seed{seed}'];x=c[f'turn7_error_control_v2_actual_remote_CPU_seed{seed}'];a=x['report']
    assert f['exit_code']==1 and f['report'] is None and "AttributeError: 'str' object has no attribute 'read_bytes'" in f['worker_log_tail']
    assert x['exit_code']==0 and x['completion']['exit_code']==0
    assert a['status']=='PASS_ACTUAL_CPU_RANK16_NORM_DC_ERROR_AND64_BASIS_OPERATOR_CHECK' and a['seed']==seed and a['rank']==16 and a['basis64_exact']
    assert not a['CUDA_initialized'] and not a['GPU_evaluation_completed'] and not a['root_or_efficacy_proven']
    assert a['full_detector_forwards']==a['BASE_forwards']==a['parent_forwards']==a['evaluator_calls']==a['optimizer_updates']==a['training_epochs']==0
    assert a['code_sha256']==repair['source_sha256'] and a['protocol_sha256']==repair['protocol_sha256']
    assert a['proposal_sha256']==sha(controls/f'seed{seed}_CPU/report.json') and a['weights_npz_sha256']==sha(controls/f'seed{seed}_CPU/weights.npz')
    actual_CPU.append(x);failures.append(f)
    for p in (controls/f'seed{seed}_CPU').iterdir():evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
for family in ('p2_coefficient_error_CPU12_20261011_v1','p2_coefficient_error_CPU12_20261011_v2'):
    for p in (R/'英文小论文/code'/family).iterdir():evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
out=R/'outputs/reviewed_remote_tests_actual12_20261011_v7';out.mkdir(exist_ok=False)
r=dict(old)
r.update(status='RANK16_ACTUAL_EPOCH5_6_LOCAL_TERMINAL_READER_READY_AND_TWO_REAL_ERROR_MATCHED_CPU_CONTROLS_CHECKED',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_immutable_report_sha256=sha(prior/'report.json'),new_lowrank_actual_training=training,
    local_rank16_terminal_reader=dict(source_review=readerreview,actual_runtime_and_cached_BASE_validation=readervalidation,joint_decision_source_binding=ds,
        actual_new_fixed12_candidate_packets_available=False,stats_and_all548_array_exact_replay_tolerance_unchanged=True),
    P2_weight_coefficient_direction_energy=energy,
    P2_new_coefficient_error_matched_control=dict(prespecified_NumPy_proposal=proposal,CPU_v1_engineering_failures_preserved=failures,
        CPU_v2_path_coercion_fix=repair,actual_two_remote_CPU_checks=actual_CPU,actual_native_GPU_evaluations=0,
        GPU_runtime_adapter_and_dispatch_prepared=False,failed_training_retried=False,model_replay_tolerance_changed=False),
    current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    pending=['Finish both live new rank16 configurations through fixed12 EMA and the first independent saved-EMA closeout; no extension, no failed-training retry or best-epoch substitution.',
        'Download successful terminal ZIPs and use the now-prepared local reader plus both-seed decision. If closeout fails, preserve native primary weights/logs and the exact failure without retrying it.',
        'The existing A SiLU/attention first-closeout and A DC16/SVD16 direction queues remain sequential behind known owners. Wait and collect each real once-only outcome.',
        'Implement and review a separate native runtime adapter for the new coefficient-error-matched rank16 weight. Both real CPU checks passed; no GPU outcome exists yet.',
        'Run that new altered-weight diagnostic once per seed after existing exact owners finish, without parent/BASE forwards or new training. Compare to saved DC16 and SVD16 outcomes; preserve A failed parent replay.',
        'Report BASE12 and originalP2 changes separately. A reduction of only0.1099% of full-model parameters does not establish a useful detector speedup or compensate for lower AP.',
        'Use final native stages and the matched-error direction control to distinguish global coefficient approximation from task/feature responses. Weight coefficients are not measured useful foreground information or a unique root.'])
r['current_scientific_conclusions']=[x for x in old['current_scientific_conclusions'] if not x.startswith('Actual latest rank16')]
r['current_scientific_conclusions']+=['Latest actual rank16: seed1 epoch5 AP14.091978/APs8.260936; seed2 epoch6 AP13.933903/APs8.170204. Same-epoch APs deltas against BASE are+0.419876/+0.300871. These are different ongoing epochs, not fixed12 efficacy.',
    'The top16 trained singular directions retain86.879%/86.338% matrix-coefficient energy; spatial-constant DC projection retains37.784%/38.484%. Norm-matching rescales them approximately1.07 versus1.61. Original DC-versus-SVD comparison also changes approximation error.',
    'A new deterministic rank16 direction rotation matches the DC normalized coefficient error and parent weight norm. Actual two-host CPU rank/norm/error/64basis checks passed, after preserving an early hash-path typing error from v1. No new GPU metric has been measured.']
write(out/'report.json',r)
lines=['真实进度与新对照准备','']
for x in training:
    row=x['same_epoch_metrics'][-1];a,b,p=row['candidate'],row['BASE'],row['original_P2']
    lines.append(f'种子{x["seed"]}，第{x["completed_epochs"]}轮已完成：AP{a["AP"]:.6f} / APs{a["APs"]:.6f}；同轮BASE AP{b["AP"]:.6f} / APs{b["APs"]:.6f}。')
    lines.append(f'对同轮BASE：AP{a["AP"]-b["AP"]:+.6f} / APs{a["APs"]-b["APs"]:+.6f}；对原P2：AP{a["AP"]-p["AP"]:+.6f} / APs{a["APs"]-p["APs"]:+.6f}。')
lines+=['两台实际训练进程已核实存活；仍按固定第12轮EMA判定，不能用当前轮或最好轮替代。',
    '', '本地终点核验脚本已准备并用两台真实运行配置及已有BASE产物校验；23个错误身份/配方/路径检查被拒绝。',
    '检查保留802个状态、794个共同状态、8个新增状态和6,176个新增参数的真实口径；原生PR和548图重放零容差要求不变。',
    '目前新的12轮终点/重放包仍未出现；这个准备通过不等于新模型精度通过。',
    '', '权重方向分析：SVD16保留约86%—87%矩阵系数能量，空间平均方向约38%；范数匹配放大约1.07倍和1.61倍，算子逼近误差也不同。',
    '因此需要补充同秩、同范数、同全局系数误差的对照。该权重已构造，两台实际CPU检查通过，误差差值约10^-9，64基向量算子等价通过。',
    '尚未进行新的GPU精度测试，下一步是实现/审查独立原生推理接口并在已有作业结束后各执行一次。',
    '矩阵系数能量不等于小目标有效信息量。新对照仍未匹配特征协方差、输出能量或目标/背景响应。',
    '', 'CPU检查v1在脚本哈希校验处发生路径类型错误，两台退出1；失败日志保留。v2只修复Path转换，拟议矩阵和约束没有改动。',
    '没有重跑失败训练；没有新增基线/父模型前向或训练轮数。',
    '参数减少11,264，分支减少64.59%，整个检测器只减少0.1099%；没有实测速度优势，不能以此抵消精度下降。']
(out/'真实进度与新对照准备.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(out/'manifest.json',dict(files=evidence,report_sha256=sha(out/'report.json'),text_sha256=sha(out/'真实进度与新对照准备.txt'),producer_sha256=sha(__file__)))
state=dict(status=r['status'],canonical_report='../reviewed_remote_tests_actual12_20261011_v7/report.json',canonical_report_sha256=sha(out/'report.json'),
    current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,new_lowrank_actual_training=training,
    local_terminal_reader_real_inputs_checked=True,new_candidate_fixed12_result_available=False,
    both_coefficient_error_controls_real_CPU_passed=True,coefficient_error_controls_GPU_evaluations=0,coefficient_CPU_v1_failures_preserved=True,
    new_BASE_forwards=0,new_BASE_training_epochs=0,failed_training_retried=False,
    two_seed_formal_efficacy_proven=False,root_or_novelty_proven=False,pending=r['pending'])
for family,num,previous in [('stem_component12_20261011_v1',48,47),('original_stem12_20261011_v1',51,50),
    ('user_backbone12_20261011_v1',25,24),('user_backbone_primary_collection12_20261011_v2',10,9),
    ('p2_subspace_diagnostic12_20261011_v1',8,7),('p2_lowrank12_20261011_v3',4,3),('p2_direction_diagnostic12_20261011_v2',2,1)]:
    write(R/f'outputs/{family}/execution_state_v{num}.json',dict(state,supersedes=f'execution_state_v{previous}.json'))
for family in ('p2_lowrank12_local_reader_20261011_v1','P2_weight_direction_energy12_20261011_v1','p2_coefficient_error_control12_20261011_v1'):
    write(R/f'outputs/{family}/execution_state_v1.json',state)
print(json.dumps(dict(status=r['status'],epochs=[x['completed_epochs'] for x in training],error_matched_controls_actual_CPU_passed=2,error_matched_GPU_results=0,new_BASE_forwards=0,goal_complete=False),ensure_ascii=False))
