"""Current evidence ledger: primary, formal replay, actual remote tests and failures."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence={}
def record(p):
    p=Path(p);evidence[p.relative_to(ROOT).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return read(p)
def captures(p):
    x=record(p);out={}
    for k,v in x.items():
        if isinstance(v,dict) and 'payload' in v:
            raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256'];out[k]=json.loads(raw)
    return out
def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
original=captures(ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v1_inputs.json')
terminal=captures(ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v1_terminal_inputs.json')
user=[]
for seed,arms in [(1,['p2_skip','backbone_silu']),(2,['p2_skip','backbone_silu','backbone_attention'])]:
    for arm in arms:
        primary=record(ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed{seed}_{arm}_v1/primary_verification_v1.json')
        assert primary['status']=='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS' and primary['actual_stage_images']==548
        desc=record(ROOT/f'outputs/user_backbone_primary_analysis12_20261011_v2/seed{seed}_{arm}/report.json');order=record(ROOT/f'outputs/user_backbone_native_order12_20261011_v2/seed{seed}_{arm}/report.json')
        assert desc['identity']==primary['identity'] and desc['completed_epochs']==12
        assert order['status']=='USER_ARM_AND_CACHED_BASE_NATIVE_ALL_AREA_PR20200_CELLS_AND_GT_PREFIXES_REPRODUCED_EXACTLY'
        formal=None
        if seed==2:
            formal=record(ROOT/f'outputs/user_seed2_formal_collection12_20261011_v1/seed2_{arm}/verification.json')
            assert formal['status']=='PASS_USER_FORMAL_FIXED12_PRIMARY_AND_SAVED_EMA_EXACT' and formal['independent_saved_EMA_replay_verified'] and formal['formal_terminal_reader_passed']
            assert formal['identity']==primary['identity'] and formal['native_gate_passed_this_seed']==desc['native_gate_passed']
        user.append(dict(seed=seed,arm=arm,fixed_epochs=12,weights='EMA',BASE12=desc['BASE12'],candidate12=desc['candidate12'],delta_points=desc['delta_points'],primary_packet_verified=True,saved_EMA_independent_replay_and_formal_reader_verified=bool(formal),native_gate_passed_this_seed=desc['native_gate_passed'],source_checkpoint_sha256=primary['last_checkpoint_sha256'],native_order_all20200_cells_exact=True))
oldformal=[]
for arm in ['original_stem','native_stem_scratch','detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample']:
    f=record(ROOT/f'outputs/stem_fixed12_seed2_formal_collection12_20261011_v1/seed2_{arm}/verification.json')
    assert f['status'].startswith('PASS_') and f['all548_stage_arrays_reloaded_exact'] and f['first100_inputs_and_pre_forward_RNG_exact']
    oldformal.append(dict(seed=2,arm=arm,reader_status=f['status'],formal_packet_verified=True,source_report_sha256=evidence[f'outputs/stem_fixed12_seed2_formal_collection12_20261011_v1/seed2_{arm}/verification.json']['sha256']))
runtime=record(ROOT/'outputs/p2_subspace_collection12_20261011_v1/seed2/verification.json');assert runtime['status']=='PASS_COMPLETE_P2_RUNTIME_ARTIFACTS_AND_TWO_NATIVE_ENDPOINTS'
selection=record(ROOT/'outputs/native_selection_collection12_20261011_v2/seed2_all_three_tensor_verification_v4.json');assert len(selection['records'])==3
for r in selection['records']:
    f=record(ROOT/f"outputs/native_selection_collection12_20261011_v2/seed2_{r['arm']}/tensor_verification_v4.json")
    assert f['status']=='PASS_ACTUAL548_SELECTION_TENSORS_AND_FORMAL_NATIVE_ENDPOINT_STAGE_JOIN' and f['all548_observed_anchors_exact_saved_EMA_buffer'] and f['nominal_feature_cell_assignment_preserved']
    assert sha(ROOT/f"outputs/native_selection_collection12_20261011_v2/seed2_{r['arm']}/tensor_verification_v4.json")==r['report_sha256']
anchor=record(ROOT/'outputs/native_selection_anchor_audit12_20261011_v2/seed2/report.json')
assert anchor['status']=='PASS_CHECKPOINT_BOUND_EMA_MODEL_AND_NATIVE_GENERATED_ANCHOR_TENSOR_AUDIT'
for arm,c in anchor['cases'].items():
    assert c['model_equal_generated'] and c['model_generated_max_abs_difference']==0
    p=ROOT/f'outputs/native_selection_anchor_audit12_20261011_v2/seed2/{arm}.npz';assert sha(p)==c['export_sha256'];evidence[p.relative_to(ROOT).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
for seed in (1,2):
    cpu=original[f'native_selection_v3_CPU_actual12_seed{seed}'];assert cpu['completion']['exit_code']==0 and cpu['import']['status']=='PASS_REAL_ENTRYPOINT_STDLIB_QUEUE_AND_TORCH_IMPORT' and cpu['fixture']['status']=='PASS_NATIVE_PRODUCER_AND_STAGE_JOIN_CPU_FIXTURES'
a=terminal['user_A_final_training_snapshot12'];b=terminal['native_selection_v3_B_three_terminal12']
assert b['queue_completion']['status']=='ALL_THREE_NATIVE_SELECTION_DIAGNOSTICS_FINISHED' and b['collector_completion_status']=='ALL_THREE_DIAGNOSTICS_PACKAGED_NO_NEW_MODEL_CALLS' and not b['queue_same_live']
P2=original['p2_subspace_A_CPU_actual_compact12_seed1'];assert P2['CPU_exit']==0 and P2['CPU_status']=='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS' and P2['barrier_live'] and not P2['barrier_failure_exists']
out=ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v1';out.mkdir(exist_ok=False)
report=dict(status='REVIEWED_AND_ACTUALLY_TESTED_FIVE_FIXED12_USER_ENDPOINTS_TWO_P2_RUNTIME_CONTROLS_THREE_NATIVE_SELECTION_RECORDS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),user_fixed12_results=user,
    original_six_Stem_formal_seed2_results=oldformal,P2_runtime_seed2=runtime,selection_seed2=selection,anchor_buffer_audit=anchor,
    CPU_native_import_and_native_method_fixture_checks_passed_seeds=[1,2],actual_native_selection_replays_completed_seed2=3,
    actual_runtime_P2_interventions_completed_seed2=2,actual_new_diagnostic_model_forwards_seed2=411+274,actual_new_diagnostic_BASE_forwards=0,
    new_training_configurations_started_this_turn=0,existing_training_configuration_epochs_capped_at12=True,
    seed1_current_training=a,seed1_P2_CPU_and_exact_owner_barrier=P2,
    failures_preserved=[dict(scope='v2 native8400 observer',reason='queue.py shadows stdlib queue; PyTorch import failed before model construction; repaired only in new v3 instrumentation',failed_training_retried=False),dict(scope='source delivery',reason='Oversized shell argument rejected before deployment; byte-exact reviewed ZIP deployed instead'),dict(scope='tensor reader v3',reason='Generated fp32 grid1e-6 assumption falsified by checkpoint EMA buffers; v4 requires exact savedEMA arrays for every image and preserved feature-cell assignment',replay_repeated=False),dict(scope='CPU anchor audit v1 metadata',reason='Multiple files named ema.py; positive exported tensors retained; v2 binds engine/optim/ema.py and actual generator source'),dict(scope='PTY observation envelope',reason='Malformed large JSON rejected; compact read-only recapture used without experiment restart')],
    interpretation='P2 fixed12 primary AP and APs improve in both seeds; seed1 savedEMA formal replay is still pending. Rank16 norm-matched controls on seed2 show mean averaging damages APs while preserving strongest learned directions keeps accuracy with a small gain. This is a fixed-weight runtime finding, not trained efficacy or a unique phase/high-frequency root. Seven of ten class APs improve and three decline under SVD; feature/output energy and operator approximation error are not matched.',
    pending=['A attention:10/12 at last actual observation; finish existing authorized configuration','A native-owned savedEMA/formal closeouts, then three repaired native8400 diagnostics','A same frozen two P2 runtime controls: CPU passed; GPU waits exact prior owner and positive P2 replay','Prepare checkpoint-bound anchor handling for strict six-record offline comparison; do not use failed nominal-grid consumer or claim six-record findings','Only after two-seed controls and failure-location evidence define a new mechanism trained from the common initialization, at most12 epochs each'],
    two_seed_P2_primary_numeric_gate_passed=True,two_seed_P2_formal_replay_efficacy_complete=False,new_runtime_control_training_efficacy_proven=False,unique_causal_root_proven=False,novelty_proven=False,goal_complete=False,current_goal_turn='progress',no_progress_streak=0)
write(out/'report.json',report)
lines=['已审查并上机测试：截至A远程UTC '+a['UTC'],'','固定第12轮EMA，与同种子既有BASE12比较；没有重训基线。','服务器/改动 | AP | APs | ΔAP | ΔAPs | 本地核验']
for r in user:lines.append(f"{'A' if r['seed']==1 else 'B'}/{r['arm']} | {r['candidate12']['AP']:.4f} | {r['candidate12']['APs']:.4f} | {r['delta_points']['AP']:+.4f} | {r['delta_points']['APs']:+.4f} | "+('完整产物+独立EMA重放' if r['saved_EMA_independent_replay_and_formal_reader_verified'] else '完整产物；EMA重放待完成'))
lines+=['','B：已有P2第12轮权重上的两个干预，各完整验证548张图。','局部均值/同秩16/同权重范数：AP11.0691，APs5.8166；相对原P2 −3.1557/−3.1878。','保留最强16方向/同权重范数：AP14.2927，APs9.1255；相对原P2 +0.0679/+0.1211。','SVD干预下7个类别APs升、3个降；这是固定权重干预，训练效果及两种子稳健性尚未成立。','','B三组原生选择诊断完成并下载：增强Stem、随机原始Stem、高频RGB对照。','各548张图的8400位置分类输出、原生Top300位置、选择后回归输出已核验。','所有原阶段数组与正式训练产物相同，选择前锚点+回归偏移可重建初始参考框，最大误差约8.8e-8。','这批锚点是未回归先验，不能当作8400个预测框来计算召回。','','修复并保留两类检查问题：queue.py遮蔽标准库造成导入失败；理论网格与EMA缓存轻微漂移造成离线假设失败。','实际模型锚点与生成网格一致；三组EMA缓冲区相同，中心最大偏差1.5544e-5。记录与各自EMA缓存逐项相同。','这解释了离线检查失败，未证明它造成AP变化，不据此重跑训练。','','A注意力当前10/12；P2/SiLU已12/12；A两组P2对照CPU已通过，等待前序GPU诊断和P2正向重放完成。','当前保留方向：验证P2中信息混合的方向，而非只增加注意力复杂度或把下采样等价改写当创新。','继续完成A同方案对照及严格两种子诊断，再决定下一组从共同初始化开始的12轮训练。']
(out/'审查与远程实测结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(out/'manifest.json',dict(files=evidence,report_sha256=sha(out/'report.json'),text_sha256=sha(out/'审查与远程实测结果.txt'),producer_sha256=sha(__file__)))
state=dict(status='ACTIVE_REVIEWED_REMOTE_TESTS_AND_CHECKPOINT_BOUND_NATIVE_SELECTION_EVIDENCE',previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,canonical_report='../reviewed_remote_tests_actual12_20261011_v1/report.json',canonical_report_sha256=sha(out/'report.json'),user_fixed12_results=user,latest_A_training=a,seed2_original_six_formal_packets_passed=True,seed2_user_three_formal_packets_passed=True,seed2_runtime_P2_controls_verified=True,seed2_three_native8400_tensor_packets_verified=True,seed1_same_P2_controls_CPU_passed=True,seed1_controls_GPU_status='waiting_exact_prior_diagnostic_owner_and_positive_P2_EMA_replay',new_training_configurations=0,new_BASE_forwards=0,new_BASE_training_epochs=0,original_training_source_files_changed=False,existing_process_signals=0,two_seed_formal_efficacy_proven=False,root_or_novelty_proven=False,pending=report['pending'])
for family,number,prev in [('stem_component12_20261011_v1',42,41),('original_stem12_20261011_v1',45,44),('user_backbone12_20261011_v1',19,18),('user_backbone_primary_collection12_20261011_v2',4,3),('p2_subspace_diagnostic12_20261011_v1',2,1)]:
    write(ROOT/f'outputs/{family}/execution_state_v{number}.json',dict(state,supersedes=f'execution_state_v{prev}.json'))
print(json.dumps(dict(status=report['status'],fixed12_user_results=len(user),B_original_formal=len(oldformal),B_runtime_controls=2,B_native_selection_packets=3,A_attention_epochs=a['jobs']['backbone_attention']['completed_epochs'],goal_complete=False)))
