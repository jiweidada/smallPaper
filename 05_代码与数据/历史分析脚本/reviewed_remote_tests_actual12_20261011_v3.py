"""All six fixed12 user endpoints and actual two-host branch engineering review."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence={}
def record(p):
    p=Path(p);evidence[p.relative_to(ROOT).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return read(p)
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def captures(p):
    out={}
    for k,v in record(p).items():
        raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256']
        out[k]=json.loads(raw)
    return out
previous=ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v2'
old=record(previous/'report.json');m=record(previous/'manifest.json')
assert sha(previous/'report.json')==m['report_sha256']
for n,v in m['files'].items():assert sha(ROOT/n)==v['sha256'],n
c=captures(ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v3_inputs.json')
t=captures(ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v3_terminal_inputs.json')
A=c['latest_A'];assert all(v['completed_epochs']==12 and not v['trainer_live'] and v['done']=='NATIVE_FIT_FINISHED_EXACT12' for v in A['user'].values())
primary=record(ROOT/'outputs/user_backbone_primary_collection12_20261011_v2/seed1_backbone_attention_v1/primary_verification_v1.json')
desc=record(ROOT/'outputs/user_backbone_primary_analysis12_20261011_v2/seed1_backbone_attention/report.json')
order=record(ROOT/'outputs/user_backbone_native_order12_20261011_v2/seed1_backbone_attention/report.json')
assert primary['status']=='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS' and primary['actual_stage_images']==548 and primary['files_verified']==617
assert desc['identity']==primary['identity'] and not desc['native_gate_passed'] and desc['completed_epochs']==12
assert order['status']=='USER_ARM_AND_CACHED_BASE_NATIVE_ALL_AREA_PR20200_CELLS_AND_GT_PREFIXES_REPRODUCED_EXACTLY'
assert desc['candidate12']['AP']==A['user']['backbone_attention']['AP'] and desc['candidate12']['APs']==A['user']['backbone_attention']['APs']
user=list(old['user_fixed12_results'])+[dict(seed=1,arm='backbone_attention',fixed_epochs=12,weights='EMA',BASE12=desc['BASE12'],candidate12=desc['candidate12'],delta_points=desc['delta_points'],primary_packet_verified=True,saved_EMA_independent_replay_and_formal_reader_verified=False,native_gate_passed_this_seed=False,source_checkpoint_sha256=primary['last_checkpoint_sha256'],native_order_all20200_cells_exact=True)]
assert len(user)==6 and len({(r['seed'],r['arm']) for r in user})==6
assert {r['arm'] for r in user if r['native_gate_passed_this_seed']}=={'p2_skip'}
branch=ROOT/'英文小论文/code/p2_lowrank_branch_review12_20261011_v2';spec=record(branch/'protocol.json')
assert all(sha(branch/n)==h for n,h in spec['files'].items())
assert sha(branch/'branch.py')==sha(ROOT/'英文小论文/code/p2_lowrank_branch_review12_20261011_v1/branch.py')
cpu=[]
for seed in (1,2):
    failure=c[f'lowrank_branch_actual_CPU12_v1_seed{seed}'];assert failure['exit_code']==1 and 'matrix_rank' in failure['log_tail'] and failure['report'] is None
    actual=c[f'lowrank_branch_actual_CPU12_v2_seed{seed}'];r=actual['report']
    assert actual['exit_code']==0 and r['status']=='PASS_PRIVATE_RANK16_BRANCH_ZERO_PATH_AND_GRADIENT_CPU_REVIEW'
    assert hashlib.sha256(json.dumps(r,indent=2).encode()).hexdigest()==actual['report_sha256']
    assert r['files_sha256']==spec['files'] and r['protocol_sha256']==sha(branch/'protocol.json') and r['seed']==seed
    assert r['parameters']==6176 and r['parameters_saved_vs_original_P2']==11264
    assert r['full_detector_forwards']==r['BASE_forwards']==r['optimizer_updates']==r['training_epochs']==0 and not r['CUDA_initialized'] and not r['training_allowed']
    assert r['RNG_preserved_by_fork'] and r['upstream_gradients_after_private_weight_perturbation_active'] and r['equivalent_rank_limited_stride2_conv_verified']
    assert r['rank_factor_product64']<=16 and r['rank_rounded_float32_product']<=16
    assert len(r['cases'])==3 and all(x['zero_output_exact'] and x['restore_gradient_active'] and x['mix_and_reduce_initial_gradient_zero'] for x in r['cases'])
    cpu.append(dict(seed=seed,report_sha256=actual['report_sha256'],UTC=actual['UTC'],report=r,v1_rank_checker_failure_preserved=True))
out=ROOT/'outputs/reviewed_remote_tests_actual12_20261011_v3';out.mkdir(exist_ok=False)
report=dict(status='ALL_SIX_USER_CONFIGURATIONS_FIXED12_VERIFIED_AND_TWO_HOST_LOW_RANK_BRANCH_CPU_REVIEWED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_immutable_report_sha256=sha(previous/'report.json'),
    user_fixed12_results=user,all_six_fixed12_primary_packets_verified=True,B_three_independent_saved_EMA_formal_packets_verified=True,A_user_independent_saved_EMA_formal_packets_verified=False,
    A_attention_small_GT_stage_description=desc['outcomes'] if 'outcomes' in desc else None,
    B_P2_fixed_weight_runtime_percent=old['seed2_P2_fixed_weight_runtime_percent'],B_P2_small_geometry_class=old['seed2_P2_rank16_small_GT_geometry_and_class_comparisons'],
    B_common_location_small_GT=old['seed2_common_physical_location_small_GT_comparisons'],
    actual_two_host_review_only_branch_checks=cpu,latest_A_remote=A,A_old_native_closeout_progress=t['A_actual_closeout_process_states12']['original']['reload_verification'],
    scientific_interpretation='P2 alone passes the AP/APs numeric gate in both seeds. SiLU and whole-backbone aggregation replacement fail that gate. B fixed-weight rank16 SVD retains a small gain and changes support before Decoder. The trainable rank16 branch is now engineering-reviewed on both hosts, but has no training/performance result and is a rank-limited stride2 convolution, not a novelty claim.',
    pending=['Finish A original native-owned savedEMA closeouts and collect formal packets; original enhancedStem savedEMA replay is already positive remotely.',
      'A already-deployed three repaired native-selection diagnostics and two P2 fixed-weight controls remain behind exact live owners.',
      'After the A same-scheme evidence arrives, review whether to train a distinct low-rank bridge from the common initialization, with at most12 epochs per seed.'],
    new_training_configurations_started_this_turn=0,new_BASE_training_epochs=0,new_BASE_forwards=0,existing_process_signals=0,failed_training_retried=False,
    review_only_branch_detector_training_allowed=False,new_lowrank_training_efficacy_proven=False,two_seed_formal_efficacy_complete=False,root_or_novelty_proven=False,
    goal_complete=False,current_goal_turn='progress',no_progress_streak=0)
write(out/'report.json',report)
lines=['已审查并在远程测试；三项改动两种子均已到固定第12轮EMA。','AP/APs单位为百分比；Δ为相对同种子既有BASE12的百分点；未重跑基线。','','改动 | A AP / APs | A ΔAP / ΔAPs | B AP / APs | B ΔAP / ΔAPs']
for arm in ('p2_skip','backbone_silu','backbone_attention'):
    a=next(r for r in user if r['seed']==1 and r['arm']==arm);b=next(r for r in user if r['seed']==2 and r['arm']==arm)
    lines.append(f"{arm} | {a['candidate12']['AP']:.4f} / {a['candidate12']['APs']:.4f} | {a['delta_points']['AP']:+.4f} / {a['delta_points']['APs']:+.4f} | {b['candidate12']['AP']:.4f} / {b['candidate12']['APs']:.4f} | {b['delta_points']['AP']:+.4f} / {b['delta_points']['APs']:+.4f}")
lines+=['','P2两种子数值门槛通过；SiLU和整组主干聚合替换未通过。六组训练完整产物已本地核验；A的独立EMA重放仍在收尾。','','B P2第12轮权重上的同秩16、同权重范数干预：','原P2 AP14.2248/APs9.0044；局部均值方向 AP11.0691/APs5.8166；最强16方向 AP14.2927/APs9.1255。','后者在Decoder前多覆盖370个小目标(IoU50)，最终多覆盖261个。IoU75新增319个原生TP中285个原先没有合格最终框。','分类相对次序也变化；这些计数不是APs贡献分解。A同样干预待前序收尾，尚未形成两种子结论。','','Stem同一物理位置对照：增强相对随机原Stem，附近入选位置少覆盖647个GT，初始IoU50框却多支持1687个GT。','相对高频RGB对照，新增2884个初始框支持中1756个来自双方共选位置；部分收益已出现在Decoder前。','增强Stem仍低于预训练正式BASE12。不能把高于随机原Stem写成高于正式基线。','','后续候选代码：学习到的rank16 P2空间混合；A/B均完成CPU分支审核。','初始化保持原P3输出；末端有效梯度；上游在末端变为非零后可学习；奇数尺寸、RNG和秩约束检查通过。','分支参数17440→6176，少11264；尚未接入整网训练，没有新的AP结果。','该算子等价于受秩限制的2×2 stride2卷积；不能将等价改写写成创新。','CPU检查中保留了浮点精度用错导致的秩判断失败；分支本身未改，修复检查后两台均通过。','','继续完成A同方案对照及正式重放，再决定下一项独立配置，每配置最多12轮；目标保持进行中。']
(out/'审查与远程实测结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(out/'manifest.json',dict(files=evidence,report_sha256=sha(out/'report.json'),text_sha256=sha(out/'审查与远程实测结果.txt'),producer_sha256=sha(__file__)))
state=dict(status='ACTIVE_ALL_SIX_FIXED12_PRIMARY_RESULTS_AND_TWO_HOST_BRANCH_CPU_REVIEW',current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    canonical_report='../reviewed_remote_tests_actual12_20261011_v3/report.json',canonical_report_sha256=sha(out/'report.json'),all_six_user_fixed12_primary_verified=True,user_fixed12_results=user,
    latest_A_training=A,lowrank_branch_review_only_CPU_passed_seeds=[1,2],lowrank_detector_training_started=False,A_user_formal_closeout_pending=True,
    A_three_selector_and_two_P2_controls_status='waiting_exact_previous_live_owners',new_BASE_forwards=0,new_BASE_training_epochs=0,new_training_configurations=0,existing_process_signals=0,
    original_training_source_files_changed=False,two_seed_formal_efficacy_proven=False,root_or_novelty_proven=False,pending=report['pending'])
for family,number,prev in [('stem_component12_20261011_v1',44,43),('original_stem12_20261011_v1',47,46),('user_backbone12_20261011_v1',21,20),('user_backbone_primary_collection12_20261011_v2',6,5),('p2_subspace_diagnostic12_20261011_v1',4,3)]:
    write(ROOT/f'outputs/{family}/execution_state_v{number}.json',dict(state,supersedes=f'execution_state_v{prev}.json'))
print(json.dumps(dict(status=report['status'],fixed12_verified_user_results=len(user),both_host_branch_CPU_review_passed=True,new_lowrank_training_started=False,goal_complete=False)))
