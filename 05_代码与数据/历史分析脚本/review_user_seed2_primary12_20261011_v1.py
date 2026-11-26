"""Review two real user endpoints; preserve seed/replay limits and exact class arithmetic."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def report(p):
    m=read(p/'manifest.json')
    for n,h in m['files'].items():assert sha(p/n)==h,n
    return read(p/'report.json')
dest=ROOT/'outputs/user_backbone_seed2_review12_20261011_v1';dest.mkdir(exist_ok=False)
arms={};sources={};exact_cells=0
base_endpoint=ROOT/'outputs/scale_protection12_20261010_v1/terminal_collection/seed2_target_protect_v1/extracted/baseline_eval/endpoint_epoch12'
with np.load(base_endpoint/'precision_recall.npz',allow_pickle=False) as z:bp=z['precision'][:,:,:,1,2].copy()
assert np.all(bp>=0)
base=read(base_endpoint/'summary.json');assert abs(bp.mean()*100-base['native_stats'][3]*100)<1e-12
for arm in ('backbone_silu','p2_skip'):
    collection=ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed2_{arm}_v1'
    v=read(collection/'primary_verification_v1.json');assert v['status']=='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
    assert v['files_verified']==617 and v['actual_stage_images']==548 and v['all_collected_files_verified'] and v['all_ZIP_entries_CRC_verified']
    assert not v['saved_EMA_independent_replay_verified'] and not v['formal_terminal_reader_passed']
    stage=ROOT/f'outputs/user_backbone_primary_analysis12_20261011_v2/seed2_{arm}'
    order=ROOT/f'outputs/user_backbone_native_order12_20261011_v2/seed2_{arm}'
    s,o=report(stage),report(order)
    assert s['seed']==2 and s['arm']==arm and s['identity']==v['identity']
    assert s['all548_images'] and s['all38759_GT'] and s['all10_categories'] and s['completed_epochs']==12 and s['evaluated_weights']=='EMA'
    assert s['evidence']['primary_verification_sha256']==sha(collection/'primary_verification_v1.json')
    for family,r in (('user_backbone_primary_analysis12_20261011_v2',s),('user_backbone_native_order12_20261011_v2',o)):
        code=ROOT/'英文小论文/code'/family
        assert sha(code/'analyze.py')==r['analysis_sha256'] and sha(code/'protocol.json')==r['protocol_sha256']
        for n,h in read(code/'protocol.json')['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    endpoint=Path(v['downloaded_packet'])/'run/endpoint_epoch12'
    with np.load(endpoint/'precision_recall.npz',allow_pickle=False) as z:cp=z['precision'][:,:,:,1,2].copy()
    assert np.all(cp>=0) and abs(cp.mean()*100-s['candidate12']['APs'])<1e-12
    per_iou=(100*(cp-bp).mean(axis=(1,2))/10).tolist()
    assert abs(sum(per_iou)-s['delta_points']['APs'])<1e-12
    class_rows=[dict(class_id=c['class_id'],name=c['name'],BASE_APs=c['BASE_APs'],candidate_APs=c['warm_APs'],delta_APs_points=c['delta_APs_points'],contribution_to_total_APs_points=c['contribution_to_total_APs_points']) for c in s['class_APs_arithmetic']]
    assert abs(sum(c['contribution_to_total_APs_points'] for c in class_rows)-s['delta_points']['APs'])<1e-12
    cohorts={t:{group:{k:values for k,values in s['outcomes'][t][group].items() if k in ('GT','native_all_area_TP_gained','native_all_area_TP_lost','native_all_area_TP_net','supported_GT_by_boundary','gained_TP_first_observed_geometric_support_change','gained_TP_unchanged_three_boundary_support','lost_TP_candidate_outcomes')} for group in ('COCO_small','COCO_small_input_under4')} for t in ('0.5','0.75')}
    arms[arm]=dict(BASE12=s['BASE12'],candidate12=s['candidate12'],delta_points=s['delta_points'],native_gate_passed_seed2=s['native_gate_passed'],class_APs_arithmetic=class_rows,APs_delta_contributions_by_IoU=[dict(IoU=.5+i*.05,contribution_points=d) for i,d in enumerate(per_iou)],small_cohorts=cohorts,ranking_ambiguous_images=s['ranking_ambiguous_images'],score_mismatch_images=s['score_mismatch_images'],native_TP_proxy_conflicts=s['native_TP_proxy_conflicts'],identity=v['identity'],files_verified=617,stages_verified=548,other_seed_endpoint_pending=True,saved_EMA_independent_replay_pending=True,formal_terminal_reader_pending=True,training_extension_or_retry_allowed=False)
    for p in (collection/'primary_verification_v1.json',stage/'manifest.json',order/'manifest.json'):sources[p.relative_to(ROOT).as_posix()]=sha(p)
    exact_cells+=20200
    print(json.dumps(dict(arm=arm,delta=s['delta_points'],class_APs=class_rows,small_cohorts=cohorts),ensure_ascii=False),flush=True)
assert not arms['backbone_silu']['native_gate_passed_seed2'] and arms['p2_skip']['native_gate_passed_seed2']
assert arms['p2_skip']['identity']['added_parameter_count']==17440 and arms['backbone_silu']['identity']['added_parameter_count']==0
result=dict(status='TWO_ORIGINAL_USER_SEED2_FIXED12_PRIMARY_ENDPOINTS_VERIFIED_P2_PASS_SILU_APS_FAIL_OTHER_SEED_PENDING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=arms,source_proofs=sources,native_all_area_PR_cells_reproduced=exact_cells,native_PR_max_abs_error=0,
P2_source_scope='Original local user p2_skip code SHA17c10f8cc047ce7fa0e4bf395567757783eae3ac4eaa595694c595fe1d225d81. Stage0 stride4 feature64 channels is reduced to16, phase-unshuffled to64 at stride8, zero-initialized restore projects64 to256 and adds to native P3; P3/P4/P5 change. Encoder stays3 levels8400 locations and300 queries. Reduction is lossy; unshuffle alone is reversible. Not a fourth encoder level or a lossless64-channel P2 path.',
P2_supported_hypothesis='In seed2 geometric support for small GT improves at selected regressed references and persists to final boxes atIoU.50 and.75; native true matches also increase. This supports testing an early geometric benefit, but does not isolate location selection versus encoder-reference regression, phase information versus capacity, or prove a two-seed causal mechanism.',
SiLU_supported_observation='Seed2 selected-reference IoU.50 support changes+13, first-layerpre support-184 and final support-238; small native all-area TP net-218. Uniform small-target improvement by simply replacing all30 backbone ReLU activations is contradicted in this seed. No unique faulty activation, class-head or decoder layer is established.',
next_action='Finish existing seed1 SiLU/P2 and both attention jobs at fixed12. If originalP2 passes both seeds, compare its frozen phase-mean control after a full model AMP/criterion/backward review; otherwise preserve failure. No phase-mean training or new model route started by this review.',
two_seed_efficacy_proven=False,independent_saved_EMA_replay_verified=False,root_proven=False,novelty_proven=False,new_training_epochs=0,new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_native_matching_calls=0,analysis_source_sha256=sha(__file__))
write(dest/'report.json',result)
lines=['原始用户模块：第二种子固定第12轮结果与方向审查','每组617个文件、548张阶段输出已校验。复用既有同种子BASE12；没有重跑基线。独立EMA重放、正式终点读取以及另一种子仍待完成。','','配置 | AP | APs | ΔAP | ΔAPs']
for arm,a in arms.items():lines.append(f"{arm} | {a['candidate12']['AP']:.6f} | {a['candidate12']['APs']:.6f} | {a['delta_points']['AP']:+.6f} | {a['delta_points']['APs']:+.6f}")
lines+=['','小目标GT（26586个）的几何覆盖与原生全尺度匹配变化；这些数量不是APs分解：','配置 | IoU | 选中并回归的初始参考框覆盖Δ | 首层pre覆盖Δ | 最终覆盖Δ | 当前原生TP净变化']
for arm,a in arms.items():
    for t in ('0.5','0.75'):
        c=a['small_cohorts'][t]['COCO_small'];b=c['supported_GT_by_boundary']
        lines.append(f"{arm} | {t} | {b['encoder_initial_boxes']['delta']:+d} | {b['pre_after_first_layer_boxes']['delta']:+d} | {b['final_boxes']['delta']:+d} | {c['native_all_area_TP_net']:+d}")
lines+=['','P2第二种子：AP +0.5134，APs +0.3675。IoU0.50的小目标覆盖增益从初始+987到最终+415；IoU0.75从初始+532到最终+350。结果支持优先研究早期几何信息路径，但不能据此声称已证明“高频信息导致提升”。','SiLU第二种子：AP提高但APs下降0.2849，不满足要求。IoU0.50初始覆盖只增加13，首层pre减少184、最终减少238。不能将全部30个激活函数替换描述为小目标普遍受益。','你的P2接入不是四层Encoder：它通过P2→P3旁路影响后续三层特征。64→16压缩有损；pixel_unshuffle本身只是可逆重排。若双种子均通过，再验证相位均值对照；当前没有启动这个新训练。','','全部类别APs变化（宏平均每类权重1/10）：']
for arm,a in arms.items():
    lines.append(arm)
    lines.extend(f"  {c['name']}: ΔAPs {c['delta_APs_points']:+.6f}，对总APs贡献 {c['contribution_to_total_APs_points']:+.6f}" for c in a['class_APs_arithmetic'])
lines+=['','审查保留两组全部10类、10个IoU及分数重建不确定性。旧阶段消费者的warm_APs字段名沿用此前版本，本审查按实际identity转为candidate_APs；实际数据分别来自SiLU/P2，不是Warm Stem实验。','归因仍需：另一种子复现；拆分Top8400→300选择与选中框回归；相位信息对照；独立EMA校验。没有写成创新已成立或根因已证明。']
(dest/'原始SiLU与P2第12轮数字和分析.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
print(json.dumps(dict(status=result['status'],native_PR_cells=exact_cells)),flush=True)
