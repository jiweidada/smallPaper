"""Preserve actual fixed12 branch algebra and native FP evidence without a causal AP claim."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def verified(folder):
    for n,h in read(folder/'manifest.json')['files'].items():assert sha(folder/n)==h,n
    return read(folder/'report.json')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
inputs=read(ROOT/'outputs/p2_equivalence_and_FP_progress12_20261011_v1_inputs.json');captured={}
for family,v in inputs.items():
    b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256']
    captured[family]=json.loads(b)
eqdir=ROOT/'outputs/p2_linear_equivalence_audit12_20261011_v1/seed2';eq=read(eqdir/'report.json');co=read(eqdir/'completion.json');start=read(eqdir/'start.json')
assert co['exit_code']==0 and co['report_sha256']==sha(eqdir/'report.json') and co['receipt']==start
assert captured['equivalence_observation']['report']==eq and captured['equivalence_observation']['completion']==co
assert not captured['equivalence_observation']['same_live'] and captured['equivalence_observation']['failure'] is None
dispatch=captured['equivalence_dispatch'];assert dispatch['pid']==start['pid'] and dispatch['start_ticks']==start['start_ticks']
code=ROOT/'英文小论文/code/p2_linear_equivalence_audit12_20261011_v1'
spec=read(code/'protocol.json');assert eq['source_sha256']==sha(code/'audit.py')==spec['audit_sha256'] and eq['protocol_sha256']==sha(code/'protocol.json')
source_review=read(ROOT/'outputs/p2_linear_equivalence_audit12_20261011_v1_source_review.json')
for n,h in source_review['files'].items():assert sha(code/n)==h,n
primary=read(ROOT/'outputs/user_backbone_primary_collection12_20261011_v2/seed2_p2_skip_v1/primary_verification_v1.json')
assert primary['identity']==eq['identity'] and primary['last_checkpoint_sha256']==eq['checkpoint_sha256']
assert eq['status']=='PASS_TRAINED_USER_P2_EMA_LINEAR_EQUIVALENCE_AND_PHASE_MEAN_CAPACITY_AUDIT'
assert eq['trained_restore_rank']==64 and eq['phase_mean_tied_restore_rank']==16
assert eq['parameter_count_both_branches']==17440 and eq['restore_coefficient_count_original']==16384 and eq['independent_coefficients_phase_mean']==4096
assert eq['basis64_mapping_verified'] and eq['wrong_phase_kernel_rejected'] and eq['original_EMA_branch_state_unchanged'] and eq['checkpoint_unchanged'] and eq['original28_training_sources_unchanged']
assert eq['full_detector_forwards']==eq['BASE_forwards']==eq['new_evaluator_calls']==eq['new_optimizer_updates']==eq['new_training_epochs']==0
assert not eq['scientific_AP_or_APs_measured'] and not eq['training_control_approved']
assert len(eq['cases'])==10 and [(c['batch'],*c['HW']) for c in eq['cases'][:5]]==[tuple(c) for c in spec['cases']]
for c in eq['cases']:
    assert c['parameters']==17440 and c['nonzero_reduction_and_restore_gradients']
    tol=spec['tolerances'][c['dtype']]
    assert c['output_max_abs_error']<=tol['atol'] and c['phase_mean_to_tied_kernel_max_abs_error']<=tol['atol']
sources={eqdir.relative_to(ROOT).as_posix()+'/'+p.name:sha(p) for p in eqdir.iterdir() if p.is_file()}
dest=ROOT/'outputs/p2_equivalence_and_user_FP_review12_20261011_v1';dest.mkdir(exist_ok=False)
runs={};native_cells=0;people=[]
for tag in ('BASE12','backbone_silu','p2_skip'):
    family='native_fp_geometry12_20261011_v1' if tag=='BASE12' else 'native_fp_geometry12_20261011_v2'
    tracefamily='native_fp_query_trace12_20261011_v1' if tag=='BASE12' else 'native_fp_query_trace12_20261011_v2'
    gp=ROOT/f'outputs/{family}/seed2_{tag}';tp=ROOT/f'outputs/{tracefamily}/seed2_{tag}'
    g,t=verified(gp),verified(tp);assert g['identity']==t['identity'] and g['tag']==t['tag']==tag
    assert sha(gp/'manifest.json')==t['native_FP_geometry_manifest_sha256']
    for r,f in ((g,family),(t,tracefamily)):
        source=ROOT/'英文小论文/code'/f
        assert sha(source/'analyze.py')==r['analysis_sha256'] and sha(source/'protocol.json')==r['protocol_sha256']
        for n,h in read(source/'protocol.json')['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    if tag=='BASE12':
        packet=ROOT/'outputs/scale_protection12_20261010_v1/terminal_collection/seed2_target_protect_v1/extracted';run=packet/'baseline_eval'
    else:
        v=read(ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed2_{tag}_v1/primary_verification_v1.json')
        assert v['identity']==g['identity'];run=Path(v['downloaded_packet'])/'run'
    endpoint=run/'endpoint_epoch12';summary=read(endpoint/'summary.json')
    assert sha(endpoint/'manifest.json')==g['proof']['endpoint_manifest_sha256']
    for n,h in read(endpoint/'manifest.json')['files'].items():assert sha(endpoint/n)==h,n
    native=npz(endpoint/'precision_recall.npz')['precision'][:,:,:,0,2];ga=npz(gp/'native_FP_categories.npz');pr=np.zeros_like(native)
    for cid in range(10):
        codes=ga[f'class{cid}_codes'];assert codes.shape[0]==10
        assert np.all(np.diff(ga[f'class{cid}_native_scores'])<=0)
        # Exact native category GT denominator, no approximate score matching or rematching.
        gt=summary['categories_percent'][cid]['GT'] if 'GT' in summary['categories_percent'][cid] else None
        if gt is None:
            annotation=read(ROOT/'outputs/scale_protection12_20261010_v1/terminal_collection/seed2_target_protect_v1/extracted/reference/val_annotations.json')
            gt=sum(a['category_id']==cid for a in annotation['annotations'])
        tc=np.cumsum(codes==0,axis=1).astype(float);fc=np.cumsum((codes!=0)&(codes!=9),axis=1).astype(float)
        rc=tc/gt;precision=tc/(tc+fc+np.spacing(1));envelope=np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1]
        for ti in range(10):
            ids=np.searchsorted(rc[ti],summary['protocol']['recall_thresholds'],side='left');valid=ids<codes.shape[1]
            pr[ti,valid,cid]=envelope[ti,ids[valid]]
            for row in g['classes'][cid]['by_IoU'][ti]['prefixes']:
                n=row['rank'];assert row['TP']==tc[ti,n-1] and row['FP']==fc[ti,n-1]
    assert np.array_equal(pr,native)
    native_cells+=native.size
    gp1000=next(p for p in g['classes'][1]['by_IoU'][0]['prefixes'] if p['rank']==1000)
    tp1000=next(p for p in t['classes'][1]['by_IoU'][0]['prefixes'] if p['rank']==1000)
    people.append(dict(tag=tag,geometry=gp1000,trace=tp1000))
    runs[tag]=dict(geometry_manifest_sha256=sha(gp/'manifest.json'),trace_manifest_sha256=sha(tp/'manifest.json'),identity=g['identity'],all10_classes_and10_IoUs_retained=True,native_PR_cells_verified=int(native.size),native_PR_max_abs_error=0,detections=g['detections'],multiple_admissible_query_entries=g['detections_with_multiple_admissible_queries'],all_class_IoU50_final_prefixes=[c['by_IoU'][0]['prefixes'][-1] for c in g['classes']])
    sources[gp.relative_to(ROOT).as_posix()+'/manifest.json']=sha(gp/'manifest.json');sources[tp.relative_to(ROOT).as_posix()+'/manifest.json']=sha(tp/'manifest.json')
assert native_cells==30300
user=verified(ROOT/'outputs/user_backbone_seed2_review12_20261011_v1')
classrows=user['arms']['p2_skip']['class_APs_arithmetic'];vehicle=sum(c['contribution_to_total_APs_points'] for c in classrows if c['name'] in ('car','van'))
delta=user['arms']['p2_skip']['delta_points']['APs'];positive=[c['name'] for c in classrows if c['delta_APs_points']>0];negative=[c['name'] for c in classrows if c['delta_APs_points']<0]
result=dict(status='TRAINED_P2_OPERATOR_EQUIVALENCE_AND_USER_FP_TRACE_VERIFIED_CHANGED_CONTROL_DECISION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=2,P2_fixed12=user['arms']['p2_skip']['candidate12'],P2_delta_points=user['arms']['p2_skip']['delta_points'],runs=runs,people_IoU50_prefix1000=people,native_all_area_PR_cells_verified=int(native_cells),native_PR_max_abs_error=0,
trained_P2_equivalence=dict(checkpoint_sha256=eq['checkpoint_sha256'],parameters=17440,cases=10,original_rank=64,phase_mean_rank=16,restore_original_coefficients=16384,phase_mean_independent_coefficients=4096,phase_contrast_weight_energy_fraction=eq['phase_contrast_weight_energy_fraction'],max_output_abs_error=max(c['output_max_abs_error'] for c in eq['cases']),max_input_gradient_abs_error=max(c['P2_gradient_max_abs_error'] for c in eq['cases']),max_parameter_gradient_abs_error=max(c['parameter_gradient_max_abs_error'] for c in eq['cases']),full_detector_forwards=0,new_training_epochs=0),
class_APs_arithmetic=dict(car_and_van_contribution_points=vehicle,total_delta_points=delta,ratio=vehicle/delta,positive_classes=positive,negative_classes=negative),
source_proofs=sources,source_user_stage_review_sha256=sha(ROOT/'outputs/user_backbone_seed2_review12_20261011_v1/report.json'),analysis_source_sha256=sha(__file__),
changed_next_action='Continue four existing jobs at fixed12. OriginalP2 two-seed and independent replay gates still apply. Do not treat unshuffle-to-stride2 reparameterization as a new improvement or launch a12epoch phase-mean experiment as a sole matched-capacity causal test. Before a new trained variant, separate the trained branch DC/contrast orientation from reduced operator rank through rank-controlled runtime interventions on the same savedEMA, then test a specific supported mechanism.',
rejected_interpretations=['Unshuffle alone generates information or proves a novel phase mechanism.','Phase mean has matched functional capacity merely because both store17440 nominal parameters.','Phase contrast coefficient energy61.5% equals anAP contribution61.5%.','P2 uniformly improves every category or suppresses all background-like false positives.','The people score-prefix table is anAPs matching decomposition or identifies corresponding queries across runs.'],
small_area_raw_native_matching_not_saved=True,new_model_forwards=0,new_BASE_forwards=0,new_training_epochs=0,new_optimizer_updates=0,new_evaluator_calls=0,new_native_matching_calls=0,root_proven=False,novelty_proven=False,two_seed_efficacy_proven=False,independent_saved_EMA_replay_verified=False)
write(dest/'report.json',result)
lines=['P2提升解释与下一组对照的审查','实际固定第12轮EMA系数，远程CPU验证；没有新训练、完整模型前向或基线前向。','','原始P2重排+1×1还原与2×2步长2卷积等价：K[o,c,dy,dx]=W[o,4c+2dy+dx,0,0]。奇数尺寸仍在右边/下边补零。','验证：64个基向量和错误相位映射反例；5个尺寸×float64/float32，共10组正反向。前向最大误差 %.12g；输入梯度最大误差 %.12g。'%(result['trained_P2_equivalence']['max_output_abs_error'],result['trained_P2_equivalence']['max_input_gradient_abs_error']),
'相位均值把4个空间核位置系数绑成相同值；已训练还原矩阵秩从64降到16，自由还原系数从16384降到4096。保留17440个名义参数不代表容量相同。','训练后相位对比分量的权重平方范数占61.5164%，仅说明存在该线性分量；不是AP贡献，也不是保留图像高频的证据。','','people类原生全尺度前1000条（IoU0.50，全部10类/10IoU已保存）：','配置 | TP/FP | 同类已占用目标FP | 异类重叠FP | 低于0.1重叠FP | 同类FP最早支持：初始/pre/最终']
for row in people:
    c=row['geometry']['counts'];s=row['trace']['first_FP_same_occupied_GT_support']
    lines.append(f"{row['tag']} | {row['geometry']['TP']}/{row['geometry']['FP']} | {c['same_class_eligible_GT_all_already_native_matched_earlier']} | {c['other_class_overlap_at_threshold']} | {c['overlap_below0_1_any_class']} | {s['selected_regressed_encoder_reference']}/{s['first_layer_pre']}/{s['final_only']}")
lines+=['','P2在此例中降低同类占用和异类重叠误检，但低重叠误检从9增加到22，不能写成普遍背景抑制。','P2总APs增加 %.6f，其中car+van的宏平均贡献为%.6f，约%.2f%%。正增益类别：%s；负增益类别：%s。'%(delta,vehicle,100*vehicle/delta,'、'.join(positive),'、'.join(negative)),
'三组误检编码独立还原30,300个原生全尺度PR单元，误差0。上述前段不能替代小面积原生匹配/APs解释。','','下一步决策：','1. 继续已启动的原始用户四组训练，固定第12轮判定。','2. 不把等价卷积改写当作新改进，不直接启动“相位均值参数一样所以容量一致”的12轮归因实验。','3. 先在同一训练后权重上，用明确的秩控制比较信息方向（例如DC投影和最优秩16投影），这是运行时干预，不是新训练收益或纯训练容量归因。之后再选择有依据的新模型机制。','4. P2另一种子、独立EMA重放、Top8400→300与参考框回归的区分仍未完成；保持目标进行中。']
(dest/'P2算子等价与归因对照审查.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
write(eqdir/'manifest.json',dict(files={p.name:sha(p) for p in eqdir.iterdir() if p.is_file()}))
print(json.dumps(dict(status=result['status'],native_PR_cells=int(native_cells),rank64_to16=True,car_van_APs_contribution=vehicle,car_van_ratio=vehicle/delta)),flush=True)
