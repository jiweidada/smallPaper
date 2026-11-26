"""Publish verified diagnostic results and advance the active research state."""
from pathlib import Path
import datetime,hashlib,json

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/encoder_pool_diagnostic_20261010_v1'
CODE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    summary=read(OUT/'local_analysis/summary.json')
    verified=read(OUT/'local_analysis_verification.json')
    assert verified['status']=='PASS_LOCAL_FULL_ARRAY_RECOMPUTATION_ALL_GRAPHS_OPTIMALITY_AND_CSV_ROWS'
    assert summary['diagnostic_priority_gate_both_seed_ge10pp'] is False
    live=read(OUT/'final_live_completion_check.json')
    assert all(not any(r['original_job_pids_present'].values()) and r['compute_apps']=='' for r in live['receipts'])
    runs=[]
    for seed in (1,2):
        collected=OUT/f'collection_seed{seed}'
        capture=read(collected/'capture_v1/completion.json')
        controller=read(collected/'controller_completion.json')
        source=read(collected/'capture_v1/source_identity.json')
        start=read(collected/'controller_start.json')
        ref=read(OUT/f'analysis_reference_seed{seed}/analysis_completion.json')
        assert controller['exit_code']==ref['analysis_exit_code']==0
        assert capture['native_sources_unchanged'] and capture['original_forward_control_parity'] and capture['all_augmented_prior_outputs_exact']
        assert all(r['exact'] for r in capture['parameter_checks'])
        runs.append({'seed':seed,'controller_start':start,'capture':capture,'controller_completion':controller,'analysis_completion':ref,'native_source_files':len(source['native_files_sha256']),'checkpoint_sha256':source['checkpoint_sha256']})
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    next_work='Examine candidate production and localization coordinate constraints using recorded per-stage outputs, with an explicit discriminating hypothesis and frozen protocol before any new computation. The current pool diagnostic does not identify a representation, backbone, anchor, regression-head or decoder-reference root cause. Do not launch a selector, arbitrary attention module, closed range route, or another ROI/reclassification variant from this result. Any new intervention still requires <=12 total epochs/configuration, fixed epoch12 EMA vs same-seed BASE, APs increase and overall AP nondecline.'
    decision={'status':'FULL_POOL_MEASURED_NATIVE_TOPK_PRIORITY_GATE_NOT_MET','observed_local_utc':now,'full_encoder_pool_measured':True,'primary':summary['primary'],'frozen_priority_gate_both_seed_ge10pp':False,'selection_mechanism_training_promoted':False,'TopK_has_no_effect_proven':False,'encoder_or_backbone_root_cause_established':False,'candidate_production_priority':'Prefer investigating why accurate candidate boxes are not produced, after limited full-pool surplus was measured. This is a diagnostic priority, not a causal conclusion.','final_decoder_can_correct_some_early_geometry':'Aggregate unique coverage exceeds diagnostic full encoder coverage in both mature primary strata; the GT ownership differs by pool, so no same-query causal claim.','no_accepted_new_mechanism':True,'new_AP_result':False,'epochs_added':0,'model_forward_calls':92,'diagnostic_dense_head_calls':80,'model_backward_calls':0,'optimizer_steps':0,'automatic_retry':False,'normal_capacity_prior_gate_unchanged':False,'closed_routes_remain_closed':True,'next':next_work}
    save(OUT/'decision.json',decision)
    fallacies={
      '1_Simpson':'Seeds, endpoints, input sizes, class, density and three contexts retained separately. Primary is unchanged BASE132 ordinary val_eval.',
      '2_Ecological':'Repeated GT/image exposures across checkpoints and contexts are not independent samples. No generalization or p-value claim.',
      '3_Berkson':'Panel32 and GT<=300/post-resize short<4 strata are explicit selections; no full548-image prevalence/AP inference.',
      '4_Collider':'Score TopK selection and quality graph edges depend on model outputs; their association does not identify a harmful learning mechanism.',
      '5_Base_rates':'Every primary count uses329 GT. Geometric IoU eligibility, unique matching and class eligibility are separately reported; none equals COCO AP/TP/recall.',
      '6_Regression_to_mean':'No new trained intervention or best-epoch comparison. Fixed BASE12/132 snapshots are descriptive.',
      '7_Survivorship':'Both captures and analyses exit0; all predeclared40 batches/seed and330 capture files retained. Dense>300 descriptive strata excluded only as frozen.',
      '8_Look_elsewhere':'Many historical routes were screened and remain closed. Failing10pp priority gate is retained, not replaced with a favorable subgroup/threshold.',
      '9_Forking_paths':'Primary/gate before capture. Matching tie clarification before first derived geometry result. Post-result local graph instrumentation only verifies unchanged matching; disclosed timing.',
      '10_Correlation_causation':'Dense frozen head full-pool oracle and GT-assisted matching are not trainable/deployable interventions or causal evidence for AP gains.',
      '11_Reverse_causality':'Poor full-pool boxes may reflect feature, head, matching or optimization limitations. TopK omission can co-occur without being the primary cause. Train/eval contrast is not training trajectory.'}
    validation={'material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','scientific_scope':'ANALYZED','engineering_verification':'VERIFIED_ALL_ARRAY_SOURCE_SELECTION_GRAPH_AND_STATISTIC_PARITY'},'observed_local_utc':now,'runs':runs,'local_verification':verified,'final_live_check':live,'fallacy_scan':{'checked':11,'total':11,'items':fallacies},'clock_scope':'Local Windows and remote wall clocks differ. Exact deployed protocol hash/tool ordering establish freeze before capture; do not compare absolute cross-host timestamps as elapsed times.','limits':['Unselected full-pool predictions use the frozen native bbox head on unselected memory after the original forward. That is an algebraic diagnostic, and those tokens were not trained under identical selected-token matching.','Shape-dependent dense GEMM differences were bounded by frozen2e-6; exact native selected boxes overwrite their pool rows. Raw predictions remain preserved.','Full valid candidate pool contains8084 of8400 input640 tokens. TopK uses original all-token scores; invalid anchors excluded only from geometric pool.','Every image uses all GT in maximum matching; tiny-first augmentation maximizes tiny and global cardinality, independently certified. Other size bins are not lexicographic maxima.','The original augmented/train-mode and ordinary val_eval cohorts differ. Only ordinary val_train/eval share exact inputs/targets.','GT-assisted geometry/class matching ignores score-ranked AP and is not a deployable selector. No AP/causal root/novelty claim.','Scientific capture/primary/gate frozen prospectively; analysis matching clarification and later verification instrumentation timings remain explicitly disclosed.','No historical training gradients, actual CUDA AMP/optimizer recorder integration, or complete deterministic training replication verified by this run.'],'new_AP_result':False,'no_accepted_new_mechanism':True}
    save(OUT/'validation_report.json',validation)
    lines=['完整编码器候选池 vs TopK：结果与证据索引','',
      '结论：两组完整候选池比编码器选出的300个框，多覆盖4.56和6.08个百分点。均未达到事先冻结的10个百分点优先级门槛。本轮不晋级TopK改造训练。结果降低了“只改选框就能解决大部分问题”的优先级，但没有证明TopK完全无影响，也没有确定骨干、编码器或解码器的根因。',
      '', '主比较：原生BASE132 EMA；普通验证图eval模式；固定32张，640输入；GT<=300图像中输入短边<4像素目标。每组329个GT。下表为IoU>=0.5最大一对一几何覆盖，使用全部GT竞争框，避免一框重复计多个目标。',
      'seed | 编码器TopK300 | 完整有效候选池 | 最终解码器300 | 完整池-TopK']
    for r in summary['primary']:
        n=r['GT_rows'];fmt=lambda k:f"{r[k]}/{n} ({100*r[k]/n:.2f}%)"
        lines.append(f"{r['seed']} | {fmt('selected_encoder_50_unique')} | {fmt('full_50_unique')} | {fmt('final_50_unique')} | +{r['full_vs_selected_encoder_unique50_gain_pp']:.2f}个百分点")
    lines+=['','更严格IoU>=0.75：seed1 TopK8/329、完整池11/329、最终20/329；seed2 TopK6/329、完整池6/329、最终20/329。',
      '附加类别argmax正确条件，IoU>=0.5：seed1 TopK72/329、完整池82/329、最终91/329；seed2 TopK62/329、完整池73/329、最终95/329。没有用分类分数阈值替换几何门槛。',
      '', '同图训练/推理状态对照及增强图（BASE132、同一小目标定义；这些不改变主比较）：',
      'seed | context | GT数 | TopK唯一覆盖 | 完整池唯一覆盖 | 最终唯一覆盖 | 完整池-TopK']
    for r in summary['strata']:
        if r.get('input_size_bin')=='lt4' and not r['dense_gt300'] and r['endpoint']==132:
            lines.append(f"{r['seed']} | {r['context']} | {r['GT_rows']} | {r['selected_encoder_50_unique']} | {r['full_50_unique']} | {r['final_50_unique']} | +{r['full_vs_selected_encoder_unique50_gain_pp']:.2f}个百分点")
    lines+=['','同图train/eval的输出存在差异。增强与普通图的目标分布也不同，因此不能把此前增强训练态现象直接当成普通推理阶段的普遍问题。',
      '两组最终解码器几何覆盖均高于完整编码器池：119 vs107、121 vs102。这保留了“后续层能修正部分早期定位”的证据；不同候选池最大匹配不共享固定目标归属，不能据此证明同一个query的因果修复。',
      '', '完成与核验：',
      '两组40批/46次模型前向，共80批/92次；另外共80次已有编码器bbox头的密集诊断调用。0次模型反向、0次优化器更新、新增训练0轮，无重试。',
      '两端采集和分析正常exit0，原生468个源码文件/权重不变，12个普通前向控制的输出、私有BN和RNG一致，增强图与前次冻结采集的输出一致。',
      '两组采集压缩包全部下载并核验330个文件；本地重新计算35760次GT暴露、320次图像暴露、所有数组SHA和选择关系。重复暴露不是独立样本数。',
      '3840张邻接图的字节哈希和匹配数与远端SciPy独立最大匹配完全一致；每张图检查所有GT与极小目标子图两种最大值，共7680次。逐GT/逐图CSV、主层及全部分层结果也完全一致。',
      '当前两卡各2MiB、0%，无计算任务；原采集与分析PID均不存在。',
      '', '实施记录：协议/主层/10pp门槛在采集前冻结；tiny-first匹配细节在首次派生结果前单独冻结；本地图哈希验证包装器在远端统计后加入，未改变匹配、主层或门槛。local/remote时钟不同，冻结顺序依部署协议SHA与操作顺序核验。',
      '传输方面：现有SSH密钥对当前端口无法认证。seed1用密码SCP成功；seed2沿已有认证SSH分块传输。传输不重跑模型。密钥认证失败属于传输问题；本轮模型采集没有失败。',
      '', '边界与下一步：',
      '这不是AP提升实验。完整池oracle借用了GT，未选中token没有与选中token相同的训练匹配，不能承诺把全池纳入推理就会改善结果。当前仍未找到经训练验证有效的新机制。',
      '下一步检查候选框生成与逐层定位坐标约束，区分回归可达范围和表征/优化不足后再确定干预。旧ROI、重分类、范围加上下文及已中断路线保持原结论。本轮未启动新训练。之后每个新配置仍总计<=12轮，固定第12轮同seed BASE比较，APs提升且整体AP不下降才算筛选成立。',
      '', '本地证据（完整路径）：']
    for name in ['protocol.json','matching_clarification.json','analysis_execution.json','decision.json','validation_report.json','local_analysis_verification.json','download_verification_seed1.json','download_verification_seed2.json','reference_download_verification.json','final_live_completion_check.json','local_analysis/summary.json','local_analysis/per_GT_pool_coverage.csv','local_analysis/per_image_unique_coverage.csv','local_analysis/matching_certificates.json','collection_seed1.tar.gz','collection_seed2.tar.gz']:
        lines.append(str(OUT/name))
    lines+=['','实现：'+str(CODE),'','11/11统计谬误扫描完成，详见validation_report.json。当前科学范围ANALYZED；原生输出、数组、图及统计一致性的工程核验VERIFIED，未宣称完整训练复现或根因成立。']
    (OUT/'结论与证据索引.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    delivery={'status':'COMPLETE_DIAGNOSTIC_VERIFIED_NO_SELECTOR_TRAINING_PROMOTED','observed_local_utc':now,'goal_status':'active','human_report':str(OUT/'结论与证据索引.txt'),'primary':summary['primary'],'code_sha256':{p.name:sha(p) for p in sorted(CODE.glob('*.py'))},'evidence_sha256':{n:sha(OUT/n) for n in ['protocol.json','matching_clarification.json','analysis_execution.json','decision.json','validation_report.json','local_analysis_verification.json','download_verification_seed1.json','download_verification_seed2.json','reference_download_verification.json','final_live_completion_check.json','local_analysis/summary.json','local_analysis/per_GT_pool_coverage.csv','local_analysis/per_image_unique_coverage.csv','local_analysis/matching_certificates.json','结论与证据索引.txt']},'training_epochs_added':0,'AP_evidence':False,'root_cause_established':False,'accepted_new_mechanism':False}
    save(OUT/'delivery.json',delivery)
    statepath=ROOT/'outputs/sod_roi12_20261009/execution_state.json';state=read(statepath)
    state['previous_goal_turn_classification']=state.get('last_goal_turn_classification')
    state['last_goal_turn_classification']='PROGRESS: both frozen full encoder pool captures and independent SciPy analyses completed exit0; full archives/330 files, all3840 graph hashes+7680 optimality checks and all CSV/statistics verified locally. Both primary gains4.559/6.079pp fail frozen10pp selector-priority gate. No selector training/AP/root cause claimed.'
    state['status']='PROBLEM_DRIVEN_RESEARCH_ACTIVE_FULL_ENCODER_POOL_DIAGNOSTIC_VERIFIED'
    state['encoder_pool_diagnostic']={'status':delivery['status'],'delivery':str(OUT/'delivery.json'),'human_report':str(OUT/'结论与证据索引.txt'),'protocol':str(OUT/'protocol.json'),'primary':summary['primary'],'diagnostic_priority_gate_both_seed_ge10pp':False,'epochs_added':0,'AP_evidence':False,'root_cause_established':False}
    state['next']=next_work;state['updated_utc']=now
    state.setdefault('research_progress_history',[]).append({'utc':now,'kind':'encoder_pool_vs_native_TopK','delivery_sha256':sha(OUT/'delivery.json'),'priority_gate_passed':False,'epochs_added':0})
    save(statepath,state)
    print(json.dumps({'status':delivery['status'],'report':delivery['human_report'],'goal_status':'active','primary_gains_pp':[r['full_vs_selected_encoder_unique50_gain_pp'] for r in summary['primary']],'epochs_added':0},ensure_ascii=False))

if __name__=='__main__':main()
