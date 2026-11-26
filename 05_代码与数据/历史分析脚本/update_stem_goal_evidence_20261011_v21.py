from pathlib import Path
import argparse,base64,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--input-base64',required=True);a=p.parse_args();b=json.loads(base64.b64decode(a.input_base64))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def check(i,nsha='sha256',nbytes='bytes'):
    raw=i['payload'].encode();m=i['transport'];assert len(raw)==m[nbytes] and hashlib.sha256(raw).hexdigest()==m[nsha];return raw,json.loads(raw)
ref=Path('outputs/native_stem_scratch12_20261011_v3');comp=Path('outputs/stem_component12_20261011_v1');queue=Path('outputs/stem_component_queue_pulse_20261011_v2');warm=Path('outputs/warm_stem12_20261011_v1')
refs={};qs={};ws={}
for seed in (1,2):
    key=str(seed)
    raw,o=check(b['reference'][key],'expected_sha256','expected_bytes');assert o['code_sha256_verified'] and not o['failures'] and o['completed_epochs']<=12
    with (ref/f'compact_payload_seed{seed}_v13.json').open('xb') as f:f.write(raw)
    refs[key]=o
    raw,o=check(b['queue'][key]);assert o['code_SHA_verified'] and o['component_SHA_verified'] and not o['failure'] and o['completion'] and not o['helper']['same_live']
    with (queue/f'remote_observation_seed{seed}_v5.json').open('xb') as f:f.write(raw)
    qs[key]=o
    raw,o=check(b['warm'][key]);assert o['source_SHA_verified'] and o['dispatcher']['same_live'] and not o['failure']
    with (warm/f'remote_compact_observation_seed{seed}_v2.json').open('xb') as f:f.write(raw)
    ws[key]=o
write(ref/'compact_observation_v13.json',refs);write(ref/'compact_transport_v13.json',{s:i['transport'] for s,i in b['reference'].items()})
write(queue/'remote_transport_v5.json',{s:i['transport'] for s,i in b['queue'].items()});write(warm/'compact_transport_v2.json',{s:i['transport'] for s,i in b['warm'].items()})
progress=read(comp/'progress_v27.json');pairs=[]
for seed in (1,2):
    r=refs[str(seed)];row=next(x for x in progress['rows'] if x['seed']==seed and x['arm']=='original_stem' and x['epoch']==r['completed_epochs']);v=r['latest_metrics']['test_coco_eval_bbox']
    pairs.append(dict(seed=seed,epoch=r['completed_epochs'],enhanced_AP=row['AP'],random_native_AP=v[0]*100,delta_AP_pp=row['AP']-v[0]*100,
        enhanced_APs=row['APs'],random_native_APs=v[3]*100,delta_APs_pp=row['APs']-v[3]*100))
write(comp/'same_epoch_random_native_comparison_v9.json',dict(status='PROVISIONAL_SAME_EPOCH_BOTH_SEEDS_9_AND11',current_S_pairs=pairs,
    full_progress_sha256=sha(comp/'progress_v27.json'),random_native_observation_sha256=sha(ref/'compact_observation_v13.json'),
    all_seeds_preserved=True,fixed12_efficacy_established=False,root_cause_established=False,new_model_forwards=0,new_optimizer_updates=0))
rs=read(ref/'execution_state_v8.json');rs.update(supersedes='execution_state_v8.json',evidence='compact_observation_v13.json',
    completed_epochs={s:o['completed_epochs'] for s,o in refs.items()},latest_live_processes={s:o['processes'] for s,o in refs.items()},
    optimizer_updates={s:o['runtime']['training_audit_live.json']['actual_optimizer_steps'] for s,o in refs.items()},current_goal_turn='progress',no_progress_streak=0)
write(ref/'execution_state_v9.json',rs)
write(queue/'execution_state_v5.json',dict(status='BOTH_HELPERS_COMPLETED_TWO_UNCHANGED_NATIVE_DISPATCHES',supersedes='execution_state_v4.json',
    evidence='remote_observation_seed1/2_v5.json',transport='remote_transport_v5.json',helpers={s:o['helper'] for s,o in qs.items()},
    completions={s:o['completion'] for s,o in qs.items()},conductors={s:o['conductor'] for s,o in qs.items()},
    existing_training_code_changed=False,existing_process_signals_this_turn=0,automatic_retry_or_extension=False,goal_complete=False))
hs=read('outputs/stem_highfreq_primary_analysis12_20261011_v1/report.json')
assert hs['status']=='ANALYZED_BOTH_SEEDS_FULL_VS_RGB_PRIMARY12_SAME_GT_REPLAY_PENDING'
for n,h in read('outputs/stem_highfreq_primary_analysis12_20261011_v1/manifest.json')['files'].items():
    assert sha(Path('outputs/stem_highfreq_primary_analysis12_20261011_v1')/n)==h,n
warmstate=read(warm/'execution_state_v1.json');warmstate.update(supersedes='execution_state_v1.json',evidence='remote_compact_observation_seed1/2_v2.json',
    dispatchers={s:o['dispatcher'] for s,o in ws.items()},phases={s:o['live']['phase'] for s,o in ws.items()},
    actual_new_training_epochs={s:o['completed_epochs'] for s,o in ws.items()},full_detector_preflight_passed=all(o['detector_preflight'] is not None for o in ws.values()))
write(warm/'execution_state_v2.json',warmstate)
state=dict(status='ACTIVE_STEM_ATTRIBUTION_AND_DISTINCT_INITIALIZATION_PRESERVING_CANDIDATE',supersedes='execution_state_v20.json',
    previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    reference_state='../native_stem_scratch12_20261011_v3/execution_state_v9.json',component_last_complete_snapshot='training_observation_v27.json',
    component_fixed12_numeric_comparison='progress_v27.json',random_same_epoch_comparison='same_epoch_random_native_comparison_v9.json',
    completed_fixed12_arms_both_seeds=['original_stem','detail_lowres','highfreq_rgb'],
    pending_old_training_arms_both_seeds=['attention_spatial_mean','early_final_downsample','native_stem_scratch'],
    independent_saved_EMA_replays_and_formal_terminal_readers_pending=True,
    highfreq_same_GT_primary_analysis='../stem_highfreq_primary_analysis12_20261011_v1/report.json',
    highfreq_full_minus_RGB=[dict(seed=r['seed'],delta=r['full_minus_RGB_pp'],
        small50_initial_support_delta=r['outcomes']['0.5']['COCO_small']['supported_GT_by_boundary']['encoder_initial_boxes']['delta'],
        small50_native_TP_net=r['outcomes']['0.5']['COCO_small']['native_TP_net']) for r in hs['seeds']],
    warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v2.json',
    frozen_existing_training_code_changed=False,new_BASE_training_epochs=0,new_BASE_detector_forwards=0,existing_process_signals_this_turn=0,
    interpretation='Both seeds highfrequency residual input improves APs over RGB-input control, with observed differences at selected300 initial regressed boxes. AP seed1 slightly worse. Direct enhanced Stem remains below the pretrained native BASE12. Root cause, frequency-specific mediation and novelty unproven. Warm route retains794 original states and adds the complete user enhanced residual with16 zero-init gain parameters; bothCPU and standalone CUDA AMP checks passed, full detector preflight and training still resource-gated.',
    pending='Complete all authorized fixed12 endpoints, native-owned saved EMA closeouts, dense-score/selection diagnostics and full-versus-random/component attribution. Verify the warm candidate complete detector and run each seed at most12 epochs before judging cached BASE12 gates.')
write(comp/'execution_state_v21.json',state)
write('outputs/original_stem12_20261011_v1/execution_state_v24.json',dict(status=state['status'],supersedes='execution_state_v23.json',
    authoritative_state='../stem_component12_20261011_v1/execution_state_v21.json',reference_state=state['reference_state'],goal_complete=False,frozen_code_changed=False))
review=Path('outputs/stem_review_20261011_v2');review.mkdir(exist_ok=False)
lines=['当前方向与结果（2026-10-11；原始日志/产物已校验；独立EMA回放待完成）','',
    '1. 同种子、同轮次：你的完整增强Stem vs 仅Stem随机初始化的原始结构。其他754个状态共享。']
for r in pairs:lines.append(f"种子{r['seed']} 第{r['epoch']}轮：AP {r['random_native_AP']:.3f}→{r['enhanced_AP']:.3f}（{r['delta_AP_pp']:+.3f}点）；APs {r['random_native_APs']:.3f}→{r['enhanced_APs']:.3f}（{r['delta_APs_pp']:+.3f}点）。尚未用两组固定第12轮作最终结论。")
lines+=['','2. 分支消融：完整增强Stem减去对应对照，固定第12轮。']
for seed in (1,2):
    for arm,label in [('detail_lowres','细节分支保持高分辨率'),('highfreq_rgb','高频残差输入而非RGB输入')]:
        r=next(x for x in progress['rows'] if x['seed']==seed and x['arm']==arm and x['epoch']==12)
        lines.append(f"种子{seed} {label}：AP {r['full_minus_control_AP_pp']:+.3f}点，APs {r['full_minus_control_APs_pp']:+.3f}点。")
lines+=['注意力/下采样时机两组消融继续原有12轮实验。细节分支收益仍有种子差异；高频输入的APs收益两种子均为正，总体AP种子1略降。','',
    '3. 高频分支实际阶段证据：两组RGB对照已下载并校验607个文件，包括last.pth、原始EMA验证结果、548张图的三个边界产物。']
for r in hs['seeds']:
    q=r['outcomes']['0.5']['COCO_small'];s=q['supported_GT_by_boundary']
    lines.append(f"种子{r['seed']}：在26,586个COCO小目标中，IoU≥0.5的初始框支持 {s['encoder_initial_boxes']['RGB']}→{s['encoder_initial_boxes']['full']}；最终框支持 {s['final_boxes']['RGB']}→{s['final_boxes']['full']}；原生全尺寸评估TP标签筛选小目标后，TP净增{q['native_TP_net']}（新增{q['native_TP_gained']}，损失{q['native_TP_lost']}）。")
lines+=['变化最早在已记录的“TopK选中300个查询的初始回归框”出现。这个环节之前还有主干、encoder、分类排序和初始框回归，尚未识别其中的独立原因。GT支持不是查询所有权，TP数量不是APs分解；RGB和高通输入也同时改变输入统计量，不能声称已单独证明高频本质。','',
    '4. 新候选代码：保留原始预训练Stem，并并联完整的用户EnhancedStem，16个通道增益从0开始。所有794个原始状态完整保留，新加11815个参数（约0.1155%）。',
    '两台服务器均通过CPU原始状态、零增益输出/原路径梯度/BN更新和残差梯度检查；通过batch4、640/832的单独Stem CUDA FP16检查。完整检测器训练代码已冻结、两台7019张图已验证、实际调度进程已启动且确认为存活。',
    '当前新候选仍在等完整检测器预检资源，新增训练0轮。三组旧任务与候选的预留显存预算27GiB，超过24GiB卡；旧任务继续运行。一个旧训练结束后会自动执行完整检测器与原生损失检查，通过后每种子只启动一次12轮训练。',
    '该方案同时改变并联结构/容量和初始化保留，不能视作纯初始化因果试验，也没有创新性或AP有效性结论。每轮记录训练/EMA残差增益，并保留固定12轮的阶段产物供进一步归因。','',
    '验收标准保持：两种子固定第12轮EMA的APs严格超过已有同种子BASE12，总体AP不下降；不重训BASE，不挑最好轮，不延长或重跑失败配置。']
(review/'本轮结果和方向.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
write(review/'evidence.json',dict(main_state_sha256=sha(comp/'execution_state_v21.json'),same_epoch_comparison_sha256=sha(comp/'same_epoch_random_native_comparison_v9.json'),
    highfreq_report_sha256=sha('outputs/stem_highfreq_primary_analysis12_20261011_v1/report.json'),warm_state_sha256=sha(warm/'execution_state_v2.json')))
print(json.dumps(dict(status='UPDATED_ALL_STATES_AND_ACTUAL_HIGHFREQ_REVIEW',same_epoch_pairs=pairs,goal_complete=False)))
