from pathlib import Path
import hashlib,json
ROOT=Path('outputs');REF=ROOT/'native_stem_scratch12_20261011_v3';MAIN=ROOT/'stem_component12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
meta=read(REF/'compact_transport_v2.json');rows=[]
for seed in (1,2):
    p=REF/f'compact_payload_seed{seed}_v2.json';raw=p.read_bytes()
    assert len(raw)==meta[str(seed)]['expected_bytes'] and sha(p)==meta[str(seed)]['expected_sha256']
    o=json.loads(raw)
    assert o['code_sha256_verified'] and not o['failures']
    assert all(p['same_live'] and p['state']!='T' for p in o['processes'].values())
    rows.append(o)
write(REF/'compact_observation_v2.json',rows)
ref=read(REF/'execution_state_v1.json')
ref.update(supersedes='execution_state_v1.json',evidence='compact_observation_v2.json',
    completed_epochs={str(x['seed']):x['completed_epochs'] for x in rows},
    optimizer_updates={str(x['seed']):x['runtime']['training_audit_live.json']['actual_optimizer_steps'] for x in rows},
    latest_live_processes={str(x['seed']):x['processes'] for x in rows},
    transport_verification='compact_transport_v2.json SHA256 and exact byte length matched')
write(REF/'execution_state_v2.json',ref)
joint=ROOT/'stem_joint_error_analysis_20261011_v2';report=read(joint/'report.json')
for n,h in read(joint/'manifest.json')['files'].items():assert sha(joint/n)==h
main=read(MAIN/'execution_state_v11.json');snapshot=read(MAIN/'training_observation_v17.json')
main.update(supersedes='execution_state_v11.json',evidence='training_observation_v17.json',baseline_comparison='progress_v17.json',
    reference_state='../native_stem_scratch12_20261011_v3/execution_state_v2.json',
    joint_existing_BASE12_diagnosis='../stem_joint_error_analysis_20261011_v2/report.json',
    previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    training_epochs_added_to_pretrained_BASE=0,
    finding='Existing BASE12 GT-ID join separates absent support at all three recorded boundaries from later geometry loss and final class-score/selection losses. Small-input cohorts differ; no unique Stem cause or new Stem efficacy established.',
    next='Collect independently replay-verified fixed12 endpoints for full, random-native reference and all controls; compare actual AP/APs and GT rescue/damage. No new method launch based only on observational counts.')
for seed in (1,2):
    main['current_live_trainers']['seed'+str(seed)]={a:snapshot['seed'+str(seed)]['jobs'][a]['processes'] for a in ('original_stem','detail_lowres','highfreq_rgb')}
    assert all(p['same_live'] and p['state']!='T' for ps in main['current_live_trainers']['seed'+str(seed)].values() for p in ps)
write(MAIN/'execution_state_v12.json',main)
full=read(ROOT/'original_stem12_20261011_v1/execution_state_v14.json')
full.update(supersedes='execution_state_v14.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v12.json',reference_state='../native_stem_scratch12_20261011_v3/execution_state_v2.json')
write(ROOT/'original_stem12_20261011_v1/execution_state_v15.json',full)
write(joint/'execution_state_v1.json',dict(status=report['status'],goal_complete=False,current_goal_turn='progress',
    no_progress_streak=0,evidence='report.json',code='英文小论文/code/stem_joint_error_analysis_20261011_v2/analyze.py',
    actual_new_Stem_fixed12_packets_analyzed=0,new_model_forwards=0,new_training_epochs=0,
    root_cause_established=False,statistical_fallacies_checked=11,
    next=main['next']))
lines=['本轮已有 BASE12 错误联合核对（非新增 Stem 终点）','Material Passport: academic-research-suite / experiment-agent validate; ANALYZED。',
    '已有保存数组、manifest 和 GT 身份连接已核实。未新增模型前向、评价器或训练。两种子完整保留，统计解释检查11/11。',
    '三个边界为已选择300候选的初始参考框、Decoder第0层pre框、最终query框；并未观察全部8400候选、Stem特征或训练匹配历史。',
    'IoU=0.5，原图面积<=1024的小目标总数各26586：']
for item in report['seeds']:
    g=item['groups']['0.5']['COCO_small'];c=g['counts']
    tiny=item['groups']['0.5']['COCO_small_input_under4']
    lines.append(f"种子{item['seed']}：原生命中{c['native_TP']}，漏检{g['native_missed']}；三边界均无覆盖{c['miss_never_supported_at_three_boundaries']}，较早覆盖但最终丢失{c['miss_earlier_supported_lost_at_final']}，最终有框但真实类未通过top300/分数筛选{c['miss_final_supported_no_true_class_after_top300_score_filter']}。")
    lines.append(f"输入短边<4像素：GT={tiny['GT']}，漏检={tiny['native_missed']}，三边界均无覆盖={tiny['counts']['miss_never_supported_at_three_boundaries']}，占漏检{100*tiny['fraction_of_missed']['miss_never_supported_at_three_boundaries']:.2f}%。")
lines.extend(['这些是观测错误构成，不是AP分解，也不能证明加高频/注意力能改善覆盖。','当前增强Stem以S/B0基线构造，只替换backbone.stem；其余754个状态张量来自同一初始化。原基线Stem有预训练，增强Stem随机；随机原始Stem已并发训练来区分初始化条件下的结构效果。','旧n版提升配方还改变了其他条件，不可作为当前S/B0 Stem-only收益证明。','实际新Stem机制比较仍待固定12轮EMA及独立权重回放，保留原预训练BASE门槛。'])
with (joint/'本轮错误核对.txt').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
print('PROGRESS_SAVED_JOINT_DIAGNOSIS_AND_ALL_EIGHT_LIVE_TRAINERS')
