"""Record a same-epoch intermediate comparison and the completed CPU-only secondary check."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
obs_path=ROOT/'observer_39_M_epoch4.json'
raw=read(obs_path);assert raw['status']=='EXECUTED' and raw['exit']==0
o=json.loads(raw['stdout']);m=o['arms']['M'];mi=m['identity']
assert mi['present'] and mi['same_start_ticks'] and mi['owned_command'] and mi['state'] not in ['Z','X']
assert m['completed_epochs']>=4 and m['reported_loss_grad_finite']
checks=o['common_epoch_checks']['M_vs_B']
assert len(checks)==4 and all(r['all_pass'] for r in checks)
b4=o['arms']['B']['evaluations']['4'];m4=m['evaluations']['4']
delta={k:m4[k]-b4[k] for k in ['AP','APs','APs75']}
secondary=read(ROOT/'postprocess_ignore_strata'/'arm_B'/'result.json')
assert secondary['status']=='COMPLETED' and secondary['COCO_evaluate_attempts']==3
assert secondary['model_forwards']==secondary['optimizer_updates']==0
full=secondary['groups']['full548']
assert full['cached_precision_max_abs_difference']==full['cached_recall_max_abs_difference']==0
plan=read(ROOT/'postprocess_ignore_strata'/'protocol_FIXED_v1.json')
assert secondary['protocol_sha256']==sha(ROOT/'postprocess_ignore_strata'/'protocol_FIXED_v1.json')
lines=['A49 第4轮普通控制对照与已有预测重算','',
       '主线仍未确认，目标active。以下为同轮中间观察，不是固定12轮裁决。','',
       '第4轮EMA：','组别\tAP(%)\tAPs(%)\tAPs75(%)']
for a,row in [('B',b4),('M',m4)]:
    lines.append('%s\t%.8f\t%.8f\t%.8f'%(a,row['AP'],row['APs'],row['APs75']))
lines+=['M−B（百分点）：AP %+.8f，APs %+.8f，APs75 %+.8f。'%(delta['AP'],delta['APs'],delta['APs75']),
        'M的普通忽略区规则目前有轻微正向AP/APs观察；单seed早期结果不支持显著性或论文创新结论。',
        'Q尚未开始。不得把B第12轮与M第4轮混比；三组第12轮门槛保持原样。',
        'M前4轮的顺序、前8批实际输入、前8批前向前RNG及全轮含ignore目标流SHA逐轮一致。','',
        '附加分析已在读取M评估结果前锁定，B终点当时已知；不是所有数据之前的预注册。',
        '本地已存B12预测的548图COCO指标重算通过：precision/recall数组与训练端缓存最大差均为0。',
        '这里只重算预测指标，没有重新执行检测器；共3次CPU COCO evaluate，新增模型前向/优化更新均0。',
        'B12分组描述：','组别\t图像\t有效GT\t小目标GT\tAPs(%)']
for name in ['with_explicit_ignore','without_explicit_ignore']:
    row=secondary['groups'][name]
    lines.append('%s\t%d\t%d\t%d\t%.8f'%(name,row['images'],row['evaluated_GT'],row['evaluated_small_GT_area_le1024'],row['metrics_percent']['APs']))
lines+=['组间图像密度与类别组成不同，上述组间差异不证明忽略区是漏检原因。后续只比较同组内B/M/Q的固定12轮差值。',
        '该描述性分析不能推翻失败的主门槛，不能用于挑子集保留候选。','',
        '当前M已进入第5轮，Q待原监督器接续。当前身份和观察路径见14回执及CURRENT_A49.json。',
        '统一指标为COCO式、maxDets300、原图GT面积及class0并集过滤，非官方VisDrone成绩。']
note=ROOT/'13_M4与预测重算核查.txt'
with note.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
files=[note,obs_path,ROOT/'postprocess_ignore_strata'/'protocol_FIXED_v1.json',
       ROOT/'postprocess_ignore_strata'/'arm_B'/'result.json',ROOT/'postprocess_ignore_strata'/'dependency_acquisition_v1.json',
       ROOT/'recalculate_fixed12_ignore_strata_A49.py',ROOT/'freeze_saved_prediction_analysis_A49.py',
       ROOT/'acquire_cpu_eval_dependency_A49.py',Path(__file__)]
c=o['observed_counters_lower_bounds']
totals={'model_forwards':114452+c['model_forwards'],'formal_optimizer_updates':87898+c['formal_optimizer_updates'],
        'capacity_updates':28+c['capacity_optimizer_updates'],
        'optimizer_updates_including_capacity':87926+c['formal_optimizer_updates']+c['capacity_optimizer_updates'],
        'formal_epochs':108+c['formal_epoch_sum']}
record={'status':'PASS','classification':'progress_and_verified_wait','previous_goal_turn_classification':'progress_and_verified_wait',
        'goal_status':'active','mainline_confirmed':False,'same_epoch4_metrics':{'B':b4,'M':m4},'M_minus_B_pp':delta,
        'M_four_epoch_input_flow_checks':checks,'owned_live_M_identity':mi,'M_progress':m['progress'],
        'previous_B_milestone_receipt':str(ROOT/'12_B终点与M启动校验回执.json'),
        'previous_B_milestone_receipt_sha256':sha(ROOT/'12_B终点与M启动校验回执.json'),
        'files_sha256':{str(p):sha(p) for p in files},'cumulative_counts_observed_lower_bounds':totals,
        'new_CPU_saved_prediction_COCO_evaluate_calls':3,'new_model_forwards_from_secondary_analysis':0,
        'new_optimizer_updates_from_secondary_analysis':0,'new_training_queued':False,
        'next_action':'Continue same M PID4991/start20463090 to M8/M12. Then collect completed exit0 M; supervisor alone sequences Q12. Do not retune or rescue using secondary subsets.'}
with (ROOT/'14_M4与辅助验证回执.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'PASS','M_minus_B_epoch4_pp':delta,'CPU_COCO_calls':3,'M_current_epoch':m['progress']['epoch'],
                  'cumulative_lower_bounds':totals,'goal_status':'active','mainline_confirmed':False},ensure_ascii=False))
