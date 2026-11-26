"""Update local research status from verified fixed-epoch completion records only."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, shutil

local = Path(__file__).resolve().parent
base = local.parent
folder = local / '完成结果_20261029_v1'
raw = folder / 'raw'
audit = json.loads((folder / 'baseline_integrity_and_curve_audit.json').read_text())
backup = json.loads((folder / 'local_backup_verification.json').read_text(encoding='utf-8'))
result = json.loads((raw / 'evaluation/initial_diagnostic_result.json').read_text())
assert backup['status'] == 'ARCHIVE_AND_BOTH_CHECKPOINT_HASHES_VERIFIED_LOCALLY'
now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-7))).isoformat()
report_name = '16_两组基线完成与P1初步结果_20261029.txt'
history = folder / 'historical_status_before_completion_update'
history.mkdir(exist_ok=True)
names = [base / '00_调研报告.txt', base / '15_远端实验_运行状态与冻结方案_20261029.txt', local / 'execution_ledger.json']
for source in names:
    dest = history / source.name
    if not dest.exists():
        shutil.copyfile(source, dest)

ledger = json.loads((local / 'execution_ledger.json').read_text(encoding='utf-8'))
ledger.update(status='COMPLETED_BASELINES_AND_INITIAL_DIAGNOSTIC', snapshot_time=now,
    formal_D0_observed=audit['arms']['D0']['completed'], formal_DA_observed=audit['arms']['DA']['completed'],
    formal_DA_started=True, scientific_effect_evaluated=True, mainline_validated=False,
    primary_checkpoint_rule='fixed epoch100 EMA, no best selection',
    local_backup=backup, queue_elapsed_seconds=audit['queue_elapsed_seconds'],
    P1_remaining_DA_small_AP_loss_pp=result['DA']['remaining_small_AP_loss_pp'],
    P1_image_bootstrap95_pp=result['DA']['gap_paired_image_bootstrap95_pp'],
    P2_region_joint_necessity_validated=False, P3_deployable_selection_validated=False,
    supplementary_cap_sensitivity_completed=False, codec_controls_completed=False,
    frozen_restoration_forward_completed=False, active_training_processes_observed=0,
    historical_training_completion_pending_eval_flag='Training completed.json was written before the subsequent diagnostic; do not treat its historical pending flag as latest state.')
(local / 'execution_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
'## Material Passport', '',
'- Origin Skill: academic-research-suite / experiment-agent',
'- Origin Mode: run（已完成训练的结果收集及完整性审计）',
'- Origin Date: 2026-10-29；America/Los_Angeles',
'- Verification Status: 训练计数、冻结文件、归档和权重SHA已核验；检测结果为单seed诊断，尚未独立复现',
'- Version Label: completed_baselines_P1_v1', '',
'两组基线已完成；当前只能确认P1的初步剩余退化', '',
f'客户端整理时间：{now}。训练耗时采用进程单调时钟，服务器Unix时间只作原始技术记录。', '',
'一、明确回答：D0已跑完，DA也已跑完', '',
'组别 | 完成轮数 | 训练耗时 | 微批次 | 真实优化器更新 | AMP跳步',
]
for arm in ['D0', 'DA']:
    c = audit['arms'][arm]['completed']
    lines.append(f"{arm} | {c['epoch']}/100 | {c['elapsed_seconds']/3600:.2f}小时 | {c['batches']} | {c['actual_optimizer_updates']} | {c['amp_skips']}")
lines += [
'', f"整队列含初始诊断共{audit['queue_elapsed_seconds']/3600:.2f}小时。原训练队列正常结束；最近查询GPU空闲，未自动追加训练。",
'两组各有100行逐轮审计，CSV数值有限；实际正式初始化相同，前8批几何合同匹配。七个冻结/修正源码及正式协议SHA一致。',
'固定第100轮EMA为比较对象，未用best或早期轮次替代；没有新策略网络、额外seed或延长训练。', '',
'二、固定诊断100图的实际结果', '',
'D0是source reference训练；DA是相同配方追加随机JPEG增强。source本身也是JPEG，不是RAW或无损真值。',
'以下小目标AP指640检测输入坐标中GT面积严格<32²的AP50:95，maxDet500；与标准原图面积AP_s分别报告。',
'采用内部COCO式crowd忽略和one2many+NMS，不是官方VisDrone挑战分数。', '',
'组别 | source小目标AP(%) | Q20小目标AP(%) | source−Q20损失(pp) | 成对图像bootstrap95%区间(pp)',
]
for arm in ['D0', 'DA']:
    r = result[arm]
    ci = r['gap_paired_image_bootstrap95_pp']
    lines.append(f"{arm} | {100*r['source']['AP_small_input_500']:.4f} | {100*r['q20']['AP_small_input_500']:.4f} | {r['remaining_small_AP_loss_pp']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}]")
gain = 100 * (result['DA']['q20']['AP_small_input_500'] - result['D0']['q20']['AP_small_input_500'])
reduction = result['D0']['remaining_small_AP_loss_pp'] - result['DA']['remaining_small_AP_loss_pp']
lines += [
'', f'【计算】DA相对D0的Q20小目标AP提高{gain:.4f}pp；压缩前后差距缩小{reduction:.4f}pp。',
f"【观测】DA仍损失{result['DA']['remaining_small_AP_loss_pp']:.4f}pp，点估计略超过预先规划的1pp筛选信号。",
'【限制】其区间下端低于1pp，不能说“以95%把握损失至少1pp”。这是固定单seed/单checkpoint条件下的图像抽样不确定性，不覆盖训练随机性；相关视频帧还可能使区间过窄。', '',
'总体AP、AP50及原图小目标AP如下，仍为内部口径：',
'组别/条件 | AP50:95(%) | AP50(%) | 原图面积AP_s(%)',
]
for arm in ['D0', 'DA']:
    for cond in ['source', 'q20']:
        r = result[arm][cond]
        lines.append(f"{arm}/{cond} | {100*r['AP500']:.4f} | {100*r['AP50_500']:.4f} | {100*r['AP_small_original_500']:.4f}")
lines += ['', '三、漏检并非把全部困难目标都归因于压缩', '',
'诊断100图有4491个有效GT；其中4129个属于输入面积小目标（3057个<16²，1072个介于16²和32²）。',
'固定confidence=0.25、IoU=0.5下：',
'组别/条件 | 小目标TP | 小目标FN | 全部FP/图',
]
for arm in ['D0', 'DA']:
    for cond in ['source', 'q20']:
        s = next(m for m in result[arm][cond]['matches'] if m['confidence'] == .25 and m['iou'] == .5)
        lines.append(f"{arm}/{cond} | {s['small_TP']} | {s['small_FN']} | {s['FP_per_image']:.2f}")
lines += [
'', 'DA中source命中而Q20漏掉的同一小目标222个，Q20反而新命中113个，净少109个；不能将Q20全部1846个FN都称作压缩新增漏检。',
'DA在预定confidence=0.05/0.1/0.25/0.5及IoU=0.5/0.75下均有小目标命中减少；同置信度不等于严格同FP预算。', '',
'四、训练后期及评价上限复核', '',
'原生训练期dev300 source health mAP50:95均值（只用于训练过程健康检查，不能与上述诊断100科学口径混用）：',
'组别 | epoch61–80 | epoch81–90 | epoch91–100',
]
for arm in ['D0', 'DA']:
    groups = audit['arms'][arm]['reference_health_curve_groups']
    vals = [100*groups[g]['metrics/mAP50-95(B)'] for g in ['epoch61_80', 'epoch81_90', 'epoch91_100']]
    lines.append(arm + ' | ' + ' | '.join(f'{v:.4f}%' for v in vals))
lines += [
'', '【推断】source验证曲线后期处于平台并小幅回落，未见末段持续上升；这不证明达到全局最优。Q20没有逐轮科学验证，不能据此宣称压缩域已充分收敛。保持固定100轮，不事后挑更好轮次或自动延长。',
'【观测】四个条件各50/51/56/56图触及500框上限，截断边界最高置信度约0.0318；没有一图的第500框分数≥0.05。因此这些cap截断未删除用于已报告confidence≥0.05统计的候选，但低分PR尾部仍可能改变AP。',
'【待验证】maxDet1000敏感性尚未运行；maxDet500保留为预定主要口径，不以敏感性结果替换主结果。', '',
'五、目前主线成立到了哪里', '',
'【观测】普通JPEG增强能够改善Q20检测；适配后仍存在约1.09pp的输入面积小目标AP差距和可追踪的GT命中变化。',
'【推断】值得进行下一步低成本淘汰检查，但证据偏窄，不能由此直接跳到区域联合恢复。',
'【待验证假设P2】冻结恢复是否优于不恢复/全局强度混合，区域联合是否优于独立选择，以及收益是否对应真实TP/FN/FP/AP，而不是loss或拼接/分配伪影。',
'【待验证假设P3】仅使用部署可见输入的选择是否可学习且优于同容量简单模型。当前没有这方面实验结果。',
'没有获得已验证的算法增量，也不能据这100图单seed结果判断论文可发表或方法新颖。', '',
'六、接下来的最小工作（尚未执行）', '',
'1. 检查maxDet1000对source/Q20差距的敏感性，保留500主要结果。',
'2. 在调参100图仅按实际字节校准原尺寸WebP和缩小分辨率JPEG，再在同一诊断100图评价；不能按诊断AP选质量。编码/解码和额外计算成本与实际字节偏差一并记录。',
'3. 若简单codec/缩放在相近传输字节和成本下已解决主要问题，且没有真实JPEG兼容需求，则停止这条JPEG接收端算法路线。不能临时虚构保留JPEG约束。',
'4. 仅在简单解释未淘汰时，按11/12的冻结FBCNN全恢复、全局alpha和四区16子集判别推进P2；不在此之前训练新策略网络。',
'本次复核及本地备份没有追加GPU训练、恢复前向、codec对照或参数搜索。', '',
'七、可核验资产', '',
f'结果根目录：{folder}',
f'原始科学结果：{raw / "evaluation/initial_diagnostic_result.json"}',
f'完整性/曲线审计：{folder / "baseline_integrity_and_curve_audit.json"}',
f'本地SHA回执：{folder / "local_backup_verification.json"}',
f'完整结果归档：{folder / "completed_baseline_results.tar.gz"}',
f'归档字节数：{backup["archive_bytes"]}；SHA256：{backup["archive_sha256"]}',
]
for arm in ['D0', 'DA']:
    c = backup['checkpoints'][arm]
    lines += [f'{arm}权重：{c["local_checkpoint"]}', f'{arm} SHA256：{c["sha256"]}']
lines += [
'', '归档包含两个最终权重、训练日志/逐轮记录、正式配置、冻结代码、评估预测、科学指标和同一批Q20载荷；本地已安全展开并重算归档及权重SHA，均一致。',
'训练completed.json中evaluation_pending=true是训练结束当时写入的历史状态；随后独立科学评价已完成，保留原回执不篡改。最新状态看本文件、execution_ledger及evaluation/initial_diagnostic_result.json。',
'历史15及旧execution_ledger已备份，00/15入口现指向本文件。凭据不进入报告或实验资产。',
]
(base / report_name).write_text('\n'.join(lines) + '\n', encoding='utf-8')

prefix = (f'最新完成状态（客户端{now}）：D0与DA均已完成100/100轮，初始诊断也已完成，训练队列已结束。'
          f'两组最终权重已下载并逐一SHA核验。DA的Q20输入面积小目标AP为23.6204%，比source低1.0856pp；区域联合恢复尚未验证。'
          f'完整完成结果见{report_name}。\n下列启动/训练中状态保留为历史快照，不代表当前仍在训练。\n\n')
p = base / '15_远端实验_运行状态与冻结方案_20261029.txt'
old = p.read_text(encoding='utf-8-sig')
if not old.startswith('最新完成状态'):
    p.write_text(prefix + old, encoding='utf-8')
p = base / '00_调研报告.txt'
old = p.read_text(encoding='utf-8-sig')
if not old.startswith('两组基线完成入口'):
    p.write_text(f'两组基线完成入口（2026-10-29）：{report_name}。D0/DA均100轮完成，初始科学诊断与本地权重备份已完成；下一阶段codec/恢复/联合选择尚未执行。以下旧运行叙述为历史快照。\n\n' + old, encoding='utf-8')
print(json.dumps({'status':'COMPLETION_REPORT_AND_LEDGER_UPDATED', 'report':str(base / report_name), 'snapshot_time':now}))
