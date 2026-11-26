import csv, datetime, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'postprocessing_probe_cpu_reference/02_results.json'
a=json.loads(p.read_text(encoding='utf8'))
names=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
with (ROOT/'23a_三组12指标_CPU缓存口径.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['seed','arm',*names])
    for s in a['seeds']:
        for g,m in s['metrics'].items():w.writerow([s['seed'],g,*[100*x for x in m['stats']]])
lines=['Material Passport: academic-research-suite / experiment-agent; cached exploratory CPU evaluation; no detector training.',
       '两个种子的缓存后处理筛查已完成；当前每query仅保留最高类别的方案不继续。',
       '时间UTC '+datetime.datetime.now(datetime.timezone.utc).isoformat(),
       '对象：original_stem12 EMA第12轮，548张val。不是预训练BASE24、C96或UAVDT。',
       'N原展平top300；D原N内每query留最高类、不补位；R所有query最高类补至预算300。评分>.001，COCO maxDets1/10/100每图每类；原GT0 sentinel保留。',
       '旧原生重放门槛未通过的文件完整保留。CPU新协议在D/R结果前登记，N重复计算的12指标、逐类和匹配结果完全相等；不称原生GPU复现，不将两套协议混用。',
       'seed | N AP | D AP | R AP | R-N(pp) | R-D(pp) | R-N APs(pp)']
for s in a['seeds']:
    m=s['metrics'];n,d,r=[100*m[g]['stats'][0] for g in ['N','D','R']]
    lines.append(f"{s['seed']} | {n:.6f} | {d:.6f} | {r:.6f} | {r-n:+.6f} | {r-d:+.6f} | {100*(m['R']['stats'][3]-m['N']['stats'][3]):+.6f}")
    c=s['GT_changes_not_AP_decomposition']['0.5']['R-N']
    sub=m['R']['subset_fresh_matching']['0.5']['new_R_query'];deleted=m['N']['subset_fresh_matching']['0.5']['deleted_N_label']
    lines.append(f"seed{s['seed']} IoU.5 R对N新增/丢失GT={c['gained_GT']}/{c['lost_GT']}；新增query检出{ sub['TP']}个TP；原N被删除次高类条目里有{deleted['TP']}个TP。新条目背景标签(anyclass maxIoU<.1)={sub['unmatched_anyclass_maxIoU_lt_.1']}。")
    lines.append(f"原N重复query名额总计{s['total_duplicate_slots_N']}，补入新query同数{s['total_new_queries_R']}。逐类、尺寸得失和GT分母见原始JSON。实际fresh匹配受每类100截断，一些补位框不进入评估，因此条目TP数不能当GT净增益或AP分解。")
assert all(s['R_minus_N_AP_pp']<0 for s in a['seeds'])
assert a['occupancy_probe_passes'] is False and a['novel_method_established'] is False
lines += ['预先判据：R须在两个seed均优于N和D且有新GT；目前两个seed均不胜N，当前方案触发停止。',
          '解释：删除次高类别的损失远大于补位收益，不能将所有重复query条目当作无用占位。补位比仅删除稍好，只约束此具体后处理版本，不证明所有预算设计无效，也不证明低AP旧模型等于成熟96轮模型。',
          '实际CPU计算、无新模型前向、无训练步；未访问远端/GPU/监测/队列。下一轮需如实回交阴性结果，不改阈值或选类挽救本版本。',
          '结果SHA256 '+hashlib.sha256(p.read_bytes()).hexdigest()]
(ROOT/'23_缓存后处理结果与解释.txt').write_text('\n'.join(lines),encoding='utf8')
print('\n'.join(lines[7:]))
