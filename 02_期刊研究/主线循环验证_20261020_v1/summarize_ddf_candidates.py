"""Report all arms and apply the preregistered screening gate, without new probes."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'ddf_candidates_frozen'
OLD=ROOT.parents[1]/'英文小论文/data/10_来源分配冻结诊断_20261020_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
data=json.loads((OUT/'02_results.json').read_text(encoding='utf8'))
assert sha(ROOT/'45_两项核查前登记.txt')==data['protocol']['spec_sha256']
assert sha(ROOT/'ddf_candidates_frozen.py')==data['protocol']['runner_sha256']
keys=['weighted_geometry','FGL_unweighted_CE','FGL_native_studentIoU_weighted','MAL_native_own_matching','mean_IoU']
pooled=[];flat=[]
for seed in [1,2]:
    rows=[r for r in data['rows'] if r['seed']==seed]
    assert len(rows)==2
    def gt_number(r):
        v=json.loads((OLD/f'seed{seed}/batch{r["evaluation_batch"]:02d}_capture.json').read_text(encoding='utf8'))
        return sum(z['GT'] for z in v['population'])
    for mode in ['fixed','native']:
        for arm in data['protocol']['arms']:
            record=dict(seed=seed,mode=mode,arm=arm)
            for key in keys:
                weights=[gt_number(r) if key=='MAL_native_own_matching' else r['arms'][arm][mode]['denominator'] for r in rows]
                record[key]=sum(r['arms'][arm][mode][key]*w for r,w in zip(rows,weights))/sum(weights)
            pooled.append(record)
            for r in rows:
                flat.append(dict(seed=seed,mode=mode,arm=arm,generator_batch=r['generator_batch'],evaluation_batch=r['evaluation_batch'],**r['arms'][arm][mode]))
lookup={(r['seed'],r['mode'],r['arm']):r for r in pooled}
decisions=[]
for candidate,controls in [('multi_layer',['single_total','matched_norm']),('relocated_target',['GT_two_bin'])]:
    checks=[]
    for seed in [1,2]:
        cand=lookup[seed,'fixed',candidate]
        for control in ['original']+controls:
            base=lookup[seed,'fixed',control]
            checks.append(dict(seed=seed,check='G_gain_ge_1e-5',control=control,
                difference=cand['weighted_geometry']-base['weighted_geometry'],
                passed=cand['weighted_geometry']-base['weighted_geometry']<=-1e-5))
            native_diff=lookup[seed,'native',candidate]['weighted_geometry']-lookup[seed,'native',control]['weighted_geometry']
            checks.append(dict(seed=seed,check='native_rematching_G_no_reversal',control=control,difference=native_diff,passed=native_diff<=0))
        base=lookup[seed,'fixed','original']
        for key,tol in [('FGL_unweighted_CE',1e-5),('MAL_native_own_matching',1e-3)]:
            difference=cand[key]-base[key]
            checks.append(dict(seed=seed,check=key+'_noninferior_screen',control='original',difference=difference,tolerance=tol,passed=difference<=tol))
        for control in ['half_positive','off_positive','teacherbetter_gate']:
            ref=lookup[seed,'fixed',control]
            dominated=(ref['weighted_geometry']<=cand['weighted_geometry'] and ref['FGL_unweighted_CE']<=cand['FGL_unweighted_CE'] and
                (ref['weighted_geometry']<cand['weighted_geometry'] or ref['FGL_unweighted_CE']<cand['FGL_unweighted_CE']))
            checks.append(dict(seed=seed,check='not_simultaneously_dominated_G_FGL',control=control,passed=not dominated))
    decisions.append(dict(candidate=candidate,passes_preregistered_training_application_screen=all(c['passed'] for c in checks),checks=checks))
captures=[]
for seed in [1,2]:
    for batch in [0,1]:
        v=json.loads((OUT/f'seed{seed}/batch{batch:02d}_capture.json').read_text(encoding='utf8'))
        captures.append(dict(seed=seed,batch=batch,**{k:v[k] for k in ['checks','residual_stats','diagnostics']}))
result=dict(utc=data['protocol']['utc'],seconds=data['seconds'],pooled=pooled,decisions=decisions,captures=captures,
    limitations=data['protocol']['limitations'],results_sha256=sha(OUT/'02_results.json'),
    interpretation='This decision screens whether to apply for training, not AP effect, significance, equivalence, or universal impossibility.')
with (OUT/'03_summary_and_decisions.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
with (OUT/'04_all_pooled_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(pooled[0]));w.writeheader();w.writerows(pooled)
lines=['两项 DDF 冻结核查：完整结果与事前门槛','',f'耗时 {data["seconds"]:.2f} 秒；2 seed × 2 development 批，各 4 图，同一 8 张增强输入。新训练 0，远端操作 0。',
    'G 和 FGL 均是损失（越低越好），不是 AP。Δ 均为同 seed、同输入、同 Adam 状态的候选减 original。按评估分母汇总，不逐图平均。','',
    'seed | arm | G | ΔG | 不加权FGL | ΔFGL | MAL | 原生重选且重匹配G']
for r in pooled:
    if r['mode']!='fixed':continue
    ref=lookup[r['seed'],'fixed','original'];nr=lookup[r['seed'],'native',r['arm']]
    lines.append(f'{r["seed"]} | {r["arm"]} | {r["weighted_geometry"]:.10f} | {r["weighted_geometry"]-ref["weighted_geometry"]:+.10f} | {r["FGL_unweighted_CE"]:.10f} | {r["FGL_unweighted_CE"]-ref["FGL_unweighted_CE"]:+.10f} | {r["MAL_native_own_matching"]:.10f} | {nr["weighted_geometry"]:.10f}')
lines+=['','逐候选判定：']
for decision in decisions:
    lines.append(f'{decision["candidate"]}: '+('值得申请训练（仅筛查通过）' if decision['passes_preregistered_training_application_screen'] else '本次暂不申请训练'))
    for c in decision['checks']:
        if not c['passed']:lines.append('未通过：'+json.dumps(c,ensure_ascii=False))
lines+=['','诊断与核验（原始整数/误差，所有批次，不挑样本）：']
for c in captures:lines.append(json.dumps(c,ensure_ascii=False))
lines+=['','关键局限：'+data['protocol']['limitations'],'两项预算已用完。不得改门槛、继续温度/阈值搜索或给本版本换名后再筛。结果交回四专家；没有候选通过是允许的终点。']
text='\n'.join(lines)
(ROOT/'47_两项DDF核查完整结果.txt').write_text(text,encoding='utf8')
print(text)
