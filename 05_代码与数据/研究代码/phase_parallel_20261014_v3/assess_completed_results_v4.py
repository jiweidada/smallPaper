"""Append a complete selected-checkpoint table after successful recollection."""
import datetime
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parents[2]
DATA = ROOT/'英文小论文/data/06_第二轮_20261014'
DEST = DATA/'恢复执行_20261015_v4/best_stg2补齐'
OLD = DATA/'结果评估_20261014/同种子完整指标与判据.json'
NAMES = ['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
old = json.loads(OLD.read_text(encoding='utf-8'))
sources = []
for item in old['source_manifest']:
    path = pathlib.Path(item['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], str(path)
    sources.append(item)
rows = [dict(r) for r in old['rows'] if r['arm'] != 'C-mean']
mapping = {'c_mean_full96': 'C-mean', 'rank16_full96': 'rank16',
           'rank16_mean_full96': 'rank16-mean', 'soep_lite_full96': 'SOEP-lite'}
recipe_fields = ['epoches','lrsheduler','lr_gamma','no_aug_epoch','warmup_iter','flat_epoch',
                 'use_amp','use_ema','ema_decay','ema_warmups','clip_max_norm']
reference = json.loads((DATA/'结果评估_20261014/C_seed1/args.json').read_text(encoding='utf-8'))
recipes = []
for path in sorted(DEST.glob('*/checkpoint_log_binding.json')):
    record = json.loads(path.read_text(encoding='utf-8'))
    for item in record['files']:
        p = path.parent/item['name']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], str(p)
        sources.append({'path': str(p), 'sha256': item['sha256']})
    sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    log = [json.loads(line) for line in (path.parent/'log.txt').read_text().splitlines() if line.strip()]
    log = [r for r in log if 'test_coco_eval_bbox' in r]
    assert [r['epoch'] for r in log] == list(range(96))
    epoch = record['checkpoint_epoch_zero_based']
    metrics = next(r['test_coco_eval_bbox'] for r in log if r['epoch'] == epoch)
    assert metrics == record['metrics_raw_0_to_1'] and record['best_stat']['epoch'] == epoch
    assert abs(record['best_stat']['coco_eval_bbox']-metrics[0]) < 1e-12
    args = json.loads((path.parent/'args.json').read_text(encoding='utf-8'))
    assert all(args[k] == reference[k] for k in recipe_fields)
    y, ref = args['yaml_cfg'], reference['yaml_cfg']
    assert y['train_dataloader']['total_batch_size'] == ref['train_dataloader']['total_batch_size'] == 4
    assert y['train_dataloader']['collate_fn'] == ref['train_dataloader']['collate_fn']
    assert y['eval_spatial_size'] == ref['eval_spatial_size'] == [640,640]
    assert all(y['optimizer'][k] == ref['optimizer'][k] for k in ['type','lr','betas','weight_decay'])
    recipes.append({'job': record['job'], 'scalar_fields_match': recipe_fields,
                    'batch_collate_eval_size_optimizer_core_match': True,
                    'note': 'Architecture-specific parameter groups and resume history are retained separately; this is not a bitwise training-replay claim.'})
    rows.append({'seed': record['seed'], 'arm': mapping[record['arm']],
                 'best_round_one_based': epoch+1, 'metrics_raw_0_to_1': dict(zip(NAMES, metrics)),
                 'metrics_percent': dict(zip(NAMES, [v*100 for v in metrics])),
                 'checkpoint_sha256': record['checkpoint_sha256'], 'source': str(path),
                 'training_complete': True, 'unified_round2_independent_evaluation_complete': False,
                 'selection': 'actual AP-selected best_stg2 metadata and bound training EMA validation log'})
index = {(r['seed'],r['arm']):r for r in rows}
assert len(rows) == len(index) == 15
for row in rows:
    row['minus_same_seed_baseline_pp'] = {k:row['metrics_percent'][k]-index[row['seed'],'baseline']['metrics_percent'][k] for k in NAMES}
    row['minus_same_seed_C_pp'] = {k:row['metrics_percent'][k]-index[row['seed'],'C']['metrics_percent'][k] for k in NAMES}
pairs = []
for seed in [0,1]:
    for first, second in [('C','C-mean'),('rank16','rank16-mean'),('C','rank16'),('C','SOEP-lite'),('C-mean','lowpass')]:
        if (seed,first) in index and (seed,second) in index:
            pairs.append({'seed':seed,'comparison':first+' - '+second,
                          'delta_pp':{k:index[seed,first]['metrics_percent'][k]-index[seed,second]['metrics_percent'][k] for k in NAMES}})
summary = {}
for comparison in ['C - C-mean','rank16 - rank16-mean']:
    values = [p['delta_pp']['APs'] for p in pairs if p['comparison'] == comparison]
    summary[comparison] = {'APs_values_pp':values,'mean_pp':statistics.mean(values),
                           'sample_sd_pp':statistics.stdev(values),'n':len(values),
                           'inference':'Descriptive only; no significance or equivalence inference.'}
scan = json.loads(json.dumps(old['fallacy_scan']))
for item in scan['items']:
    if item['name'] == 'Causal overstatement':
        item['note'] = 'Same-maximum-rank pairs are now available and positive in two observed seeds; mean still changes input subspace and gradients. No claim that all capacity or optimization explanations are excluded.'
    if item['name'] == 'Survivorship':
        item['note'] = 'All 15 completed trajectories are included; unfinished seed2 and SOEP-lite seed1 are retained as pending and not treated as negative results.'
report = {'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'verification_status':'ANALYZED', 'rows':rows, 'same_seed_pairs':pairs,
          'descriptive_summary':summary, 'recipe_checks':recipes,
          'old_missing_three_selected_metrics_now_collected':True,
          'historical_first_round_gate':old['historical_first_round_gate'],
          'fallacy_scan':scan,
          'additional_limitations':['All current results are selected-checkpoint training EMA, pending unified independent inference.',
                                    'Mean changes input subspace and gradient structure; equal maximum linear rank is not equal all capacity/optimization.',
                                    'Rank pair APs gap is +0.6294/+0.2067 pp across observed seeds; no claim that these are equal effects.',
                                    'SOEP-lite is one completed seed of a budget adaptation, not full SOEP.',
                                    'Only one dataset and detector; legacy C seed1 resume caveat remains.',
                                    'Seed2 runs are continuing from saved states; no completed seed2 included.'],
          'source_manifest':sources}
with (DEST/'15条已完成实验汇总.json').open('x',encoding='utf-8') as handle:
    json.dump(report,handle,ensure_ascii=False,indent=2)
text = '已完成15条96轮轨迹：实际best_stg2对应训练EMA验证结果\n\n全部按权重元数据定位选中轮次，按AP选模，不按APs选模。此前三个缺项已补齐；没有重跑或覆盖原结果。统一独立评估尚未完成。\n'
for seed in [0,1]:
    text += '\nseed'+str(seed)+'\n方法 | best轮 | AP | APs | ΔAP基线 | ΔAPs基线\n'
    for arm in ['baseline','C','lowpass','singlephase','C-mean','rank16','rank16-mean','SOEP-lite']:
        row = index.get((seed,arm))
        if row:
            m,b = row['metrics_percent'],row['minus_same_seed_baseline_pp']
            text += f"{arm} | {row['best_round_one_based']} | {m['AP']:.4f} | {m['APs']:.4f} | {b['AP']:+.4f} | {b['APs']:+.4f}\n"
text += '\n同种子直接配对（百分点）\n'
for p in pairs:
    text += f"seed{p['seed']} {p['comparison']}: AP {p['delta_pp']['AP']:+.6f}, APs {p['delta_pp']['APs']:+.6f}\n"
text += '''
结论更新
C/C-mean两个观察种子APs差为+0.715608/+0.732203pp，主配对方向一致、数值接近。rank16/rank16-mean同参数6176、同最大线性映射秩16的配对差为+0.629381/+0.206701pp，两个观察种子均正，但效应大小不一致。可表述为“访问子格差异在相同最大映射秩条件下也观察到额外收益”，不能说排除所有容量/优化解释或证明定位因果机制。
C对SOEP-lite seed0的APs差+0.828652pp仅适用于本版低预算受控适配，seed1仍在续训，不代表完整SOEP。C-mean与低通APs差seed0+0.174039、seed1-0.051149pp，只能描述接近，不能宣布等效。
创新边界：低开销工程实现和受控子格差异实证更有依据；结构首创、最优接入位置、全模块无损、统计显著、完整SOEP优越性及跨数据集结论仍不能主张。可以推进论文写作；投稿前仍完成已排队重复、统一独立评估、尺寸/定位分析和第二数据集验证。不新增模块来凑创新点。

完整12项指标、所有配对差、配方检查和来源SHA见同名汇总JSON。旧“待补读”报告保留历史，不再代表当前数据可用性。
'''
with (DEST/'同种子比较与结论更新.txt').open('x',encoding='utf-8') as handle:
    handle.write(text)
print('PAIRS',[(r['seed'],r['comparison'],round(r['delta_pp']['APs'],6)) for r in pairs])
print('COMPLETED',len(rows),'REPORT',DEST/'同种子比较与结论更新.txt')
