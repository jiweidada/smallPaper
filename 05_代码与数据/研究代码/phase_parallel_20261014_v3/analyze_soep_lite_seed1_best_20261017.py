"""Append reproducible selected-checkpoint comparisons; never overwrite old evidence."""
import copy
import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / '英文小论文/data/06_第二轮_20261014'
DEST = DATA / 'SOEP_lite_seed1_best_stg2_20261017_v1'
OLD = DATA / '恢复执行_20261015_v4/best_stg2补齐/15条已完成实验汇总.json'
NAMES = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
INIT = '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(base, extra):
    result = copy.deepcopy(base)
    for key, value in extra.items():
        if key == '__include__':
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


old = read(OLD)
sources = list(old['source_manifest'])
for item in sources:
    assert digest(Path(item['path'])) == item['sha256'], item['path']
binding = read(DEST / 'checkpoint_log_binding.json')
assert binding['common_init_sha256'] == INIT
assert binding['job_status']['returncode'] == 0
assert binding['job_status']['state'] in ['complete', 'trained_pending_independent_eval']
assert binding['epochs_in_raw_log'] == list(range(96))
assert binding['checkpoint_epoch_zero_based'] == binding['best_stat']['epoch']
assert binding['checkpoint_epoch_zero_based'] >= 84
assert binding['checkpoint_epoch_zero_based'] >= binding['job_status']['resume_after_round']
for item in binding['files']:
    assert digest(DEST / item['name']) == item['sha256'], item['name']
    sources.append({'path': str(DEST / item['name']), 'remote': item['remote'], 'sha256': item['sha256']})
raw_log = [json.loads(line) for line in (DEST / 'log.txt').read_text().splitlines() if line.strip()]
raw_log = [row for row in raw_log if 'test_coco_eval_bbox' in row]
assert [row['epoch'] for row in raw_log] == list(range(96))
selected = next(row for row in raw_log if row['epoch'] == binding['checkpoint_epoch_zero_based'])
assert selected['test_coco_eval_bbox'] == binding['metrics_raw_0_to_1']
assert selected['test_coco_eval_bbox'][0] == binding['best_stat']['coco_eval_bbox']

rows = copy.deepcopy(old['rows'])
rows.append({
    'seed': 1, 'arm': 'SOEP-lite', 'best_round_one_based': binding['best_round_one_based'],
    'metrics_raw_0_to_1': dict(zip(NAMES, binding['metrics_raw_0_to_1'])),
    'metrics_percent': dict(zip(NAMES, [v * 100 for v in binding['metrics_raw_0_to_1']])),
    'checkpoint_sha256': binding['checkpoint_sha256'],
    'source': str(DEST / 'checkpoint_log_binding.json'),
    'selection': 'Actual AP-selected best_stg2 metadata and bound training EMA validation log',
    'training_complete': True, 'unified_round2_independent_evaluation_complete': False,
})
index = {(row['seed'], row['arm']): row for row in rows}
assert len(rows) == len(index) == 16
for row in rows:
    for arm, field in [('baseline', 'minus_same_seed_baseline_pp'), ('C', 'minus_same_seed_C_pp')]:
        row[field] = {key: row['metrics_percent'][key] - index[row['seed'], arm]['metrics_percent'][key] for key in NAMES}

baseline_dir = DATA.parent / '05_相位机制96轮_20261012/baseline_full96_seed1'
baseline_receipt = read(baseline_dir / 'collection_receipt.json')
assert baseline_receipt['best_epoch_one_based'] == index[1, 'baseline']['best_round_one_based']
assert baseline_receipt['metrics_raw_0_to_1'] == list(index[1, 'baseline']['metrics_raw_0_to_1'].values())
for name, item in baseline_receipt['collected_files'].items():
    assert digest(baseline_dir / name) == item['sha256']
base_cfg = merge(yaml.safe_load((baseline_dir / 'common.yml').read_text()),
                 yaml.safe_load((baseline_dir / 'proto96.yml').read_text()))
c_args = read(DATA / '结果评估_20261014/C_seed1/args.json')
s_args = read(DEST / 'args.json')
audit_path = DATA / 'execution_v1/pair_audit_B_effective.json'
audit = read(audit_path)
assert audit['init_sha256'] == INIT
assert all(item['defaults']['p'] == 0.5 for item in audit['default_evidence'])
assert all(item['all_effective_fields_equal'] for item in audit['comparisons'])


def effective(value):
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        if value.get('type') == 'RandomZoomOut':
            value.setdefault('p', 0.5)
        return {key: effective(val) for key, val in value.items()}
    if isinstance(value, list):
        return [effective(val) for val in value]
    return value


scalars = ['epoches', 'lrsheduler', 'lr_gamma', 'no_aug_epoch', 'warmup_iter', 'flat_epoch',
           'use_amp', 'use_ema', 'ema_decay', 'ema_warmups', 'clip_max_norm']
assert all(c_args[key] == s_args[key] for key in scalars)
assert c_args['seed'] == s_args['seed'] == 1
cfg_keys = ['epoches', 'flat_epoch', 'no_aug_epoch', 'warmup_iter', 'lrsheduler', 'lr_gamma',
            'evaluator', 'num_classes', 'remap_mscoco_category', 'train_dataloader',
            'val_dataloader', 'eval_spatial_size', 'PostProcessor', 'DFINETransformer',
            'DEIMCriterion', 'use_amp', 'use_ema', 'ema', 'scaler', 'clip_max_norm',
            'sync_bn', 'find_unused_parameters', 'lr_scheduler', 'lr_warmup_scheduler']
for key in cfg_keys:
    assert effective(base_cfg[key]) == effective(c_args['yaml_cfg'][key]) == effective(s_args['yaml_cfg'][key]), key
for key in ['type', 'lr', 'betas', 'weight_decay']:
    assert base_cfg['optimizer'][key] == c_args['yaml_cfg']['optimizer'][key] == s_args['yaml_cfg']['optimizer'][key]
assert s_args['yaml_cfg']['HybridEncoder_SOEP_Lite']['bridge_width'] == 16

cost_path = DATA / 'execution_v1/SOEP_lite_cost_A.json'
cost = read(cost_path)
cost_index = {row['arm']: row for row in cost['results']}
parameter_deltas = {arm: cost_index[arm]['parameters'] - cost_index['baseline_full96']['parameters']
                    for arm in ['c_full96', 'soep_lite_full96']}
assert parameter_deltas == {'c_full96': 17440, 'soep_lite_full96': 15424}
pairs = []
for seed in [0, 1]:
    for first, second in [('C', 'SOEP-lite'), ('SOEP-lite', 'baseline'), ('C', 'baseline'),
                          ('C', 'C-mean'), ('rank16', 'rank16-mean'), ('C', 'rank16'), ('C-mean', 'lowpass')]:
        pairs.append({'seed': seed, 'comparison': first + ' - ' + second,
                      'delta_pp': {key: index[seed, first]['metrics_percent'][key] - index[seed, second]['metrics_percent'][key]
                                   for key in NAMES}})
summary = {}
for comparison in ['C - SOEP-lite', 'C - C-mean', 'rank16 - rank16-mean']:
    values = [pair['delta_pp']['APs'] for pair in pairs if pair['comparison'] == comparison]
    summary[comparison] = {'n': len(values), 'APs_values_pp': values, 'mean_pp': statistics.mean(values),
                           'sample_sd_pp': statistics.stdev(values),
                           'inference': 'Descriptive paired results only; no significance, equivalence or causal localization conclusion.'}
scan = copy.deepcopy(old['fallacy_scan'])
for item in scan['items']:
    if item['name'] == 'Survivorship':
        item['note'] = 'All 16 completed trajectories included; four unfinished seed2 trajectories excluded from endpoint comparisons, retained as pending.'
    if item['name'] == 'Forking paths':
        item['note'] = 'Preserve AP-selected best_stg2; SOEP-lite seed1 selects round94 rather than last96 or maximum APs.'
assert len(scan['items']) == 11
now = dt.datetime.now(dt.timezone.utc)
limitations = [
    'All table scores are actual selected-checkpoint training EMA validation; unified independent val/test-dev inference remains pending.',
    'SOEP-lite is an SPD/depthwise-spatial-convolution/nonlinear-neck-fusion budget adaptation without OmniKernel, not original SOEP or its author implementation.',
    'SOEP-lite changes several architectural variables and has fewer nominal parameters/FLOPs than C; this comparison does not isolate the phase mechanism or prove a globally best design.',
    'C/C-mean changes the input subspace and gradient structure; equal nominal parameterization does not remove every optimization explanation.',
    'Rank16/rank16-mean share maximum linear mapping rank and nominal parameters, not every effective capacity; the seed1 APs gap is only +0.2067pp.',
    'Two observed completed seeds provide directional replication only. No significance, equivalence, minimum reliably detectable difference, or precise uncertainty claim.',
    'One dataset and one detector; second dataset and direct localization/size/translation analyses remain uncompleted in this report.',
    'Legacy C seed1 resumed after epoch3 without full RNG restoration; SOEP-lite resumed from preserved round58 state after v5 path repair. No bitwise uninterrupted training claim.',
    'Seed1 large-object AP: C 48.3738 versus baseline 51.8983 and SOEP-lite 48.9255; no universal all-scale improvement claim.',
    'The original APs-only phase-main historical gate is retained unchanged; the revised contribution uses the prespecified C/C-mean and rank pairs with narrower claims.',
]
extra_sources = [OLD, DEST / 'remote_binding_B.json', DEST / 'checkpoint_log_binding.json', audit_path, cost_path,
                 DATA / 'SOEP_lite_结构与成本审查_20261014.txt',
                 DATA / '结果评估_20261014/C_seed1/checkpoint_log_binding.json',
                 DATA / '结果评估_20261014/C_seed1/args.json',
                 baseline_dir / 'proto96.yml', baseline_dir / 'common.yml', Path(__file__).resolve()]
sources.extend({'path': str(path), 'sha256': digest(path)} for path in extra_sources)
report = {
    'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent', 'origin_mode': 'validate',
                          'origin_date': now.isoformat(), 'verification_status': 'ANALYZED', 'version_label': 'soep_seed1_v1'},
    'generated_utc': now.isoformat(), 'generated_los_angeles': now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
    'scope': '16 completed 96-round trajectories, actual AP-selected best_stg2 training EMA metrics; no new training or GPU inference.',
    'common_init_sha256': INIT, 'rows': rows, 'same_seed_pairs': pairs, 'descriptive_summary': summary,
    'recipe_checks': {'C_vs_SOEP_seed1_scalar_fields_equal': scalars,
                      'baseline_C_SOEP_effective_config_fields_equal': cfg_keys,
                      'RandomZoomOut_missing_p_equals_explicit_p': 0.5,
                      'optimizer_core_equal': ['type', 'lr', 'betas', 'weight_decay'],
                      'SOEP_branch_parameter_lr': 0.0001,
                      'architecture_parameter_group_difference': 'SOEP adds encoder.p2_fusion to the backbone lr/weight-decay policy; original groups unchanged.',
                      'same_data_paths_and_eval_settings': True,
                      'uninterrupted_bitwise_replay_claim': False},
    'cost_evidence_A': cost, 'branch_parameter_deltas': parameter_deltas,
    'historical_first_round_gate': old['historical_first_round_gate'],
    'fallacy_scan': scan, 'limitations': limitations, 'source_manifest': sources,
    'reproducibility': {'method': 'Checkpoint/log/source-hash validation; no inference or training re-run',
                        'metadata_binding_verified': True, 'independent_inference_verified': False},
}
json_path = DEST / '16条已完成实验_best_stg2汇总.json'
with json_path.open('x', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)

text = f'''SOEP-lite seed1：best_stg2整理与创新边界更新
检查时间：{report['generated_los_angeles']}（America/Los_Angeles）；UTC {report['generated_utc']}
Material Passport：experiment-agent / validate；Verification Status: ANALYZED；版本 soep_seed1_v1。

本次只读收集B端已完成SOEP-lite seed1，不重跑训练或GPU推理，不下载checkpoint。
实际best_stg2为第94轮（last_epoch=93），按val AP选择EMA权重；完整96条日志连续且唯一，权重best_stat与第94轮日志一致。
SOEP checkpoint SHA256：{binding['checkpoint_sha256']}
公共初始化SHA256：{INIT}
第96轮AP/APs为26.283247/17.517962，仅是末轮值，本报告不用于best比较。
所有AP/AR用百分制，差值用百分点；当前不是统一独立评估后的论文主表。

seed1主比较
方法 | best轮 | AP | AP50 | AP75 | APs | ΔAP基线 | ΔAPs基线
'''
for arm in ['baseline', 'C', 'SOEP-lite']:
    row = index[1, arm]; m = row['metrics_percent']; d = row['minus_same_seed_baseline_pp']
    text += f"{arm} | {row['best_round_one_based']} | {m['AP']:.4f} | {m['AP50']:.4f} | {m['AP75']:.4f} | {m['APs']:.4f} | {d['AP']:+.4f} | {d['APs']:+.4f}\n"
text += '\nseed1完整12指标\n指标 | baseline | C | SOEP-lite | C−SOEP-lite\n'
for key in NAMES:
    b, c, s = [index[1, arm]['metrics_percent'][key] for arm in ['baseline', 'C', 'SOEP-lite']]
    text += f'{key} | {b:.6f} | {c:.6f} | {s:.6f} | {c-s:+.6f}\n'
text += '\n两个种子的直接配对差\n种子 | 比较 | ΔAP | ΔAPs\n'
for pair in pairs:
    d = pair['delta_pp']
    text += f"seed{pair['seed']} | {pair['comparison']} | {d['AP']:+.6f} | {d['APs']:+.6f}\n"
text += f"\nC−SOEP-lite的APs两种子均值：{summary['C - SOEP-lite']['mean_pp']:+.6f}pp，仅描述，不是显著性判断。\n"
text += '''
本次更新能支持什么
1. 近邻受控对比已从一个种子补成两个：C的AP和APs在两个已观察种子均高于本版SOEP-lite。可报告“在当前检测器/VisDrone/96轮配方和相近参数预算下，C比本版受控SOEP-lite取得更高APs”。
2. 主机制证据来自C/C-mean：APs +0.715608/+0.732203pp。可以支持“访问2×2子格差异带来额外小目标收益”；新增SOEP-lite结果是结构参照，不承担单变量机制归因。
3. rank16/rank16-mean为+0.629381/+0.206701pp：在相同名义参数和最大线性映射秩下两个观察种子均有正向差值，但效应大小不同，只缩小了“最大映射秩”解释，未排除所有容量/优化解释。
4. C新增17,440参数，SOEP-lite新增15,424；不能说C比该参照更省参数或更省FLOPs。A端同覆盖calflops为C25.814464G、SOEP-lite25.746675G。单次时延中位数21.3837/22.2863ms不证明稳定速度优势，B端同口径复核仍待队列完成。

创新判断
已有数据支持的论文贡献是“低开销P2→P3实现＋受控子格差异实证＋两个种子的近邻结构参照”。可据此推进方法和实验写作。结构首创仍不能成立；P2四相位重排并入P3且不增加P2头有近邻先例，接入位置也没有已有优势证据。
SOEP-lite省略OmniKernel、使用深度空间卷积及瓶颈融合，属于自建低预算适配；不得改称复现完整SOEP或优于UAV-DETR。访问子格差异有用也不等于已经证明其通过“保留亚像素位置”提高定位；尺寸、共同/新增/丢失检出与定位分解仍须给出直接证据。
seed1大目标AP中C低于基线3.524455pp、低于SOEP-lite0.551649pp；贡献应集中小目标，不能宣称所有尺度全面获益。

定稿前关键缺口
1. 完成已在跑的seed2四个臂，报告C/C-mean、rank16/rank16-mean的第三个同种子配对；三种子本身不保证显著或等效。
2. 队列按原计划统一独立评估best_stg2（val与test-dev），论文主表采用该结果，保留训练日志与重评差异。
3. 完成D尺寸/定位分解与A平移测试，检验收益来自少漏检还是更准定位；不预设C更抗平移。
4. 核实第二数据集的划分、忽略区及官方评估后再训练最小关键配对；支持的是跨数据集适用性，不是跨域泛化。

统计与溯源
11/11种统计与方法谬误已检查，逐项记录见JSON。只给同种子配对描述，不给p值、显著性或等效结论；保留全部16条完成轨迹，seed2未完成端点不作结果。
核心配方、训练/验证路径、640输入、batch4、增强有效配置、EMA/AMP、max_dets100/top300与优化器核心字段一致。RandomZoomOut未写p和显式p=0.5由原运行环境源码默认值核对为相同。
旧C seed1从epoch3恢复、本次SOEP-lite从原第58轮检查点在v5独立目录续训的历史如实保留，不能声称与不中断训练逐位相同。新选中第94轮来自修复后的独立输出，不用冲突目录的续训结果。
原始日志、参数、任务状态、选模源码及SHA清单已保存；旧报告与旧证据均保留。
'''
txt_path = DEST / 'SOEP_lite_seed1_同种子对比与创新判断.txt'
with txt_path.open('x', encoding='utf-8') as handle:
    handle.write(text)
manifest = {'generated_utc': now.isoformat(), 'generated_los_angeles': report['generated_los_angeles'],
            'files': [{'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
                      for path in sorted(DEST.iterdir()) if path.is_file()]}
with (DEST / 'collection_manifest.json').open('x', encoding='utf-8') as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
print(json.dumps({'report': str(txt_path), 'full_json': str(json_path),
                  'SOEP_seed1_best_round': binding['best_round_one_based'],
                  'SOEP_seed1_AP': index[1, 'SOEP-lite']['metrics_percent']['AP'],
                  'SOEP_seed1_APs': index[1, 'SOEP-lite']['metrics_percent']['APs'],
                  'C_minus_SOEP': [pair for pair in pairs if pair['comparison'] == 'C - SOEP-lite'],
                  'recipe_fields_checked': len(cfg_keys), 'fallacy_scan': '11/11'}, ensure_ascii=False))
