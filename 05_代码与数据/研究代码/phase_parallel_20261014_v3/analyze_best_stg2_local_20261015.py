"""Appraise only checkpoint-bound or explicit save-log-bound available data."""
import datetime
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parents[2]
DATA = ROOT/'英文小论文/data'
ROUND = DATA/'06_第二轮_20261014'
DEST = ROUND/'best_stg2整理_20261015_20261016T014100_614372Z'
OLD = ROUND/'结果评估_20261014/同种子完整指标与判据.json'
NAMES = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
manifest = []


def record(path, expected=None):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, str(path)
    manifest.append({'path': str(path), 'sha256': digest, 'bytes': len(raw)})
    return raw


def load(path):
    return json.loads(record(path))


old = load(OLD)
for item in old['source_manifest']:
    record(pathlib.Path(item['path']), item['sha256'])
recovery = load(DEST/'历史快照按实际best轮次恢复候选.json')
cost = load(ROUND/'execution_v1/SOEP_lite_cost_A.json')
saver = load(ROUND/'租期估算_20261015/20261015T075742Z_A.json')
rows = []
for original in old['rows']:
    row = dict(original)
    row['evidence_level'] = 'A: previous checkpoint-metadata audit and bound original log'
    rows.append(row)
mapping = {'rank16_mean_full96_seed0': ('rank16-mean', 0),
           'soep_lite_full96_seed0': ('SOEP-lite', 0),
           'rank16_full96_seed1': ('rank16', 1)}
for job, (arm, seed) in mapping.items():
    matches = recovery['matches'][job]
    assert matches and recovery['targets_zero_based'][job] == 95
    first = matches[0]
    assert all(r['metrics'] == first['metrics'] for r in matches)
    snapshot = load(pathlib.Path(first['path']))
    observed = snapshot['servers'][first['account']]['snapshot']['jobs'][job]
    assert observed['state'] == 'trained_pending_independent_eval' and observed['status']['returncode'] == 0
    assert observed['evaluated_epochs'] == 96 and observed['latest_evaluation']['epoch'] == 95
    assert any('Saved best_stg2.pth' in line for line in observed['tail_lines'])
    values = observed['latest_evaluation']['test_coco_eval_bbox']
    rows.append({'seed': seed, 'arm': arm, 'job': job, 'best_round_one_based': 96,
                 'metrics_raw_0_to_1': dict(zip(NAMES, values)),
                 'metrics_percent': dict(zip(NAMES, [v*100 for v in values])),
                 'selection': 'Completed final epoch and explicit final best_stg2 save log; no later training update.',
                 'evidence_level': 'B: exact final best_stg2 save log plus full final evaluation snapshot; checkpoint metadata/hash pending',
                 'source': first['path'], 'source_log_sha256': observed['metrics_sha256'],
                 'checkpoint_sha256': None, 'training_complete': True,
                 'unified_round2_independent_evaluation_complete': False})
index = {(r['seed'], r['arm']): r for r in rows}
assert len(index) == len(rows) == 12
for row in rows:
    base = index[row['seed'], 'baseline']['metrics_percent']
    c = index[row['seed'], 'C']['metrics_percent']
    row['minus_same_seed_baseline_pp'] = {k: row['metrics_percent'][k]-base[k] for k in NAMES}
    row['minus_same_seed_C_pp'] = {k: row['metrics_percent'][k]-c[k] for k in NAMES}
pairs = []
for seed in [0, 1]:
    for control in ['baseline', 'lowpass', 'singlephase', 'C-mean', 'rank16', 'rank16-mean', 'SOEP-lite']:
        if (seed, control) not in index:
            continue
        pairs.append({'seed': seed, 'comparison': 'C - '+control,
                      'delta_pp': {k: index[seed, 'C']['metrics_percent'][k]-index[seed, control]['metrics_percent'][k]
                                   for k in NAMES},
                      'control_evidence_level': index[seed, control]['evidence_level']})
missing = [{'seed': 0, 'arm': 'rank16', 'job': 'rank16_full96_seed0', 'best_round_one_based_from_log': 95},
           {'seed': 1, 'arm': 'C-mean', 'job': 'c_mean_full96_seed1', 'best_round_one_based_from_log': 94},
           {'seed': 1, 'arm': 'rank16-mean', 'job': 'rank16_mean_full96_seed1', 'best_round_one_based_from_log': 95}]
snapshots = sorted((ROUND/'每半小时跟进').glob('*_snapshot.json'))
latest_success = None
for path in reversed(snapshots):
    candidate = json.loads(path.read_text(encoding='utf-8'))
    if set(candidate.get('servers', {})) == {'A', 'B'} and all(
            r.get('ssh_returncode') == 0 and 'snapshot' in r for r in candidate['servers'].values()):
        latest_success = candidate
        record(path)
        break
assert latest_success
for row in missing:
    a = 'A' if row['seed'] == 0 else 'B'
    j = latest_success['servers'][a]['snapshot']['jobs'][row['job']]
    assert j['evaluated_epochs'] == 96 and j['status']['returncode'] == 0
    expected_epoch = row['best_round_one_based_from_log']-1
    assert any('(epoch '+str(expected_epoch)+')' in text for text in j['tail_lines'])
    row['evidence_status'] = 'Training complete; best epoch located in completion log, selected epoch metrics not locally available. No use of final-epoch metrics as replacement.'
pending_pairs = [{'comparison': 'C - C-mean', 'seed': 1, 'reason': 'C-mean best-round metrics missing'},
                 {'comparison': 'rank16 - rank16-mean', 'seed': 0, 'reason': 'rank16 best-round metrics missing'},
                 {'comparison': 'rank16 - rank16-mean', 'seed': 1, 'reason': 'rank16-mean best-round metrics missing'},
                 {'comparison': 'C - SOEP-lite', 'seed': 1, 'reason': 'SOEP-lite not completed at last successful inspection'},
                 {'comparison': 'key paired seed2 comparisons', 'seed': 2, 'reason': 'Training not completed at last successful inspection'}]
scan = [
    {'name': 'Simpson reversal', 'finding': 'Keep each seed separate; do not pool known and missing controls.'},
    {'name': 'Ecological inference', 'finding': 'Aggregate APs/AP75 do not establish positional information or per-object localization.'},
    {'name': 'Berkson selection', 'finding': 'Single dataset and detector; candidate selected after prior exploration.'},
    {'name': 'Collider selection', 'finding': 'Future common-detection errors are conditional on joint detection, not population causal effects.'},
    {'name': 'Base-rate neglect', 'finding': 'Size-bin/class GT denominators pending unified D analysis.'},
    {'name': 'Regression to mean', 'finding': 'Best-AP checkpoint selection and prior C selection retained; no maximum APs selection.'},
    {'name': 'Survivorship', 'finding': 'Three missing completed arms and unfinished jobs explicitly retained.'},
    {'name': 'Look elsewhere', 'finding': 'No significance inference after exploratory arm selection.'},
    {'name': 'Forking paths', 'finding': 'Predefined checkpoint rule and first-round gate preserved; no new success threshold.'},
    {'name': 'Causal overstatement', 'finding': 'Mean changes input subspace and gradients; rank pairs missing; lite comparison changes several factors.'},
    {'name': 'Reverse causality', 'finding': 'Architecture precedes training, but this does not prove the positional mechanism.'}
]
report = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'material_passport': {'origin_skill': 'academic-research-suite', 'origin_mode': 'bounded validation and appraisal',
                                'verification_status': 'ANALYZED', 'single_assistant': True},
          'last_successful_inspection_los_angeles': latest_success['checked_los_angeles'],
          'connection_status': 'Both former SSH endpoints returned connection reset during collection; GPU reapplication in progress per user. No current training-state claim.',
          'protocol': 'Actual AP-selected best_stg2 training EMA validation; no unified independent evaluation rerun in this analysis.',
          'available_rows': rows, 'missing_completed_rows': missing, 'same_seed_pairs': pairs,
          'pending_pairs': pending_pairs,
          'evidence_counts': {'checkpoint_metadata_audited': 9, 'explicit_best_save_log_bound': 3,
                              'completed_but_selected_metrics_missing': 3},
          'claims': [
              {'claim': 'Low-overhead C improves small-object detection in tested DEIM/VisDrone setting',
               'judgment': 'supported in two observed training seeds; not statistical significance or generality'},
              {'claim': 'Access to subcell differences adds value over mean-only C parameterization',
               'judgment': 'supported in completed audited seed0 pair only; seed1 finished but selected metrics missing'},
              {'claim': 'C outperforms the budget-limited SOEP-inspired adaptation',
               'judgment': 'supported descriptively in seed0 save-log-bound comparison; not full SOEP, not replicated across seeds'},
              {'claim': 'Higher maximum linear rank alone explains all benefits',
               'judgment': 'not yet assessable because both exact rank16/rank16-mean pairs are missing one selected result'},
              {'claim': 'Novel operator, first P2 four-phase fusion, optimal placement, lossless block or proven positional mechanism',
               'judgment': 'not supported; earlier prior-art and placement/equivalence audits still apply'}
          ],
          'cost': {'C_parameter_delta': 17440, 'rank16_parameter_delta': 6176, 'rank16_mean_parameter_delta': 6176,
                   'C_mean_parameter_delta': 17440, 'SOEP_lite_parameter_delta': 15424,
                   'C_minus_SOEP_lite_parameters': 2016, 'same_card_cost_raw': cost,
                   'limit': 'No proven speedup or Pareto superiority; precision results are budget-adaptation specific.'},
          'historical_first_round_gate': old['historical_first_round_gate'],
          'placement_C8_minus_O8_pp': old['placement_C8_minus_O8_pp'],
          'seed0_independent_testdev_delta_pp': old['seed0_independent_testdev_delta_pp'],
          'limitations': ['No checkpoint metadata/hash or actual args collected for three save-log-bound new rows.',
                          'Three other completed arms have no locally available exact selected-row metrics.',
                          'Old C seed1 resumed from epoch3 without complete RNG-restoration guarantee.',
                          'Only one dataset/detector; same public initialization; two training-seed observations.',
                          'Unified independent inference, D/A and second-dataset verification remain pending.'],
          'fallacy_scan': {'coverage': '11/11', 'items': scan}, 'source_manifest': manifest,
          'next_data_read': [{'job': r['job'], 'best_round_one_based': r['best_round_one_based_from_log'],
                              'need': ['best_stg2 CPU metadata/hash', 'full log.txt', 'args.json']} for r in missing],
          'actions': {'training_mutated': False, 'new_training': False, 'GPU_inference': False,
                      'checkpoint_download': False, 'old_artifacts_overwritten': False}}
txt = '''已完成实验 best_stg2 同种子整理（2026-10-15，本地可用证据版）

范围与限制
最后成功核对两端：{time}（America/Los_Angeles）。随后两个旧SSH端口均连接重置；用户正在重新申请GPU。当前实际训练状态未知，不用旧快照冒充实时状态。
已整理12条实际best对应指标：9条有此前权重元数据与原始日志绑定；新增3条由结束日志明确确认末轮保存best_stg2，再与该轮12项指标绑定。新增3条权重元数据/SHA、实际args仍待补核。另3条已完成训练，只定位到best轮次，缺该轮指标，严格留空。
统一独立评估尚未完成。所有下表数值都是best_stg2对应的训练EMA验证指标，不是独立推理最终指标；不选最高APs。

证据A：权重元数据及原始日志已审计；B：实际best保存日志及同轮完整快照确认，权重元数据尚待补核；缺项：不是训练未完成。
'''.format(time=latest_success['checked_los_angeles'])
arms = ['baseline', 'C', 'lowpass', 'singlephase', 'C-mean', 'rank16', 'rank16-mean', 'SOEP-lite']
for seed in [0, 1]:
    txt += '\nseed'+str(seed)+'（AP/APs为百分制，差值为百分点）\n'
    txt += '方法 | best轮 | AP | APs | ΔAP对基线 | ΔAPs对基线 | ΔAPs对C | 证据\n'
    for arm in arms:
        row = index.get((seed, arm))
        if row:
            m = row['metrics_percent']; b = row['minus_same_seed_baseline_pp']; c = row['minus_same_seed_C_pp']
            txt += f"{arm} | {row['best_round_one_based']} | {m['AP']:.4f} | {m['APs']:.4f} | {b['AP']:+.4f} | {b['APs']:+.4f} | {c['APs']:+.4f} | {row['evidence_level'][0]}\n"
        else:
            missing_row = next((r for r in missing if r['seed'] == seed and r['arm'] == arm), None)
            if missing_row:
                txt += f"{arm} | {missing_row['best_round_one_based_from_log']}（保存日志定位） | 待补读 | 待补读 | 待补读 | 待补读 | 待补读 | 缺项\n"
            elif seed == 1 and arm == 'SOEP-lite':
                txt += 'SOEP-lite | 尚无完成结果 | — | — | — | — | — | 排除中期结果\n'
txt += '\n可计算的同种子差值（C减对照，百分点）\n'
for pair in pairs:
    txt += f"seed{pair['seed']} {pair['comparison']}: AP {pair['delta_pp']['AP']:+.6f}, APs {pair['delta_pp']['APs']:+.6f}\n"
txt += '''
当前结论
1. C的低开销实用效果成立于已观察设置：两个种子APs各+0.946533/+1.225355pp，AP各+0.657411/+0.943089pp。不能称统计显著或普遍适用。
2. C/C-mean seed0 APs +0.715608pp、AP +0.603302pp，是访问子格差异的直接初步证据。seed1训练已完成，缺的是第94轮指标；不能拿它第96轮APs17.9188替代。
3. seed0 C比本版SOEP-lite APs高+0.828652pp、AP高+0.638841pp，形成第一条预算近邻比较证据。SOEP-lite比C少2016参数；C不能称参数更少、速度更快或成本最优，也不能据此声称超过完整SOEP。seed1未完。
4. rank16 seed1以6176新增参数获得APs +0.676019pp、AP +0.644643pp；比C的APs低0.549336pp。只能说明低秩实现也有收益，不能替代rank16/rank16-mean的机制配对。
5. 两个rank配对都已训练完成，但各缺一条best选中指标，目前不能判断相同最大映射秩下子格差异的增量贡献，也不能排除映射秩解释。此处不需要新增四次训练，只需补取现有证据。

创新性判断
结构/算子首创仍不成立，主干接入优势、全模块无损与定位机制已证实都不能写。能够保留的方向是低开销工程实现，以及P2子格差异在受控条件下的实证研究。新增seed0 SOEP-lite比较使工程部分更有依据，但机制主配对还没有形成两个种子的闭环。
当前没有据此放弃研究线的理由；可以继续准备方法、已确认结果和相关工作，暂不直接投稿。不要把“第96轮看起来不错”当best配对结论，也不要把未读回的结果当阴性结果。

下一步最小工作
先补读c_mean seed1（best第94轮）、rank16 seed0（best第95轮）、rank16-mean seed1（best第95轮）的best元数据与日志；补核本次3条B级结果。完成已排队的SOEP-lite seed1与关键seed2，不新加模块。然后统一独立评估、尺寸/定位分解，核实第二数据集协议后再验证。平移分析不预设有利于C。
待补材料都来自已有训练，不需要为数据采集缺口重跑。当前目录保留本次失败采集的stderr和空stdout，便于区分读取失败与训练失败。

完整12指标、配对差、缺失原因、证据等级、11/11推理误区扫描及来源SHA见同名JSON。本次没有改队列、停止/重启训练或新增GPU工作。
'''
with (DEST/'同种子best_stg2完整指标与比较.json').open('x', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
with (DEST/'已完成实验同种子比较与创新边界.txt').open('x', encoding='utf-8') as handle:
    handle.write(txt)
print(txt)
print('DESTINATION', DEST)
