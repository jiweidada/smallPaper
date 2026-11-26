"""Read-only evidence appraisal; write new local review artifacts only."""
import datetime
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parents[2]
DATA = ROOT / '英文小论文/data'
DEST = DATA / '06_第二轮_20261014/结果评估_20261014'
BASE = DEST / '同种子完整指标与判据.json'
MATCH = DATA / ('00_实验计划与证据索引/C完整重复验收辅助_20261003/'
                'dual_view_execution_20261004_v1/repeat_small_matching/repeat_small_matching_result.json')
SNAP = DATA / '06_第二轮_20261014/每半小时跟进/20261015T022128_401570Z_snapshot.json'
COST = DATA / '06_第二轮_20261014/execution_v1/SOEP_lite_cost_A.json'
HAAR = DATA / '00_实验计划与证据索引/模型数据与既有文献联合调研_20261006_v1/04_相位与Haar线性等价边界.json'
PLAN = DATA / '00_实验计划与证据索引/第二轮实验计划与Codex规格_20261014.md'
manifest = []


def record(path, expected=None):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    manifest.append({'path': str(path), 'sha256': digest, 'bytes': len(raw)})
    return raw


def load(path):
    return json.loads(record(path))


data, old, snapshot, cost, equivalence = [load(p) for p in [BASE, MATCH, SNAP, COST, HAAR]]
record(PLAN)
for item in data['source_manifest']:
    record(pathlib.Path(item['path']), item['sha256'])
for item in old['sources']:
    record(pathlib.Path(item['path']), item['sha256'])
index = {(r['seed'], r['arm']): r for r in data['rows']}
assert len(index) == 9
pairs = []
for seed in [0, 1]:
    for other in ['baseline', 'lowpass', 'singlephase', 'C-mean']:
        if (seed, other) not in index:
            continue
        first, second = index[seed, 'C'], index[seed, other]
        pairs.append({'seed': seed, 'comparison': 'C - ' + other,
                      'delta_pp': {k: first['metrics_percent'][k] - second['metrics_percent'][k]
                                   for k in first['metrics_percent']}})
jobs = {}
for account, row in snapshot['servers'].items():
    assert row['ssh_returncode'] == 0 and 'snapshot' in row
    for name, job in row['snapshot']['jobs'].items():
        jobs[name] = {'account': account, 'state': job['state'],
                     'evaluated_epochs': job.get('evaluated_epochs'),
                     'unified_independent_evaluation_splits_obtained':
                     list(job.get('independent_evaluations', {}))}
assert jobs['c_mean_full96_seed0']['state'] == 'trained_pending_independent_eval'
assert jobs['c_mean_full96_seed1']['state'] == 'running'
assert all(jobs[f'soep_lite_full96_seed{s}']['state'] == 'pending' for s in [0, 1])
old_models = old['models']
old_small = {k: (old_models['C4_repeat']['APs_by_IoU'][k]
                 - old_models['baseline_p05']['APs_by_IoU'][k]) * 100
             for k in ['0.50', '0.75']}
classes = old_models['C4_repeat']['per_class_APs']
old_class_deltas = {
    k: (classes[k]['APs'] - old_models['baseline_p05']['per_class_APs'][k]['APs']) * 100
    for k in classes}
assert len(old_class_deltas) == 10 and all(v > 0 for v in old_class_deltas.values())
cost_index = {r['arm']: r for r in cost['results']}
soep_delta = cost_index['soep_lite_full96']['parameters'] - cost_index['baseline_full96']['parameters']
c_delta = cost_index['c_full96']['parameters'] - cost_index['baseline_full96']['parameters']
assert (soep_delta, c_delta) == (15424, 17440)

assessment = {
    'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Single-assistant bounded appraisal, not a manuscript peer-review panel or acceptance prediction.',
    'checked_los_angeles': snapshot['checked_los_angeles'],
    'primary_metrics_protocol': 'Nine completed 96-epoch trajectories, training EMA validation at actual AP-selected best_stg2 epoch; not unified independent evaluation.',
    'rows': data['rows'], 'recomputed_same_seed_pairs': pairs,
    'job_states_at_check': jobs,
    'supported_engineering_contribution': 'Low-overhead compressed zero-initialized P2-to-P3 residual integration, retaining P3-P5; incremental engineering evidence, not operator or topology priority.',
    'provisional_empirical_contribution': 'C exceeds trained phase-mean control in the completed seed0 pair by 0.7156082776774433 pp APs; no second completed C-mean pair yet.',
    'unsupported_claims': ['First P2 four-phase transfer without P2 head',
                           'Backbone placement is superior', 'The block is lossless',
                           'Subpixel positional mechanism proved', 'All capacity or optimization explanations excluded',
                           'Best cost or better than full SOEP', 'Statistical significance or equivalence',
                           'Cross-dataset or cross-domain generalization'],
    'placement_C_minus_O_pp': data['placement_C8_minus_O8_pp'],
    'existing_independent_seed0_testdev_delta_pp': data['seed0_independent_testdev_delta_pp'],
    'legacy_diagnostic': {'comparison': 'C4_repeat versus historical p05; separate from current seed1 baseline and C-mean controls',
                          'protocol': old['protocol'], 'small_AP_by_IoU_delta_pp': old_small,
                          'per_class_APs_delta_pp': old_class_deltas,
                          'paired_small_GT': {k: old['paired_small_GT'][k] for k in ['0.50', '0.75']},
                          'bin_definition_incompatibility': 'Legacy uses short side after 640 warp; new plan uses sqrt(w*h) in input coordinates. Recompute before combining.'},
    'engineering_cost': {'C_parameter_delta': c_delta, 'SOEP_lite_parameter_delta': soep_delta,
                         'relative_C_FLOPs_overhead': data['cost_evidence']['relative_FLOPs_overhead'],
                         'latency_limit': 'Existing one-session same-card measurements do not establish speedup.'},
    'linear_reparameterization_boundary': equivalence,
    'required_for_recommended_submission': [
        'Bind all main-table checkpoints to one independent val/test-dev evaluation protocol; retain resume and selection history.',
        'Finish C/C-mean paired replications and rank16/rank16-mean pairs; use planned third seed descriptively, never require or invent significance.',
        'Finish same-recipe SOEP-lite accuracy and cost comparison; identify as inspired adaptation, not full SOEP replication.',
        'Validate one second dataset with checked split, ignore policy, official evaluation and matched baseline/C/mean control.',
        'Complete unified size and detection/localization analysis with GT counts; audit closest literature for the exact empirical question.'
    ],
    'optional_unless_claimed': ['Translation suite unless making shift-equivariance or robustness claims',
                               'P2 head comparator unless claiming superiority to adding P2 detection',
                               'A second detector is useful for a method-general claim, not a prerequisite for a detector-specific paper',
                               'Baseline seed2 is needed for three-seed C-minus-baseline claims, not for the C-minus-C-mean seed2 pair'],
    'editorial_judgment': 'Not ready for SIVP submission today. Incremental engineering contribution is supported; the candidate controlled empirical contribution remains partial. Readiness depends on the above evidence and an honest distinct contribution, not the number of modules or experiments.',
    'SIVP_official_scope': 'https://link.springer.com/journal/11760/aims-and-scope',
    'journal_policy_boundary': 'Two datasets or three seeds are recommendations for this paper, not stated SIVP numerical submission rules.',
    'source_manifest': manifest,
    'actions': {'new_training': False, 'GPU_inference': False, 'queue_mutation': False,
                'remote_readonly_monitor': True, 'existing_results_overwritten': False}
}
text = '''SIVP 投稿证据审查（单助手、限定证据范围）

判断：现在不建议直接投 SIVP。已经成立的是低开销实现的工程价值；最值得作为研究贡献的“受控子格差异证据”已有首个主配对支持，但尚未闭合。结构首创、位置优势、无损传递、定位因果机制及近邻性能优越性均不能主张。

本次核验时间（America/Los_Angeles）：{time}
现有9条完成96轮轨迹。各条目的数字来自按总体AP选出的实际best_stg2对应训练EMA日志，不是末轮或最高APs；统一独立评估尚未收齐。

当前同种子比较（百分点）：
'''.format(time=snapshot['checked_los_angeles'])
for pair in pairs:
    text += 'seed{seed} {comparison}: ΔAP={ap:+.6f}, ΔAPs={aps:+.6f}\n'.format(
        seed=pair['seed'], comparison=pair['comparison'],
        ap=pair['delta_pp']['AP'], aps=pair['delta_pp']['APs'])
text += '''
可保留的两个贡献方向：
1. 低开销P2到P3残差桥接，保留原检测层。新增17,440参数，已有同口径FLOPs约+1.04%。这是增量工程贡献，不是新算子/首个无P2头浅层融合，也不能称成本最优。SOEP-lite新增15,424参数，精度尚待训练。
2. 在当前DEIM/VisDrone设置中，用训练期均值替换来检验P2特征2×2子格差异的价值。C/C-mean seed0的APs差+0.715608pp是最直接的初步证据。同名义参数不代表相同有效容量；mean还改变梯度结构。rank配对只检验相同最大线性映射秩下是否仍存在差异。

不能把这些写成第三个创新：
主干接入优势（C8-O8 APs -0.027865pp）；固定相位排列；全模块无损；推理打乱后的掉点证明位置机制；两个种子证明显著性；VisDrone test-dev是跨数据集验证。SPD后无中间非线性的1×1映射，在压缩特征上可改写为2×2步长2卷积；完整Haar基后自由线性映射亦属等价函数族，不说明训练过程相同。

旧分析已经有的证据：
C4_repeat对历史p05在独立验证中10类APs均正；小目标AP@0.50增益+{ap50:.6f}pp，AP@0.75增益+{ap75:.6f}pp。IoU0.75下<4/4-8/8-16/>=16组净TP分别-19/+77/+183/+4；不能声称越小越有效，也不能把净TP份额当APs贡献。旧分组使用640拉伸后短边，不是新计划sqrt(w*h)。该旧比较的p05也不是本轮新seed1基线，须独立列为诊断证据。

建议投稿前闭合的证据：
1. 所有主表臂统一独立评估：actual best_stg2、相同初始化/配方说明、val与test-dev、相同maxDets/忽略规则/预处理，保留旧C seed1恢复限制。不可与协议不同的外部论文直接按AP排名。
2. 完成C-mean剩余种子和rank16/rank16-mean配对，收齐已排队第三种子。观察配对方向和量级，保留反例；不把三个种子当作显著性或等效保证。
3. SOEP-lite同种子精度/成本比较与结构说明；若没有优势，不写优于近邻，需由受控实证贡献支撑稿件。完整SOEP仍注明未复现。
4. 第二数据集按核实后的划分、忽略区与官方评测，匹配baseline/C/均值控制。若要声称机制可迁移，只有baseline/C两臂不够。
5. 统一尺寸与检出/定位分解：sqrt(w*h)的<4/4-8/8-16/16-32组及GT数，small AP50/AP75，共同/新增/丢失检出及共同检出上的误差分布；文献差异表需覆盖SPD、SOEP、HMF及相关polyphase/抗混叠研究，不能只因两篇没有某对照就称全领域首次。

可选项：
平移分析若不作抗平移主张可作为补充，不能要求结果必须有利于C；P2检测头和第二检测器可后置，收窄为DEIM中的研究；baseline seed2不影响C/C-mean第三配对，但缺少它就不能报告C对baseline的三种子增益。

以上是对这篇稿件的证据要求，不是SIVP规定必须两个数据集或三个种子。期刊官方范围包含图像处理研究与实用工程发展；最终双栏最多10页，第10页只能放参考文献。官方范围：https://link.springer.com/journal/11760/aims-and-scope

完整12指标、现场任务状态、来源SHA和结论分级见同名JSON。本次不新增训练/推理、不改队列、不覆盖已有结果。
'''.format(ap50=old_small['0.50'], ap75=old_small['0.75'])
stem = DEST / 'SIVP投稿证据审查_20261014_v1'
for suffix, content in [('.json', json.dumps(assessment, ensure_ascii=False, indent=2)), ('.txt', text)]:
    with stem.with_suffix(suffix).open('x', encoding='utf-8') as handle:
        handle.write(content)
print(json.dumps({'report': str(stem.with_suffix('.txt')), 'json': str(stem.with_suffix('.json')),
                  'source_hash_checks': len(manifest), 'completed_trajectories': len(index),
                  'checked_los_angeles': snapshot['checked_los_angeles']}, ensure_ascii=False))
