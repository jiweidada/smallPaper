"""Build a source-bound table input; never execute a model or alter raw observations."""
from pathlib import Path
import json
import hashlib
import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/experiment_table_20261011_v1'
OUT.mkdir(exist_ok=True)
sources = {}


def read(relative):
    p = ROOT / relative
    raw = p.read_bytes()
    sources[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


names = {
    'baseline_pretrained': ('预训练 Stem 基线', '预训练保留', None),
    'native_stem_scratch': ('随机原始 Stem 基线', '随机初始化', None),
    'original_stem': ('你的增强 Stem', '随机初始化', 'native_stem_scratch'),
    'backbone_attention': ('主干注意力替换', '预训练保留', 'baseline_pretrained'),
    'backbone_silu': ('主干 SiLU 替换', '预训练保留', 'baseline_pretrained'),
    'p2_skip': ('原 P2 接入', '预训练保留', 'baseline_pretrained'),
    'p2_lowrank16': ('新 rank16 P2', '预训练保留', 'baseline_pretrained'),
    'highfreq_rgb': ('Stem 高频输入改为 RGB', '随机初始化', 'native_stem_scratch'),
    'detail_lowres': ('Stem 细节分支降分辨率', '随机初始化', 'native_stem_scratch'),
    'attention_spatial_mean': ('Stem 注意力改为空间均值', '随机初始化', 'native_stem_scratch'),
    'early_final_downsample': ('Stem 提前第二次下采样', '随机初始化', 'native_stem_scratch'),
    'warm_stem': ('保留预训练原路的 Stem', '预训练原路+随机分支', 'baseline_pretrained'),
}
metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
records = {}


def add(arm, seed, metrics, source, replay, evidence, note='', checkpoint=''):
    name, initialization, comparator = names[arm]
    assert all(isinstance(metrics[k], (int, float)) for k in metric_names)
    records[arm, seed] = dict(arm=arm, name=name, seed=seed, epochs=12, weights='EMA', initialization=initialization,
        metrics=metrics, comparator=comparator, replay=replay, evidence=evidence, note=note,
        source_relative_path=source, source_sha256=sources[source], checkpoint_sha256=checkpoint)


for seed in (1, 2):
    for arm in ('native_stem_scratch', 'original_stem'):
        path = f'outputs/stem_fixed12_seed{seed}_formal_collection12_20261011_v1/seed{seed}_{arm}/verification.json'
        d = read(path)
        assert d['all548_stage_arrays_reloaded_exact'] is True and d['fixed_endpoint']['epochs'] == 12
        add(arm, seed, d['candidate_overall_percent'], path, '通过', '本地完整产物核验')
        if arm == 'native_stem_scratch':
            add('baseline_pretrained', seed, d['baseline_overall_percent'], path, '已有缓存通过', '复用已有基线产物')

snapshot_path = 'outputs/terminal_actual_snapshot_20261012_013030_v1.json'
snapshot = read(snapshot_path)
for key, capture in snapshot.items():
    raw = capture['raw_payload'].encode()
    assert hashlib.sha256(raw).hexdigest() == capture['transport']['sha256'] and len(raw) == capture['transport']['bytes']
    assert json.loads(raw) == capture['payload']
old_report = read('outputs/reviewed_remote_tests_actual12_20261011_v7/report.json')
for row in old_report['user_fixed12_results']:
    arm, seed = row['arm'], row['seed']
    path = f'outputs/user_backbone_primary_collection12_20261011_v2/seed{seed}_{arm}_v1/primary_verification_v1.json'
    d = read(path)
    metrics = dict(zip(metric_names, [v * 100 for v in d['native_stats'][:6]]))
    assert metrics['AP'] == row['candidate12']['AP'] and metrics['APs'] == row['candidate12']['APs']
    complete = read(str(Path(path).parent / 'extracted/run/training_completion.json').replace('\\', '/'))
    assert complete['completed_epochs'] == 12
    replay = '通过'
    evidence = '本地训练产物已核验；独立重载通过'
    note = ''
    if seed == 1 and arm == 'p2_skip':
        replay = '失败：极小数值差'
        evidence = '本地训练产物已核验；重载未完全一致'
        note = '原始 AP/APs 保留；不将种子1重载判为通过。'
    elif seed == 1:
        actual = snapshot['seed1']['payload']['A_remaining_completion']['results']
        assert any(x['arm'] == arm and x['exit_code'] == 0 and x['verification_status'] == 'PASS_SAVED_CANDIDATE_RELOAD' for x in actual)
        evidence = '本地训练产物已核验；远程首次重载通过，完整重载包待收齐'
    add(arm, seed, metrics, path, replay, evidence, note, d['last_checkpoint_sha256'])

for seed in (1, 2):
    path = f'outputs/p2_lowrank12_formal_collection12_20261011_v1/seed{seed}_p2_lowrank16/verification.json'
    d = read(path)
    assert d['all548_stage_arrays_reloaded_exact'] and d['fixed_endpoint']['epochs'] == 12
    add('p2_lowrank16', seed, d['candidate_overall_percent'], path, '通过', '本地完整产物核验', '相对原 P2，两个种子 APs 均下降。')
    for arm in ('highfreq_rgb', 'attention_spatial_mean', 'early_final_downsample'):
        formal = f'outputs/stem_fixed12_seed{seed}_formal_collection12_20261011_v1/seed{seed}_{arm}/verification.json'
        if (ROOT / formal).exists():
            d = read(formal)
            add(arm, seed, d['candidate_overall_percent'], formal, '通过', '本地完整产物核验')
        else:
            path = f'outputs/stem_component_primary_collection_20261011_v2/seed{seed}_{arm}_v1/primary_verification_v1.json'
            d = read(path)
            complete = read(str(Path(path).parent / 'extracted/run/training_completion.json').replace('\\', '/'))
            assert complete['completed_epochs'] == 12
            add(arm, seed, dict(zip(metric_names, [v * 100 for v in d['native_stats'][:6]])), path, '待收齐核验', '本地训练产物已核验；完整重载包待收齐')
    path = f'outputs/warm_stem_primary_collection12_20261011_v1/seed{seed}_warm_stem_v1/primary_verification_v1.json'
    d = read(path)
    complete = read(str(Path(path).parent / 'extracted/run/training_completion.json').replace('\\', '/'))
    assert complete['completed_epochs'] == 12
    add('warm_stem', seed, dict(zip(metric_names, [v * 100 for v in d['native_stats'][:6]])), path, '待收齐核验', '本地训练产物已核验；完整重载包待收齐', '保留预训练路径，不能用随机原始 Stem 差值归因于新增结构。')

path = 'outputs/stem_fixed12_seed2_formal_collection12_20261011_v1/seed2_detail_lowres/verification.json'
d = read(path)
add('detail_lowres', 2, d['candidate_overall_percent'], path, '通过', '本地完整产物核验')
path = 'outputs/stem_component12_20261011_v1/training_observation_v27.json'
d = read(path)['seed1']['jobs']['detail_lowres']
assert d['records']['training_completion.json']['completed_epochs'] == 12
last = [v for v in d['metrics'] if v['epoch'] == 11]
assert len(last) == 1
add('detail_lowres', 1, dict(zip(metric_names, [v * 100 for v in last[0]['test_coco_eval_bbox'][:6]])), path, '待收齐核验', '已收回12轮日志；完整产物待收齐', '使用固定第12轮日志，不使用最佳轮。')

ordered = [records[arm, seed] for arm in names for seed in (1, 2)]
assert len(ordered) == 24
for row in ordered:
    if row['comparator']:
        baseline = records[row['comparator'], row['seed']]
        row['delta_AP_pp'] = row['metrics']['AP'] - baseline['metrics']['AP']
        row['delta_APs_pp'] = row['metrics']['APs'] - baseline['metrics']['APs']
        row['comparator_name'] = baseline['name']
    else:
        row['delta_AP_pp'] = row['delta_APs_pp'] = None
        row['comparator_name'] = '基线自身'

result = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), records=ordered, sources=sources,
    units='AP metrics are percent scale0-100; differences are percentage points.',
    endpoint='fixed12 EMA,548 validation images; no best-epoch substitution',
    comparison='Stem replacements use random native Stem as primary comparator. Pretrained-Stem experiments retain their initialization-matched comparator.',
    new_training_epochs=0, new_model_forwards=0)
dest = OUT / 'table_inputs.json'
assert not dest.exists()
dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'records': len(ordered), 'sources': len(sources), 'table_input': str(dest)}, ensure_ascii=False))
