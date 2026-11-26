"""Verify downloaded real log prefixes; compare descriptive exposure counts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path.cwd()
P = ROOT / 'outputs/scale_protection12_20261010_v1'
OUT = P / 'live_exposure_prefixes_v1'
TRAIN = ROOT / '英文小论文/code/scale_protection12_20261010_v2'
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def rows(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines()]


def totals(events, profile, seed, arm):
    by_epoch, protected_counts, memberships, seen = {}, {}, {}, set()
    for row in events:
        epoch, iid = row['epoch'], row['image_id']
        assert 4 <= epoch <= 11 and iid in profile and (epoch, iid) not in seen
        seen.add((epoch, iid))
        assert row['arm'] == arm and row['source_GT'] == profile[iid]['raw_valid_GT']
        assert row['risky_fraction'] == profile[iid]['risky_fraction']
        if arm == 'target_protect':
            expected = profile[iid]['protect']
        else:
            if epoch not in memberships:
                ranked = sorted(profile, key=lambda x: hashlib.sha256(
                    f'scale_protection_control:{seed}:{epoch}:{x}'.encode()).digest())
                memberships[epoch] = set(ranked[:995])
            expected = iid in memberships[epoch]
        assert row['applied'] == expected
        assert row['used_GT'] == (row['source_GT'] if expected else row['native_mosaic_GT'])
        assert min(row['source_GT'], row['native_mosaic_GT'], row['used_GT']) >= 0
        c = by_epoch.setdefault(epoch, Counter())
        c.update(events=1, substitutions=int(expected), source_GT=row['source_GT'],
            discarded_native_mosaic_GT=row['native_mosaic_GT'], used_GT=row['used_GT'])
        if expected:
            risky_count = round(profile[iid]['raw_valid_GT'] * profile[iid]['risky_fraction'])
            assert abs(profile[iid]['raw_valid_GT'] * profile[iid]['risky_fraction'] - risky_count) < 1e-8
            c.update(substituted_source_GT=row['source_GT'],
                substituted_source_nominal640_FP32_under4_GT=risky_count,
                substitutions_also_target_rule_positive=int(profile[iid]['protect']))
            protected_counts.setdefault(epoch, Counter())[iid] += 1
    result = {}
    for epoch, c in sorted(by_epoch.items()):
        substitutes, used, discarded = c['substitutions'], c['used_GT'], c['discarded_native_mosaic_GT']
        result[str(epoch)] = dict(c, unique_substituted_source_images=len(protected_counts.get(epoch, {})),
            source_mean_GT_per_substitution=c['substituted_source_GT'] / substitutes if substitutes else None,
            source_nominal640_FP32_under4_fraction=c['substituted_source_nominal640_FP32_under4_GT'] / c['substituted_source_GT'] if c['substituted_source_GT'] else None,
            labels_removed_from_discarded_Mosaic=discarded - used,
            discarded_Mosaic_label_reduction_percent=100 * (discarded - used) / discarded if discarded else None)
    return result


receipts = read(OUT / 'actual_remote_snapshot_receipts.json')
results = {}
for seed in (1, 2):
    remote = receipts[str(seed)]
    assert remote['exit_code'] == 0
    receipt = json.loads(remote['output'])
    archive = OUT / f'seed{seed}_snapshot.zip'
    assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['archive_sha256']
    destination = OUT / f'seed{seed}_snapshot'
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names)) == receipt['files']
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in names)
        assert all((destination / name).resolve().is_relative_to(destination.resolve()) for name in names)
        bundle.extractall(destination)
    manifest = read(destination / 'snapshot_manifest.json')
    assert sha(destination / 'snapshot_manifest.json') == receipt['manifest_sha256']
    assert manifest['seed'] == seed and manifest['identity']['arm'] == 'random_protect'
    assert manifest['source_sha256'] == sha(Path(__file__).with_name('snapshot_live_random_exposure.py'))
    assert manifest['model_forwards'] == manifest['evaluator_calls'] == manifest['training_epochs_added'] == 0
    assert manifest['original_training_files_written'] is False
    assert all(sha(TRAIN / name) == digest for name, digest in manifest['identity']['implementation'].items())
    for row in manifest['files']:
        path = destination / row['relative_path']
        assert path.stat().st_size == row['captured_bytes'] and sha(path) == row['snapshot_sha256']
        assert row['source_read_bytes'] == row['captured_bytes'] + row['incomplete_trailing_bytes_omitted']
        if row['complete_rows'] is not None:
            assert len(rows(path)) == row['complete_rows']
    profile_path = destination / 'training_profile.json'
    assert sha(profile_path) == manifest['identity']['training_profile_sha256']
    profile = {int(i): row for i, row in read(profile_path).items()}
    assert len(profile) == 6471 and sum(row['protect'] for row in profile.values()) == 995
    audits = rows(destination / 'training_audit_epochs.jsonl')
    completed = [row['epoch_zero_based'] for row in audits]
    assert completed == list(range(len(completed)))
    assert all(row['attempts_total'] == row['EMA_updates'] == 1617 * (row['epoch_zero_based'] + 1) for row in audits)
    events = [row for path in sorted((destination / 'augmentation_events').glob('*.jsonl')) for row in rows(path)]
    actual = totals(events, profile, seed, 'random_protect')
    for epoch, row in actual.items():
        row['native_epoch_training_audit_complete_at_snapshot'] = int(epoch) in completed
        row['full_epoch_event_file_coverage_not_claimed'] = True
    target_root = P / f'terminal_collection/seed{seed}_target_protect_v1/extracted'
    target_report_path = P / f'terminal_collection/seed{seed}_target_protect_v1/verification_report_v1.json'
    target_report = read(target_report_path)
    assert target_report['status'] == 'PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
    assert sha(target_root / 'deployment/training_profile.json') == sha(profile_path)
    target_events = [row for path in sorted((target_root / 'run/augmentation_events').glob('*.jsonl')) for row in rows(path)]
    target = totals(target_events, profile, seed, 'target_protect')
    for epoch, row in target.items():
        reference = target_report['augmentation']['Mosaic_events_by_epoch'][epoch]
        assert all(row[key] == reference[key] for key in ('events', 'substitutions', 'source_GT', 'used_GT'))
        assert row['discarded_native_mosaic_GT'] == reference['native_mosaic_GT']
    current_sizes = rows(destination / 'sampled_training_target_sizes.jsonl')
    target_sizes = rows(target_root / 'run/sampled_training_target_sizes.jsonl')
    def check_sizes(items):
        result = {}
        for row in items:
            key = (row['epoch_zero_based'], row['epoch_attempt'])
            assert key not in result and key[0] in (0, 4, 8, 11) and key[1] in (1, 101, 501, 1001, 1501)
            assert row['samples_shape'][0] == 4 and len(row['target_histogram']) == 6
            assert min(row['target_histogram']) >= 0 and sum(row['target_histogram']) == row['target_exposures']
            result[key] = row
        return result
    current_by_key, target_by_key = check_sizes(current_sizes), check_sizes(target_sizes)
    paired_samples = []
    for (epoch, attempt), current in sorted(current_by_key.items()):
        target_row = target_by_key[epoch, attempt]
        paired_samples.append(dict(epoch_zero_based=epoch, epoch_attempt=attempt,
            random=current, target=target_row,
            target_minus_random_total_target_exposures=target_row['target_exposures'] - current['target_exposures'],
            target_minus_random_actual_under4_exposures=sum(target_row['target_histogram'][:2]) - sum(current['target_histogram'][:2]),
            same_source_GT_or_image_identity_not_asserted=True))
    sampled_sums = {}
    for epoch in sorted({row['epoch_zero_based'] for row in paired_samples}):
        selected = [row for row in paired_samples if row['epoch_zero_based'] == epoch]
        arms = {}
        for arm in ('target', 'random'):
            histogram = [sum(row[arm]['target_histogram'][i] for row in selected) for i in range(6)]
            arms[arm] = dict(sampled_batches=len(selected), target_exposures=sum(histogram),
                actual_under4_exposures=sum(histogram[:2]), histogram=histogram)
        sampled_sums[str(epoch)] = arms
    results[str(seed)] = dict(archive_sha256=sha(archive), snapshot_manifest_sha256=sha(destination / 'snapshot_manifest.json'),
        source_snapshot_files_verified=len(manifest['files']), source_formal_training_files_verified=len(manifest['identity']['implementation']),
        native_completed_zero_based_epochs=completed, snapshot_remote_UTC=manifest['observed_end_remote_UTC'],
        actual_random_Mosaic_by_epoch=actual, same_epochs_verified_target_Mosaic={epoch: target[epoch] for epoch in actual},
        sampled_post_collate_targets=paired_samples, sampled_post_collate_sums_by_epoch=sampled_sums,
        all_observed_policy_decisions_and_source_counts_exact=True,
        target_fixed12_verification_report_sha256=sha(target_report_path),
        live_original_processes_verified_before_and_after_snapshot=manifest['processes_after'])
report = dict(status='PASS_DOWNLOADED_LIVE_PREFIX_IDENTITY_POLICY_AND_EXPOSURE_DESCRIPTION', seeds=results,
    source_snapshot_mutable_future_file_not_assumed_equal=True,
    distinctions=[
        'Actual log prefixes are downloaded and individually SHA-verified; current epochs and files may be incomplete.',
        'Training-audit completion is reported separately; it does not prove all event-file coverage for a live snapshot.',
        'Mosaic discarded/used target totals precede sanitation, MixUp, collate and DN.',
        'Substituted-source under4 uses original clipped FP32 nominal640 profile, not native-Mosaic output sizes.',
        'Post-collate samples use actual current input sizes. Five sampled batches are not a population or full-epoch estimate.',
        'Same epoch/attempt does not guarantee same source images, GT identity or pixel tensors across policies.',
        'Changing exposure counts is a mechanism concern; these descriptions are not an efficacy test or unique causal diagnosis.'],
    model_forwards=0, evaluator_calls=0, optimizer_updates=0, training_epochs_added=0,
    acceptance_gates_unchanged=True, fixed12_decision_still_pending=True)
assert 'torch' not in sys.modules and not any(name.startswith('pycocoeval') for name in sys.modules)
write(OUT / 'report.json', report)
lines = ['随机对照实际训练曝光核查：已下载日志前缀，尚不是第12轮完整终点。', '',
    'Mosaic边界计数在Sanitize/MixUp/collate/DN之前；送入模型后的目标尺寸来自稀疏采样，二者分开。']
for seed, result in results.items():
    lines += ['', f"种子{seed}，快照远端UTC：{result['snapshot_remote_UTC']}；完成epoch记录{result['native_completed_zero_based_epochs']}（从0计）。"]
    for epoch, current in result['actual_random_Mosaic_by_epoch'].items():
        target = result['same_epochs_verified_target_Mosaic'][epoch]
        lines.append(f"第{int(epoch)+1}轮random已捕获{current['events']}次Mosaic调用、{current['substitutions']}次替代；原始完整图GT均值{current['source_mean_GT_per_substitution']:.3f}，原始标注名义640下<4px占比{100*current['source_nominal640_FP32_under4_fraction']:.3f}%；丢弃Mosaic标签比选用输出多{current['discarded_Mosaic_label_reduction_percent']:.3f}%。完整target该轮分别为{target['events']}次调用、{target['substitutions']}次替代、源GT均值{target['source_mean_GT_per_substitution']:.3f}、名义<4px占比{100*target['source_nominal640_FP32_under4_fraction']:.3f}%、标签减少{target['discarded_Mosaic_label_reduction_percent']:.3f}%。")
    for epoch, arms in result['sampled_post_collate_sums_by_epoch'].items():
        a, b = arms['target'], arms['random']
        lines.append(f"第{int(epoch)+1}轮已共同记录的{a['sampled_batches']}个采样batch：target实际目标曝光{a['target_exposures']}、实际<4px曝光{a['actual_under4_exposures']}；random {b['target_exposures']}、{b['actual_under4_exposures']}。不是源GT身份配对，也不是全轮比例估计。")
lines += ['', '源标注选择、实际Mosaic替代次数、Mosaic输出标签数和模型输入目标数不是同一量。',
    '继续按两个固定第12轮EMA终点判定效果；不把日志计数或中途曲线当作创新成立。']
with (OUT / '实际训练曝光核查.txt').open('x', encoding='utf8') as stream:
    stream.write('\n'.join(lines) + '\n')
write(OUT / 'manifest.json', dict(files={p.relative_to(OUT).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
    for p in OUT.rglob('*') if p.is_file()}, snapshot_source_sha256=sha(Path(__file__).with_name('snapshot_live_random_exposure.py')),
    analysis_source_sha256=sha(__file__), no_model_or_evaluator_calls=True))
print(json.dumps(dict(status=report['status'], seeds={seed: dict(
    completed=result['native_completed_zero_based_epochs'], Mosaic=result['actual_random_Mosaic_by_epoch'],
    sampled=result['sampled_post_collate_sums_by_epoch']) for seed, result in results.items()},
    manifest_sha256=sha(OUT / 'manifest.json')), ensure_ascii=False), flush=True)
