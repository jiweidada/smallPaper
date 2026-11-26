from pathlib import Path
import datetime
import hashlib
import json

read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = read('outputs/stem_native_rank_progress_20261011_v1_inputs.json')
obs = {}
for family, seeds in inputs['observations'].items():
    obs[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf8')
        assert len(raw) == item['transport']['bytes']
        assert hashlib.sha256(raw).hexdigest() == item['transport']['sha256']
        value = json.loads(raw)
        assert value['seed'] == int(seed)
        obs[family][seed] = value
for seed in ('1', '2'):
    assert obs['component'][seed]['code_SHA_verified']
    assert not any(v['failure'] for v in obs['component'][seed]['jobs'].values())
    assert obs['warm'][seed]['source_SHA_verified'] and obs['warm'][seed]['trainer']['same_live']
    assert obs['warm'][seed]['failure'] is None
    assert obs['queue'][seed]['code_SHA_verified'] and obs['queue'][seed]['queue']['same_live']
    assert not obs['queue'][seed]['failures']
    assert obs['collection'][seed]['source_verified'] and obs['collection'][seed]['collector']['same_live']
    assert obs['collection'][seed]['failure'] is None and not obs['collection'][seed]['packets']

analyses = {
    'native_rank': ('stem_native_rank_primary12_20261011_v1', 'analyze.py'),
    'GT_prefix': ('stem_native_rank_GT_prefix12_20261011_v1', 'prefix_GT_transitions.py'),
    'supervision': ('stem_native_supervision_audit12_20261011_v1', 'audit.py'),
}
proofs = {}
reports = {}
for name, (directory, source) in analyses.items():
    root = Path('outputs') / directory
    manifest = read(root / 'manifest.json')
    for filename, digest in manifest['files'].items():
        assert sha(root / filename) == digest
    report = read(root / 'report.json')
    source_directory = 'stem_native_rank_primary12_20261011_v1' if name == 'GT_prefix' else directory
    source_path = Path('英文小论文/code') / source_directory / source
    assert sha(source_path) == report['source_sha256']
    assert report['independent_saved_EMA_verification_pending'] and not report['causal_root_proven']
    assert all(report[k] == 0 for k in ('new_model_forwards', 'new_evaluator_calls', 'new_matching_calls',
                                      'new_optimizer_updates', 'new_training_epochs'))
    proofs[name] = dict(path=str(root / 'report.json'), report_sha256=sha(root / 'report.json'),
        manifest_sha256=sha(root / 'manifest.json'), source_path=str(source_path), source_sha256=sha(source_path),
        status=report['status'])
    reports[name] = report
assert reports['native_rank']['total_native_PR_cells_reproduced'] == 40400
assert reports['native_rank']['maximum_abs_PR_difference'] == 0
assert len(reports['supervision']['actual_primary_bundles']) == 4

root = Path('outputs/stem_native_rank_progress_20261011_v1')
root.mkdir(exist_ok=False)


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


for family, seeds in inputs['observations'].items():
    for seed, item in seeds.items():
        (root / f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root / 'transport.json', {f: {s: i['transport'] for s, i in seeds.items()}
                              for f, seeds in inputs['observations'].items()})
counts = {a: {s: o['jobs'][a]['completed_epochs'] for s, o in obs['component'].items()}
          for a in ('attention_spatial_mean', 'early_final_downsample')}
counts['warm_stem'] = {s: o['completed_epochs'] for s, o in obs['warm'].items()}
for a in ('backbone_silu', 'p2_skip', 'backbone_attention'):
    counts[a] = {s: o['jobs'][a]['completed_epochs'] for s, o in obs['queue'].items()}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
report = dict(status='EXACT_PRIMARY_NATIVE_PR_AND_SAME_GT_PREFIX_ACCOUNTS_PLUS_PINNED_SUPERVISION_AUDIT_COMPLETED',
    local_saved_UTC=now, remote_observation_UTC={f: {s: o['UTC'] for s, o in seeds.items()}
        for f, seeds in obs.items()}, actual_completed_epochs=counts, analysis_provenance=proofs,
    changed_next_action='Separate encoder spatial selection from selected-box regression and final query/class retention. Active MAL already uses IoU**1.5; do not infer a missing quality target. Same-GT prefix shifts have no cross-run query/box identity. Pending exact EMA-bound diagnostics must decide the first observed stage.',
    saved_EMA_independent_replay_pending=True, actual_diagnostic_packets_collected=0,
    formal_BASE12_gate_changed=False, causal_root_proven=False, novelty_proven=False,
    frozen_training_or_model_sources_changed=False, new_BASE_epochs=0, new_BASE_forwards=0,
    existing_remote_process_signals=0, new_model_forwards_here=0, new_evaluator_calls_here=0,
    new_optimizer_updates_here=0, new_training_epochs_here=0,
    current_goal_turn_classification='progress', no_progress_streak=0, goal_complete=False)
write(root / 'report.json', report)
evidence = '../stem_native_rank_progress_20261011_v1/report.json'
for name, (directory, _) in analyses.items():
    write(Path('outputs') / directory / 'execution_state_v1.json', dict(status=proofs[name]['status'],
        evidence=evidence, report_sha256=proofs[name]['report_sha256'], source_sha256=proofs[name]['source_sha256'],
        independent_saved_EMA_verification_pending=True, causal_root_proven=False, goal_complete=False))


def update(directory, old_version, changes):
    directory = Path('outputs') / directory
    old = directory / f'execution_state_v{old_version}.json'
    value = read(old)
    value.update(changes)
    value.update(supersedes=old.name, goal_complete=False)
    write(directory / f'execution_state_v{old_version+1}.json', value)


update('warm_stem12_20261011_v1', 9, dict(evidence=evidence,
    actual_new_training_epochs=counts['warm_stem'], trainers={s: o['trainer'] for s, o in obs['warm'].items()},
    last_remote_observation_UTC={s: o['UTC'] for s, o in obs['warm'].items()}))
update('user_backbone12_20261011_v1', 5, dict(evidence=evidence, saved_local_UTC=now,
    completed_epochs={s: {a: v['completed_epochs'] for a, v in o['jobs'].items()} for s, o in obs['queue'].items()},
    queue_processes={s: o['queue'] for s, o in obs['queue'].items()},
    last_remote_observation_UTC={s: o['UTC'] for s, o in obs['queue'].items()},
    GPU_resource={s: o['live']['resource'] for s, o in obs['queue'].items()}))
update('native_selection_collection12_20261011_v1', 2, dict(evidence=evidence,
    collectors={s: o['collector'] for s, o in obs['collection'].items()}, actual_packets_collected=0,
    last_remote_observation_UTC={s: o['UTC'] for s, o in obs['collection'].items()}))
update('stem_component12_20261011_v1', 28, dict(current_goal_turn='progress', no_progress_streak=0,
    latest_compact_status=evidence, actual_component_completed_epochs={a: counts[a]
        for a in ('attention_spatial_mean', 'early_final_downsample')},
    warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v10.json',
    standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v6.json',
    read_only_diagnostic_collection_state='../native_selection_collection12_20261011_v1/execution_state_v3.json',
    primary_native_score_order='../stem_native_rank_primary12_20261011_v1/report.json',
    primary_same_GT_prefix_transitions='../stem_native_rank_GT_prefix12_20261011_v1/report.json',
    pinned_native_supervision_audit='../stem_native_supervision_audit12_20261011_v1/report.json',
    dense8400_native_regressed_box_pool_absent=True, native_quality_aware_MAL_target_already_present=True,
    missing_quality_loss_as_established_root_rejected=True))
update('original_stem12_20261011_v1', 31,
       dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v29.json'))
print(json.dumps(dict(status=report['status'], actual_completed_epochs=counts,
    PR_cells=40400, PR_difference=0, source_audit_records=4, actual_diagnostic_packets=0,
    independent_saved_EMA_pending=True, goal_complete=False)))
