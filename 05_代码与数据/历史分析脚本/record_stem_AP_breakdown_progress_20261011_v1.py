from pathlib import Path
import datetime, hashlib, json
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = read('outputs/stem_AP_breakdown_progress_20261011_v1_inputs.json'); obs = {}
for family, seeds in inputs['observations'].items():
    obs[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf8'); t = item['transport']
        assert len(raw) == t['bytes'] and hashlib.sha256(raw).hexdigest() == t['sha256']
        v = json.loads(raw); assert v['seed'] == int(seed); obs[family][seed] = v
for seed in ('1', '2'):
    assert obs['component'][seed]['code_SHA_verified']
    assert obs['warm'][seed]['source_SHA_verified'] and obs['warm'][seed]['trainer']['same_live'] and not obs['warm'][seed]['failure']
    assert obs['queue'][seed]['queue']['same_live'] and obs['queue'][seed]['code_SHA_verified'] and not obs['queue'][seed]['failures']
    assert obs['collection'][seed]['collector']['same_live'] and obs['collection'][seed]['source_verified']
    assert obs['collection'][seed]['failure'] is None and not obs['collection'][seed]['packets']
analysis = Path('outputs/stem_AP_primary_breakdown12_20261011_v1'); report = read(analysis / 'report.json')
assert report['status'] == 'ACTUAL_PRIMARY12_AP_ARITHMETIC_AND_CATEGORY_STAGE_DESCRIPTION_REPLAY_PENDING'
for name, h in read(analysis / 'manifest.json')['files'].items(): assert sha(analysis / name) == h
assert sha('英文小论文/code/stem_AP_primary_breakdown12_20261011_v1/analyze.py') == report['source_sha256']
assert not report['saved_EMA_independent_replay_verified'] and not report['formal_terminal_reader_passed']
assert report['new_model_forwards'] == report['new_evaluator_calls'] == report['new_optimizer_updates'] == report['new_training_epochs'] == 0
assert [r['seed'] for r in report['seeds']] == [1, 2] and all(len(r['classes']) == 10 for r in report['seeds'])
for r in report['seeds']:
    assert abs(sum(c['contribution_to_aggregate_AP_points']['AP'] for c in r['classes']) - r['full_minus_RGB_points']['AP']) < 1e-9
    assert abs(sum(c['contribution_to_aggregate_AP_points']['APs'] for c in r['classes']) - r['full_minus_RGB_points']['APs']) < 1e-9
car = next(c for c in report['seeds'][0]['classes'] if c['name'] == 'car')
assert car['native_area_specific_TP_delta_by_IoU']['all'][5] == 36
assert car['native_all_area_FP_delta_by_IoU'][5] == -1373 and car['delta_AP_points']['AP75'] < 0
root = Path('outputs/stem_AP_breakdown_progress_20261011_v1'); root.mkdir(exist_ok=False)
def write(path, value):
    with Path(path).open('x', encoding='utf8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
for family, seeds in inputs['observations'].items():
    for seed, item in seeds.items(): (root / f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root / 'transport.json', {f: {s: i['transport'] for s, i in seeds.items()} for f, seeds in inputs['observations'].items()})
counts = {a: {s: o['jobs'][a]['completed_epochs'] for s, o in obs['component'].items()}
          for a in ('attention_spatial_mean', 'early_final_downsample')}
counts['warm_stem'] = {s: o['completed_epochs'] for s, o in obs['warm'].items()}
for a in ('backbone_silu', 'p2_skip', 'backbone_attention'): counts[a] = {s: o['jobs'][a]['completed_epochs'] for s, o in obs['queue'].items()}
result = dict(status='PRIMARY_AP_CELL_BREAKDOWN_COMPLETED_EXISTING_TRAINERS_VERIFIED_LIVE',
    local_saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    remote_observation_UTC={f: {s: v['UTC'] for s, v in seeds.items()} for f, seeds in obs.items()},
    analysis_report_sha256=sha(analysis / 'report.json'), analysis_manifest_sha256=sha(analysis / 'manifest.json'),
    analysis_source_sha256=report['source_sha256'], actual_completed_epochs=counts,
    primary_full_minus_RGB_points=[dict(seed=r['seed'], delta=r['full_minus_RGB_points'], positive_category_AP_contribution=r['positive_class_AP_contribution'],
        negative_category_AP_contribution=r['negative_class_AP_contribution']) for r in report['seeds']],
    changed_next_action='Keep all10 categories in the later six-record selection analysis. Examine car/bus/awning-tricycle and both IoU50/75 as exploratory slices; distinguish actual selected locations from selected-box regression. Do not use aggregate TP/FP or initial geometric support as a substitute for AP/ranking or a unique Stem cause.',
    APs_area_matching_guard='APs TP counts reconstructed from corresponding native area-specific recall, not from all-area GT flags stratified by size.',
    exact_AP_cell_arithmetic_verified=True, independent_saved_EMA_verification_pending=True,
    causal_root_proven=False, novelty_proven=False, baseline_gate_changed=False,
    actual_diagnostic_packets=0, frozen_training_code_changed=False, existing_remote_process_signals=0,
    new_BASE_epochs=0, new_BASE_forwards=0, new_model_forwards_here=0, new_optimizer_updates_here=0, new_training_epochs_here=0,
    previous_goal_turn_classification='progress', current_goal_turn_classification='progress', no_progress_streak=0, goal_complete=False)
write(root / 'report.json', result)
evidence = '../stem_AP_breakdown_progress_20261011_v1/report.json'
write(analysis / 'execution_state_v1.json', dict(status=report['status'], evidence=evidence, report_sha256=result['analysis_report_sha256'],
    all10_categories_and_both_seeds_retained=True, native_area_specific_TP_from_recall=True, exact_metric_cell_arithmetic=True,
    independent_saved_EMA_verification_pending=True, goal_complete=False))
def update(old, new, changes):
    value = read(old); value.update(changes); value['supersedes'] = Path(old).name; value['goal_complete'] = False; write(new, value)
update('outputs/warm_stem12_20261011_v1/execution_state_v8.json', 'outputs/warm_stem12_20261011_v1/execution_state_v9.json',
    dict(evidence=evidence, actual_new_training_epochs=counts['warm_stem'], trainers={s: o['trainer'] for s, o in obs['warm'].items()}))
update('outputs/user_backbone12_20261011_v1/execution_state_v4.json', 'outputs/user_backbone12_20261011_v1/execution_state_v5.json',
    dict(evidence=evidence, queue_processes={s: o['queue'] for s, o in obs['queue'].items()},
        last_remote_observation_UTC={s: o['UTC'] for s, o in obs['queue'].items()}, GPU_resource={s: o['live']['resource'] for s, o in obs['queue'].items()}))
update('outputs/native_selection_collection12_20261011_v1/execution_state_v1.json', 'outputs/native_selection_collection12_20261011_v1/execution_state_v2.json',
    dict(evidence=evidence, collectors={s: o['collector'] for s, o in obs['collection'].items()}, actual_packets_collected=0))
update('outputs/stem_component12_20261011_v1/execution_state_v27.json', 'outputs/stem_component12_20261011_v1/execution_state_v28.json',
    dict(current_goal_turn='progress', no_progress_streak=0, latest_compact_status=evidence,
        actual_component_completed_epochs={a: counts[a] for a in ('attention_spatial_mean', 'early_final_downsample')},
        warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v9.json',
        standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v5.json',
        read_only_diagnostic_collection_state='../native_selection_collection12_20261011_v1/execution_state_v2.json',
        primary_AP_breakdown='../stem_AP_primary_breakdown12_20261011_v1/report.json',
        TP_FP_count_or_geometry_as_AP_proxy_rejected_by_actual_numbers=True))
update('outputs/original_stem12_20261011_v1/execution_state_v30.json', 'outputs/original_stem12_20261011_v1/execution_state_v31.json',
    dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v28.json'))
print(json.dumps(dict(status=result['status'], completed_epochs=counts, saved_EMA_pending=True, goal_complete=False)))
