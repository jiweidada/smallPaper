"""Prepare two newly authorized 24-epoch arms, preserving all old revisions."""
import hashlib
import json
import shutil
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / '英文小论文/code'
STAGE0 = CODE / 'range_context_decoder_20261005_v3'
ADAPTER = CODE / 'range_context_experiments_20261005_v2'
DEST = CODE / 'range_context_experiments_20261005_v4'
EVIDENCE = PROJECT / '英文小论文/data/00_实验计划与证据索引/逐边范围实验_20261005_v1/normalized_screen24_20261005_v4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists() or EVIDENCE.exists():
        raise RuntimeError('Preserve previous preparation')
    EVIDENCE.mkdir()
    DEST.mkdir()
    stage0 = json.loads((STAGE0 / 'stage0_delivery_audit.json').read_text(encoding='utf8'))
    assert stage0['checks_passed'] == 71
    assert all(sha(STAGE0 / row['relative_path']) == row['sha256'] for row in stage0['files'])
    for name in ['bootstrap.py', 'dataset.py', 'dual_range.py', 'dual_view.py', 'factory.py',
                 'geometry.py', 'range_context.py', 'native_recipe.yml', 'native_source_manifest.json',
                 'native_source_manifest_windows_v1.json', 'sealed_source_before.json']:
        shutil.copy2(STAGE0 / name, DEST / name)
    for name in ('native', 'local_deps'):
        shutil.copytree(STAGE0 / name, DEST / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('run_candidate.py', 'evaluate.py', 'preflight.py', 'verify_adapter_cpu.py'):
        shutil.copy2(ADAPTER / name, DEST / name)
    (DEST / 'configs').mkdir()
    arms = ['adaptive_range_global_context', 'mean_size_global_context']
    for variant in arms:
        cfg = {'kind': 'dual_view', 'fusion_width': 64, 'seed': 1, 'epochs': 96,
               'train_batch_size': 8, 'workers': 4, 'tile_size': 640, 'tile_stride': 512,
               'loss_view_weights': [.5, .5], 'merge_iou': .5, 'image_topk': 300,
               'candidate': 'range_context_normalized_screen24_v4', 'protocol': 'protocol.json',
               'variant': variant, 'adaptive_multiplier': 2.0}
        (DEST / 'configs' / (variant + '.yml')).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf8')
    old = json.loads((ADAPTER / 'protocol.json').read_text(encoding='utf8'))
    protocol = {'id': 'range_context_normalized_screen24_20261005_v4',
                'authorization': 'Latest human instruction: 那你 开始测试吧, after the stated plan of real batch8 capacity then two matched 24-epoch screening arms. No automatic 96-epoch extension or old queue resumption.',
                'reviewed_source': {'directory': str(STAGE0), 'audit_sha256': sha(STAGE0 / 'stage0_delivery_audit.json'),
                                    'range_context_sha256': sha(STAGE0 / 'range_context.py'), 'CPU_checks_passed': 71},
                'arms': {'adaptive_range_global_context': 'Mean +/- (q90-q10) per edge; global P4 conditions local P3 weights.',
                         'mean_size_global_context': 'Same full-distribution mean, parameters, normalization and read budget; radius=.25*updated mean box W/H.'},
                'planned_accounts': {'main': ['adaptive_range_global_context'], 'secondary': ['mean_size_global_context']},
                'fixed_recipe': old['fixed_recipe'], 'data_hashes': old['data_hashes'],
                'evaluation_units': old['evaluation_units'], 'capacity_gate': old['capacity_gate'],
                'normalization': 'Null available only when all physical points are invalid; valid weights sum1 per head in both reads.',
                'first_stage': {'epochs': 24, 'schedule_epochs': 96, 'primary_checkpoint': 'epoch_023.pth (fixed round24 EMA)',
                                'primary_metric': 'APs', 'prespecified_mechanism_metric': 'small_AP75',
                                'selection_reference_only': 'Highest overall AP, earliest exact tie; do not substitute for fixed24 endpoint.',
                                'all_epochs_reported': True, 'same_stream_and_initial_tensors_required': True},
                'scientific_decision': {'positive_screen': 'Fixed24 delta APs>0 and small_AP75>0 warrants further review only; report all12 and uncertainty, no effect-size/significance or publication guarantee.',
                                        'uncertain': 'A CI spanning0 or a mixed short-budget direction is inconclusive; do not automatically claim full96 refutation.',
                                        'bootstrap': 'Paired original-image resampling, same indices for both arms, recompute dataset AP; separate from training-seed variability.',
                                        'limits': 'Single seed, two identical RTX3090 environments to be verified. Overall radius scale is not exactly matched; this contrast cannot alone prove per-edge correspondence or useful global semantics.',
                                        'future_controls_not_authorized_now': ['independent', 'adaptive_range_local_context', 'axis_mean_global_context', 'opposite_swap_global_context']},
                'technical_stop': old['technical_stop'],
                'automatic_scope_limit': {'new_formal_arms': 2, 'epochs_per_arm_now': 24, 'full96_automatic': False,
                                          'post24_automatic_queue': False, 'old_queues_resumed': False, 'new_instances': False,
                                          'recipe_changes_on_failure': False},
                'legacy_baseline': 'Completed second-account96 is development reference only; no fresh baseline or reuse reevaluation is queued.'}
    # Remove the inherited ambiguous best-checkpoint selection as primary.
    protocol['fixed_recipe']['selection'] = 'Fixed round24 EMA primary; highest overall AP stored as reference only.'
    (DEST / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf8')
    (EVIDENCE / 'preparation.json').write_text(json.dumps({'source_stage0': str(STAGE0), 'adapter_source': str(ADAPTER),
                                                        'new_code': str(DEST), 'human_authorized_scope': protocol['authorization'],
                                                        'old_sources_untouched': True}, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'code': str(DEST), 'evidence': str(EVIDENCE)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
