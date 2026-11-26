"""Summarize the completed conditional stem comparison from audited artifacts."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = HERE.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


pair_path = REMOTE / '断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
lineage_path = HERE / 'NO_MPS/lineage_through_round320.json'
shared_path = HERE / 'module_shared_conditions_verified_20261108.json'
pair, lineage, shared = read(pair_path), read(lineage_path), read(shared_path)
assert pair['status'] == 'RESUMED_PAIR_ENDPOINTS_VERIFIED_PENDING_MANUSCRIPT_REVIEW'
assert lineage['status'] == 'VERIFIED_ABLATION_LINEAGE_AND_SAVED_ENDPOINTS'
assert lineage['arm'] == 'NO_MPS' and lineage['completed_rounds'] == 320
assert lineage['learning_rates']['recorded_attempts'] == 517440
assert lineage['learning_rates']['maximum_absolute_difference'] == 0
assert lineage['analysis_script_sha256'] == digest(HERE / 'audit_ablation_lineage.py')
assert shared['status'] == 'FROZEN_REMOVAL_CONDITIONS_AND_INITIAL_TENSORS_VERIFIED'
assert shared['analysis_script_sha256'] == digest(HERE / 'audit_module_shared_conditions.py')
assert lineage['protocol_sha256'] == shared['arms']['NO_MPS']['protocol_sha256']
backup_root = Path(lineage['backup_root'])
receipt_path = backup_root / 'completed_run_receipt.json'
supervisor_path = backup_root / 'completed_supervisor.json'
receipt, supervisor = read(receipt_path), read(supervisor_path)
assert receipt['status'] == supervisor['status'] == 'COMPLETED'
assert supervisor['exit_code'] == 0 and supervisor['final_receipt'] == receipt
assert receipt['arm'] == 'NO_MPS' and receipt['completed_epochs'] == 320
assert receipt['optimizer_step_attempts'] == 517440
comparisons = {}
for role, title in [('terminal', 'primary_fixed_round320'), ('best', 'secondary_highest_AP')]:
    full = pair['lineages']['FULL']['endpoints'][role]
    removal = lineage['endpoints'][role]
    assert full['role'] == removal['role'] == role
    assert full['validation_images'] == removal['validation_images'] == 548
    assert full['predictions'] == removal['predictions'] == 164400
    if role == 'terminal':
        assert full['epoch_index'] == removal['epoch_index'] == 319
    assert full['metrics'].keys() == removal['metrics'].keys()
    assert all(math.isfinite(value) and 0 <= value <= 1
               for endpoint in [full, removal] for value in endpoint['metrics'].values())
    comparisons[title] = {
        'rounds': {'FULL': full['epoch_index'] + 1, 'NO_MPS': removal['epoch_index'] + 1},
        'fractions': {'FULL': full['metrics'], 'NO_MPS': removal['metrics']},
        'absolute_percent': {arm: {key: 100 * value for key, value in endpoint['metrics'].items()}
                             for arm, endpoint in [('FULL', full), ('NO_MPS', removal)]},
        'FULL_minus_NO_MPS_percentage_points': {key: 100 * (full['metrics'][key] - removal['metrics'][key])
                                               for key in full['metrics']},
        'endpoint_checkpoint_sha256': {'FULL': full['checkpoint_sha256'], 'NO_MPS': removal['checkpoint_sha256']}}
parameters = {'FULL': pair['shared_conditions']['trainable_parameters']['FULL'],
              'NO_MPS': shared['arms']['NO_MPS']['trainable_parameters']}
report = {
    'status': 'NO_MPS_ENDPOINT_COMPARISON_VERIFIED_NO_MSCA_PENDING',
    'observed_local_utc': datetime.now(timezone.utc).isoformat(),
    'comparisons': comparisons, 'trainable_parameters': parameters,
    'FULL_minus_NO_MPS_trainable_parameters': parameters['FULL'] - parameters['NO_MPS'],
    'NO_MPS_best_and_primary_coincide': lineage['best_epoch_index'] == 319,
    'NO_MPS_late_restore_count': lineage['restore_event_count'],
    'NO_MPS_final_rolling_state_restored': lineage['rolling_state_restored_after_evaluation'],
    'evidence_sha256': {str(path.relative_to(REMOTE)): digest(path)
                        for path in [pair_path, lineage_path, shared_path]},
    'completion_receipt_sha256': digest(receipt_path),
    'completion_supervisor_sha256': digest(supervisor_path),
    'scope': 'One conditional stem replacement with the other FULL settings present. FULL continued after round 1, whereas NO_MPS started at round zero. The results include each configuration\'s realized stochastic history and do not estimate significance or seed variability.',
    'pending_arms': ['NO_MSCA'], 'model_forwards': 0, 'evaluator_runs': 0,
    'submission_ready': False}
output = HERE / 'NO_MPS_FULL_endpoint_comparison_20261108.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
state_path = HERE / 'current_execution_state.json'
state = read(state_path)
state['NO_MPS'].update(status='COMPLETED_AND_BACKED_UP', completed_rounds=320,
                        last_accepted_round=320, independent_lineage_audit_round=320,
                        terminal_and_best_endpoints_verified=True,
                        endpoint_comparison=str(output), endpoint_comparison_sha256=digest(output))
state['observed_local_utc'] = report['observed_local_utc']
temporary = state_path.with_suffix('.nomps_completion.tmp')
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
temporary.replace(state_path)
print(json.dumps({'status': report['status'], 'primary': comparisons['primary_fixed_round320'],
                  'trainable_parameters': parameters, 'submission_ready': False}))
