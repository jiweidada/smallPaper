"""Bind the completed diagnostic receipts and locally verified evidence."""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import os
import backup_guard as guard

WORK = guard.REMOTE_WORK
HERE = Path(__file__).resolve().parent
FINAL = HERE / 'remote_final'
ROOT = WORK / 'diagnostic_backups_20261105' / guard.RUN_ID
OUTPUT = WORK / 'diagnostic_completion_receipt_20261105.json'
EXPECTED_NAMES = ['initial_ema', 'initial_raw'] + ['epoch_%04d' % n for n in range(214, 225)] + ['terminal_raw']

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    guard.check_plans()
    assert not OUTPUT.exists() and not OUTPUT.is_symlink()
    result = read(FINAL / 'diagnostic_result.json')
    state = read(FINAL / 'controller_state.json')
    status_wrapper = read(HERE / 'remote_final_status.json')
    status = json.loads(status_wrapper['stdout'])
    assert status_wrapper['exit_code'] == 0 and status['controller_state'] == state
    assert state['status'] == result['status'] == 'DIAGNOSTIC_COMPLETE_NOT_A_FAIR_PAIR'
    assert not status['controller_process'].get('identity_matches')
    assert not status['child_process'].get('identity_matches')
    assert guard.digest_file(FINAL / 'diagnostic_result.json') == state['result_sha256']
    assert result['training']['status'] == 'MAX_EPOCHS_STOP' and result['training']['epoch_index'] == 224
    assert result['optimizer_updates'] == 17787 and result['full_validation_passes'] == 14
    assert result['images_per_validation'] == 548 and result['automatic_retries'] is False
    audit = read(WORK / 'diagnostic_saved_file_audit_complete_20261105.json')
    steps = read(WORK / 'diagnostic_saved_steps_audit_20261105.json')
    tensors = read(WORK / 'diagnostic_saved_state_inspection_final_20261105.json')
    trajectory = read(WORK / 'diagnostic_analysis_final_20261105/trajectory.json')
    assert audit['status'] == 'DIAGNOSTIC_SAVED_EVIDENCE_VERIFIED' and audit['mode'] == 'complete'
    assert [x['phase'] for x in audit['phases']] == EXPECTED_NAMES
    assert all(x['max_array_summary_difference'] == 0 for x in audit['phases'])
    assert steps['status'] == 'DIAGNOSTIC_SAVED_STEPS_VERIFIED' and steps['total_learning_rate_rows'] == 17787
    assert len(steps['epochs']) == 11 and all(x['gradient_nonfinite_count'] == 0 for x in steps['epochs'])
    assert len(tensors['epochs']) == 12
    assert all(x[g]['nonfinite_elements'] == 0 for x in tensors['epochs'] for g in ['raw_model', 'ema_model', 'optimizer'])
    assert all(x[g]['negative_running_variance_elements'] == 0 for x in tensors['epochs'] for g in ['raw_model', 'ema_model'])
    assert [x['phase'] for x in trajectory['new_diagnostic']] == EXPECTED_NAMES
    events = [json.loads(line) for line in (WORK / 'diagnostic_backups_20261105/monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
    confirmed = [x for x in events if x['status'] == 'BACKUP_VERIFIED_AND_ACKNOWLEDGED']
    assert [x['phase'] for x in confirmed] == EXPECTED_NAMES
    assert events[-1]['status'] == 'CONTROLLER_FINISHED'
    phases = []
    for name in EXPECTED_NAMES:
        guard.read_ack_for_previous(ROOT, name)
        metric = read(ROOT / name / 'metrics.json')
        manifest = guard.parse_manifest((ROOT / name / 'manifest.json').read_bytes())
        assert math.isfinite(metric['metrics']['AP'])
        phases.append({'phase': name, 'epoch_index': manifest['epoch_index'],
                       'optimizer_updates': manifest['optimizer_updates'], 'validation_passes': manifest['validation_passes'],
                       'AP': metric['metrics']['AP'], 'manifest_sha256': guard.digest_file(ROOT / name / 'manifest.json'),
                       'checkpoint_sha256': metric['checkpoint_sha256']})
    assert phases[0]['AP'] == result['initial_ema']['metrics']['AP']
    assert phases[1]['AP'] == result['initial_raw']['metrics']['AP']
    assert phases[-2]['AP'] == result['training']['final_ema_AP']
    assert phases[-1]['AP'] == result['terminal_raw']['metrics']['AP']
    assert phases[-2]['checkpoint_sha256'] == phases[-1]['checkpoint_sha256']
    assert all(phases[0]['AP'] - x['AP'] < 0.05 for x in phases[2:-1])
    evidence = [FINAL / 'diagnostic_result.json', FINAL / 'controller_state.json',
                WORK / 'diagnostic_saved_file_audit_complete_20261105.json',
                WORK / 'diagnostic_saved_steps_audit_20261105.json',
                WORK / 'diagnostic_saved_state_inspection_final_20261105.json',
                WORK / 'diagnostic_analysis_final_20261105/trajectory.json',
                WORK / 'diagnostic_analysis_final_20261105/trajectory.png',
                WORK / 'diagnostic_backups_20261105/monitor_events.jsonl']
    output = {'status': 'ONE_AUTHORIZED_DIAGNOSTIC_COMPLETED_AND_EVIDENCE_VERIFIED',
              'recorded_local_utc': datetime.now(timezone.utc).isoformat(), 'run_id': guard.RUN_ID,
              'phase_count': len(phases), 'optimizer_updates': 17787, 'full_validation_passes': 14,
              'phases': phases, 'evidence': [{'path': str(p.relative_to(WORK)), 'sha256': guard.digest_file(p)} for p in evidence],
              'scope': 'One reset-RNG diagnostic trajectory only; neither exact replay nor a fair BASE/FULL comparison; cause remains unresolved.',
              'model_inference_passes_in_audit': 0, 'evaluator_reruns_in_audit': 0, 'optimizer_updates_in_audit': 0}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({'status': output['status'], 'phase_count': len(phases), 'receipt': str(OUTPUT)}))

if __name__ == '__main__':
    main()
