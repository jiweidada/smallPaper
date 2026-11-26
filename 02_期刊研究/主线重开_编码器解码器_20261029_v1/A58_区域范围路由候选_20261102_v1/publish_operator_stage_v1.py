from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import os
import shutil

P = Path(__file__).resolve().parent
B = P.parent
OLD = B / 'A57_全网格状态读出_20261102_v1'
R54 = B / 'A54_场景尺度约束候选_20261101_v1'


def load(p):
    return json.loads(p.read_text('utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda: f.read(4 << 20), b''):
            h.update(v)
    return h.hexdigest()


def save(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def replace(p, value):
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, p)


def main():
    cpu = load(P / 'CPU_component_result_v1.json')
    gpu = load(P / 'GPU_component_local_verification_v1.json')
    assert cpu['status'] == 'PASS_SYNTHETIC_OPERATOR_AND_GRADIENT_CHECKS'
    assert gpu['status'] == 'GPU_COMPONENT_BYTES_AND_TERMINAL_STATE_VERIFIED'
    assert cpu['full_detector_forwards'] == gpu['full_detector_forwards'] == 0
    assert sha(P / 'regional_range_attention_v1.py') == cpu['source_sha256']
    transport = P / 'transport_receipts_v1'
    transport.mkdir(exist_ok=False)
    for f in sorted(R54.glob('A58_*_rpc.json')):
        shutil.copy2(f, transport / f.name)
    shutil.copy2(R54 / 'A57_downloads/A58_GPU_component_terminal_v1.zip', P / 'GPU_component_terminal_v1.zip')
    prior_path = OLD / 'A57_B12_terminal_preservation_receipt_v1.json'
    prior = load(prior_path)
    mutable = [B / '01_候选判别表.json', B / '15_当前状态与结果校验.json']
    before = P / 'before_A58_operator_stage'
    before.mkdir(exist_ok=False)
    mapping = {}
    for f in mutable:
        row = prior['files'][str(f)]
        assert sha(f) == row['sha256'] and f.stat().st_size == row['bytes']
        target = before / f.name
        shutil.copy2(f, target)
        mapping[str(f)] = {'snapshot': str(target), 'sha256': sha(target)}
    save(P / 'prior_A57_terminal_resolution.json', {
         'prior_receipt': str(prior_path), 'prior_receipt_sha256': sha(prior_path),
         'reason': 'A58 operator review becomes current; old A57 status resolves through exact immutable snapshots.',
         'files': mapping})
    scope = load(P / '00_候选问题与有限审查范围_v1.json')
    now = datetime.now(timezone.utc).isoformat()
    current = {'phase': 'A58', 'status': 'A58_OPERATOR_PASS_NEAREST_CODE_AND_CAPACITY_PENDING',
               'goal_status': 'active', 'mainline_validated': False, 'novelty_cleared': False,
               'candidate': 'A58_RANGE_QK_P3_CONTEXT', 'selected_for_operator_review': True,
               'selected_for_formal_training': False, 'new_detector_forwards': 0,
               'new_optimizer_updates': 0, 'new_training_queued': False,
               'CPU_module_forwards': 7, 'CPU_synthetic_backwards': 2, 'CPU_route_calls_nested_and_direct': 11,
               'GPU_module_forwards': 2, 'GPU_synthetic_backwards': 2, 'GPU_route_calls_nested_and_direct': 4,
               'operator_GPU_job_terminal': True, 'expected_new_parameters': 148609,
               'report': str(P / '03_候选问题与最小判别方案.txt'),
               'matrix': str(P / '02_最近方法与证据矩阵_v1.json'),
               'hypothesis': scope['question'], 'module_source_sha256': cpu['source_sha256'],
               'baseline': 'ECDet-S pinned author source + known common10class COCOtask initialization; fresh matchedB planned',
               'next': 'Resolve AdaCluster criticalselector overlap with bounded code reads; if distinct, build common initialization and natural batch8 capacity, then freeze3x12 protocol beforeAP. No oldA57 rescue.',
               'no_new_real_detector_instance_yet': True, 'observed_utc': now,
               'limits': 'Synthetic operator checks, not targetbenefit, noveltyclearance or detectorcapacity. No thirdparty code executed.'}
    save(P / 'CURRENT_A58.json', current)
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        d = load(B / name)
        d.update(status=current['status'], A58_execution=current, goal_status='active',
                 current_priority_candidate=current['candidate'], current_priority_question=current['hypothesis'],
                 candidate_selected_for_pilot=False, mainline_selected_for_pilot=False, mainline_validated=False,
                 current_host_candidate='ECDet-S for boundedA58 review; no newfullmodelforward yet',
                 latest_report=current['report'], latest_update_utc=now,
                 pilot_validation_status='A58 synthetic CPU/CUDA operator verified; closestselector code, commoninitialization and realcapacity pending; no newtraining')
        if name.startswith('01_'):
            assert not any(r['id'] == current['candidate'] for r in d['candidates'])
            d['candidates'].append({'id': current['candidate'], 'name': 'P3双侧区域范围路由',
                'status': current['status'], 'question': scope['question'], 'arms': scope['arms'],
                'old_references': scope['history_difference'], 'operator': scope['operator'],
                'selected_for_operator_review': True, 'selected_for_training': False,
                'mainline_validated': False, 'novelty_cleared': False,
                'report': current['report'], 'matrix': current['matrix'], 'training_allocation': 0,
                'evidence_scope': '7CPU/2CUDA moduleforward checks only, not detectorperformance'})
        else:
            assert d['model_forwards'] == d['cap'] == 246151 and d['training_updates'] == 204470
            previous = d.get('current_goal_turn')
            d.update(recorded_utc=now, current_phase='A58_OPERATOR_REVIEW_NO_NEW_DETECTOR_CALLS',
                     previous_goal_turn_classification=previous,
                     current_goal_turn='progress: three mechanisms screened, two-sided region candidate implemented, synthetic CPU/CUDA operator checks passed and preserved; fullmodel/training not started',
                     current_native_job_terminal=True, counts_terminal=True, training_counts_terminal=True,
                     full_forward_cap_closed=True, current_user_task_status='NewA58 candidate concrete; sourceoverlap/capacity pending, mainline notvalidated, no newtraining queue')
            d.setdefault('authoritative_phase_history', []).append({
                'phase': 'A58', 'utc': now, 'classification': 'progress', 'goal_status': 'active',
                'mainline_validated': False, 'new_detector_forwards': 0, 'new_optimizer_updates': 0,
                'CPU_module_forwards': 7, 'GPU_module_forwards': 2, 'report': current['report']})
        replace(B / name, d)
    save(P / 'goal_progress_A58_operator_verified_v1.json', {
         'previous_goal_turn_classification': 'progress: A57 B12 terminalverified, M/Qcancelleduntrained',
         'current_goal_turn_classification': 'progress: concrete newcandidate/source distinctions/operatorimplementation and CPU/CUDA checks',
         'goal_status': 'active', 'mainline_validated': False, 'current': current,
         'global_detector_calls_unchanged': 246151, 'global_optimizer_updates_including_capacity_unchanged': 204470})
    save(P / 'A58_turn_handoff_operator_v1.json', {
         'goal_status': 'active', 'classification': 'progress', 'mainline_validated': False,
         'current': current, 'authenticated_bridge_session': 19671, 'credentials_recorded': False,
         'remote_root': '/root/uav_a58_20261102_v1',
         'remote_GPU_component': 'completed once with2moduleforwards/2backwards; no detectorcalls; do not rerun',
         'remote_transfer_correction': 'First2PUT attempts failed before directory creation; corrected unconfiguredpythonalias and verifiedmkdir beforePUTv3; samepayload executed once. Not an experimentalretry.',
         'native_runtime_note': 'BasePython torch1.12.1/timm0.9.11; esm lackedtimm. Mambaenvironment torch2.7.1+cu126 used for operator; prefer that for freshmatchedB/M/Q after constructor readiness. No packageinstall required yet.',
         'nearest_source_pending': 'AdaCluster repository redirected toUSTC-MLSys/Adacluster; triton_kernel list found; maskhelperread is notcriticalscorer. At most4selectedsourcefiles toresolve overlap, not indefinite noveltyhunt.',
         'old_model_state': 'A57 terminal AP.4245832/APs.1623091; M/Qcancelleduntrained; all oldstates/weights/receipts preserved.',
         'next': current['next']})
    for name, row in prior['files'].items():
        f = Path(mapping[name]['snapshot']) if name in mapping else Path(name)
        assert sha(f) == row['sha256'] and f.stat().st_size == row['bytes'], name
    save(P / 'prior_A57_terminal_receipt_reverified.json', {
         'status': 'PASS', 'files_verified': len(prior['files']), 'resolution': str(P / 'prior_A57_terminal_resolution.json')})
    files = [f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts] + mutable
    save(P / 'A58_operator_preservation_receipt_v1.json', {'status': 'A58_OPERATOR_REVIEW_PRESERVED',
         'goal_status': 'active', 'mainline_validated': False,
         'files': {str(f): {'bytes': f.stat().st_size, 'sha256': sha(f)} for f in sorted(set(files))},
         'new_full_detector_forwards': 0, 'new_optimizer_updates': 0})
    print(json.dumps({'status': current['status'], 'preserved_files': len(files),
                      'global_detector_calls': 246151, 'new_training': False, 'goal_status': 'active'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
