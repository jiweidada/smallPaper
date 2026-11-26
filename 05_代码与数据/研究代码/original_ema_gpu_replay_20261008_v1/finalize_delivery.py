"""Preserve code readiness separately from real GPU evidence or authorization."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX = HERE.parent.parent/'data'/'00_实验计划与证据索引'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


tests = json.loads((HERE/'engineering_static.json').read_text(encoding='utf-8'))
assert tests['check_count'] == 38 and tests['status'] == 'PASS_STDLIB_REPLAY_GUARDS_AND_SOURCE_REVIEW'
for name, digest in tests['files_sha256'].items():
    assert sha(HERE/name) == digest
paths = [p for p in HERE.iterdir() if p.is_file()]
receipt = {'at_UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'BOUNDED_UNDEPLOYED_ORIGINAL_EMA_REPLAY_CODE_READY_NOT_RUN',
    'authorization_for_GPU_replay_received': False,
    'authorization_for_training_received': False,
    'pending_question_unchanged': 'Allow original GPU replay only, full six-run training conditional on both gates passing, or no new model computation.',
    'engineering_checks': 38, 'bound_native_source_files': 468,
    'actual_prepare_bundles_created': 0, 'actual_GPU_replay_verified': False,
    'model_forwards': 0, 'checkpoint_loads': 0, 'evaluator_calls': 0,
    'optimizer_or_parameter_updates': 0, 'CUDA_initializations': 0, 'remote_accesses': 0, 'subagents': 0,
    'previous_turn_classification': 'no_new_scientific_progress_scope_preparation_only',
    'current_turn_classification': 'engineering_progress_missing_replay_entrypoint_implemented_and_guard_checked',
    'new_root_cause_or_effectiveness_evidence': False,
    'goal_complete': False,
    'files_sha256': {p.name: sha(p) for p in paths}}
write_new(HERE/'01_交付回执与SHA.json', receipt)
status = {'at_UTC': receipt['at_UTC'], 'status': receipt['status'],
          'replay_delivery': str(HERE/'01_交付回执与SHA.json'),
          'delivery_sha256': sha(HERE/'01_交付回执与SHA.json'),
          'readme': str(HERE/'00_使用说明_原EMA未融合GPU复现入口.txt'),
          'actual_GPU_replay_verified': False, 'new_scientific_evidence': False,
          'pending_question_already_asked_do_not_duplicate': True,
          'scope_unchanged': '2 x 548 original GPU replays; if both pass and fully authorized, original six x 132 training plan.',
          'main_candidate_code_unchanged': 'fine_residual_geometry_20261008_v5',
          'goal_status_expected': 'active_not_complete'}
write_new(INDEX/'单一候选真实验证准入_20261008_v1'/'02_原GPU复现入口已准备_待原问题答复.json', status)
print(json.dumps({'status': receipt['status'], 'checks': 38, 'bound_delivery_files': len(paths),
                  'GPU_replay_executed': False, 'delivery_sha256': sha(HERE/'01_交付回执与SHA.json')}, ensure_ascii=False))
