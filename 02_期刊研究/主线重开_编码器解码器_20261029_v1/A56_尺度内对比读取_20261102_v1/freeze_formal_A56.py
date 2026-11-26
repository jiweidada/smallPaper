from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone

P = Path(__file__).resolve().parent
B = P.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (P/'protocol_FROZEN_v1.json').exists()
    capacity = json.loads((P/'capacity_result.json').read_text('utf-8'))
    exit_record = json.loads((P/'capacity_exit.json').read_text('utf-8'))
    assert capacity['status'] == 'PASS' and exit_record['child_exit_code'] == 0
    assert capacity['whole_model_forwards'] == 9 and capacity['capacity_only_optimizer_updates'] == 6
    assert all(r['initial_FP32_output_bitwise_equal'] for r in capacity['results'])
    assert len(set(r['new_initial_tensors_sha256'] for r in capacity['results'])) == 1
    first = json.loads((P/'capacity_protocol_FROZEN_v1.json').read_text('utf-8'))
    assert capacity['protocol_sha256'] == sha(P/'capacity_protocol_FROZEN_v1.json')
    for name, value in first['files_sha256'].items():
        assert sha(P/name) == value
    for name in ['train_A56.py', 'launch_formal_A56.py', 'supervise_A56.py']:
        ast.parse((P/name).read_text('utf-8'))
    previous = json.loads((B/'A54_场景尺度约束候选_20261101_v1'/'protocol_FROZEN_v1.json').read_text('utf-8'))
    files = {n: sha(P/n) for n in [*first['files_sha256'], 'train_A56.py',
                                  'launch_formal_A56.py', 'supervise_A56.py']}
    plan = {**first['formal_draft_only'], 'phase': 'A56', 'stage': 'FORMAL_FIXED12',
            'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'goal_status': 'active', 'mainline_confirmed': False, 'novelty_cleared': False,
            'candidate': first['candidate'], 'host': first['host'], 'dataset': first['dataset'],
            'B': first['B'], 'M': first['M'], 'Q': first['Q'],
            'sampling': first['sampling'], 'placement': first['placement'], 'gain': first['gain'],
            'new_initialization': first['new_initialization'],
            'new_residual_precision': first['new_residual_precision'],
            'model_parameters_each_M_Q': 9877256, 'extra_parameters_each_M_Q': 74144,
            'baseline_reference': previous['baseline_reference'],
            'baseline_outcomes_already_visible': previous['baseline_outcomes_already_visible'],
            'baseline_reuse_conditions': ['identical native initial checkpoint', 'all12 epoch data orders',
                                         'all12 full-epoch target/ignore metadata hashes',
                                         'first8 actual pixel/target hashes per epoch',
                                         'first8 pre-forward RNG hashes per epoch',
                                         'same optimizer policy, batch8, BF16, 12x809 updates and evaluator'],
            'not_claimed': ['all-image pixel hash equality from first8 checks', 'blind validation',
                            'first differential detector', 'pure background reads', 'full-model offset invariance',
                            'causally established native bottleneck', 'publication-ready novelty'],
            'precision': first['precision'],
            'evaluation': 'unchanged source-aligned ignore handling and COCO evaluator; public val548; EMA at4/8/12 recorded, only12 decides',
            'secondary_outputs': ['AP75, APs75', 'cached R50/R75 for original-small GT with input shortside4to8',
                                  'per-image and per-class changes', 'unchanged single-image latency function; no forward diagnostic hooks'],
            'formal_optimizer_updates_each': 9708,
            'whole_model_forwards_each_including_eval_and_latency': 10159,
            'wall_cap_each_hours': 24, 'post_Q_queue': None,
            'failure_rule': 'stop failed process, preserve evidence, cancel unstarted arm; no automatic restart/rescue sweep',
            'interpretation': first['interpretation'],
            'capacity_protocol_sha256': sha(P/'capacity_protocol_FROZEN_v1.json'),
            'capacity_result_sha256': sha(P/'capacity_result.json'),
            'capacity_exit_sha256': sha(P/'capacity_exit.json'),
            'expected_M_Q_new_initial_tensors_sha256': capacity['results'][0]['new_initial_tensors_sha256'],
            'files_sha256': files, 'external_sha256': first['external_sha256']}
    (P/'protocol_FROZEN_v1.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'FORMAL_FIXED12_FROZEN',
                      'protocol_sha256': sha(P/'protocol_FROZEN_v1.json'),
                      'new_arms': ['M', 'Q'], 'epochs_each': 12,
                      'fixed_endpoint': plan['fixed_endpoint'], 'gates': plan['primary_gates_pp']}))


if __name__ == '__main__':
    main()
