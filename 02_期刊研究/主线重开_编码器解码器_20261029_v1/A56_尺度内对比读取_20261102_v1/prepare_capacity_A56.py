"""Freeze one capacity allocation. This file does not launch it or approve a paper claim."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone

P = Path(__file__).resolve().parent
B = P.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = P / 'capacity_protocol_FROZEN_v1.json'
    assert not destination.exists()
    cpu = json.loads((P / '02_CPU算子核验_v1.json').read_text('utf-8'))
    assert cpu['status'] == 'PASS' and cpu['whole_detector_forwards'] == 0
    assert cpu['implementation_sha256'] == sha(P / 'contrastive_read.py')
    for path in P.glob('*.py'):
        ast.parse(path.read_text('utf-8-sig'))
    before = P / 'before_A56'
    before.mkdir()
    for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
        shutil.copy2(B / name, before / name)
    source_plan = json.loads((B / 'A54_场景尺度约束候选_20261101_v1' /
                              'protocol_FROZEN_v1.json').read_text('utf-8'))
    files = ['contrastive_read.py', 'runtime_A56.py', 'capacity_A56.py',
             'launch_capacity_A56.py', 'supervise_capacity_A56.py']
    plan = {
        'phase': 'A56', 'stage': 'CAPACITY_ONLY', 'client_date': '2026-11-02',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'candidate': 'native read plus level-mass-matched contrast residual',
        'mainline_confirmed': False, 'novelty_cleared': False, 'goal_status': 'active',
        'dataset': 'VisDrone2019DET train only; 6471 images; 343204 valid GT',
        'host': 'original ECDet-S, pinned EdgeCrafter d8752b75c9beaecfcdd0e5cd207bae3b6946ae23',
        'native_parameters': 9803112, 'extra_parameters_each_M_Q': 74144,
        'placement': 'all four existing decoder cross-attention reads; ordinary and DN queries use the same operator',
        'B': 'unmodified native model',
        'M': 'retain native output; add gated globally normalized differential read',
        'Q': 'identical parameters/points/gate; negative mass matches positive mass separately per level',
        'sampling': 'native read unchanged; additional M/Q residual both use border sampling at the native locations',
        'gain': 'per-head tanh gate; zero initialization; no extra loss',
        'new_initialization': 'forked CPU RNG seed20261112+layer; identical M/Q tensors; native RNG restored',
        'new_residual_precision': 'FP32 softmax, signed weighting and grid sampling; cast only when adding to native output',
        'batches': 'batch8; fixed seed20261101; first batch, greatest maximum-GT batch, greatest total-GT distinct batch',
        'precision': 'FP32 initial eval; BF16 train forwards with native FP32 criterion',
        'max_whole_detector_forwards': 9, 'max_temporary_optimizer_updates': 6,
        'schedule': 'B/M/Q one eval forward each; M/Q three temporary training steps each',
        'checks': ['all original native initial tensors identical', 'M/Q new tensors identical',
                   'whole-model logits and boxes bitwise identical at zero gate', 'all native losses finite',
                   'new gates and negative-map gradients live after the initial zero-gate step',
                   'actual batch8 peak allocation below24GiB'],
        'failure_rule': 'stop and preserve error; no automatic retry or formal training',
        'updated_states': 'discard every temporary state; any future formal arm starts fresh',
        'formal_training_queued': False,
        'formal_draft_only': {'baseline': 'reuse A49 native B12 only if all frozen data/RNG/init/training settings match',
                              'new_arms': ['M', 'Q'], 'epochs_each': 12, 'fixed_endpoint': 'epoch12 EMA',
                              'seed': 20261101, 'batch': 8, 'input': '640x640 stretch',
                              'augmentation': 'native frozen resize+horizontalflip; no Mosaic/crop',
                              'primary_gates_pp': {'Q_minus_B_APs_at_least': .5,
                                                   'Q_minus_M_APs_at_least': .3,
                                                   'Q_minus_B_AP_at_least': -.2},
                              'no_rescue': 'no extra seed/epochs/layer-site/gain/point/normalization sweep after endpoint'},
        'interpretation': 'Capacity proves neither task benefit nor novelty. Level-offset cancellation is a fixed-query residual property, not whole-model invariance or proof of a pure background estimate.',
        'files_sha256': {name: sha(P / name) for name in files},
        'external_sha256': source_plan['external_sha256'],
        'CPU_component_receipt_sha256': sha(P / '02_CPU算子核验_v1.json'),
    }
    destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    (P/'CURRENT_A56.json').write_text(json.dumps(
        {'phase': 'A56', 'status': 'CPU_OPERATOR_PASS_CAPACITY_FROZEN_NOT_LAUNCHED',
         'goal_status': 'active', 'mainline_confirmed': False, 'novelty_cleared': False,
         'selected_for_formal_training': False, 'formal_training_queued': False,
         'whole_detector_forwards_new': 0, 'detector_optimizer_updates_new': 0,
         'synthetic_native_attention_calls': 7, 'synthetic_component_optimizer_steps': 4,
         'capacity_protocol_sha256': sha(destination)}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'CAPACITY_FROZEN', 'protocol_sha256': sha(destination),
                      'own_frozen_files': len(files), 'external_frozen_files': len(plan['external_sha256'])}))


if __name__ == '__main__':
    main()
