"""Export a terminal run using files and CPU tensors only; no detector instance."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import os
import sys
import json
import hashlib
import subprocess
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(2)
P = Path('/root/uav_a57_20261102_v1')
E = Path('/root/uav_a54_20261101_v1/A57_exports')
PROTO = 'd233e4a817a997a7a0c4ce203a098e3e0794d0622921fc6333659e374fc13a37'
INIT = '984c6a2413f08caf8db96c869f43a402bfd0449cb3a3dca6b439b73cfe508040'


def load(p):
    return json.loads(p.read_text('utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda: f.read(4 << 20), b''):
            h.update(v)
    return h.hexdigest()


def main():
    sys.path.insert(0, str(P))
    from supervise_A57 import validate, identity, owned_alive
    validate()
    supervisor = load(P / 'supervisor_exit.json')
    assert supervisor == load(P / 'supervisor_state.json')
    assert supervisor['status'] == 'COMPLETED_BOUNDED_SCREEN' and supervisor['exit_code'] == 0
    assert supervisor['cancelled'] == ['M', 'Q'] and supervisor['queue'] == []
    assert supervisor['active_arm'] is None and supervisor['post_training_queue'] is None
    assert supervisor['verdict']['status'] == 'BASELINE_INADEQUATE_SHORT_HORIZON_STOPPED'
    assert supervisor['verdict']['candidate_trained'] is False
    assert not (P / 'arm_M').exists() and not (P / 'arm_Q').exists()
    jobs = [dict(j, current_identity=identity(j['pid']), owned_live=owned_alive(j))
            for j in [supervisor['supervisor']] + supervisor['children']]
    assert {(j['pid'], j['start_ticks']) for j in jobs} == {(17829, 26463967), (17830, 26464019)}
    assert all(j['current_identity'] is None and not j['owned_live'] for j in jobs)
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    assert not gpu.strip()
    arm = P / 'arm_B'
    exit_record = load(arm / 'exit.json')
    assert exit_record['status'] == 'COMPLETED_FIXED12_RAW' and exit_record['exit_code'] == 0
    assert exit_record['model_counter'] == {'entries': 19867, 'completions': 19867,
                                           'decoder_entries_including_recomputation': 235698}
    expected = {'epochs_completed': 12, 'optimizer_updates': 9708, 'optimizer_step_attempts': 9708,
                'train_forward_completions': 19416, 'val_forward_completions': 411,
                'latency_forward_completions': 40}
    assert all(exit_record['counts'][k] == v for k, v in expected.items())
    assert exit_record['counts']['no_nonfinite_skip'] is True
    manifest = load(arm / 'run_manifest.json')
    raw = next(v for v in manifest['checkpoints'] if v['file'] == 'epoch12_RAW.pth')
    optimizer_file = manifest['optimizer']
    for row in [raw, optimizer_file]:
        f = arm / row['file']
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256']
    ck = torch.load(arm / raw['file'], map_location='cpu', weights_only=True)
    initial = torch.load(P / 'common_initialization.pth', map_location='cpu', weights_only=True)['model']
    state = ck['model']
    assert ck['epoch'] == 12 and ck['optimizer_updates'] == 9708 and ck['arm'] == 'B'
    assert ck['protocol_sha256'] == PROTO and ck['initialization_sha256'] == INIT
    assert len(state) == 2011 and state.keys() == initial.keys()
    assert all(bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point())
    shapes = [{'name': k, 'initial': list(initial[k].shape), 'current': list(v.shape)}
              for k, v in state.items() if initial[k].shape != v.shape]
    assert shapes == [{'name': 'encoder.pos_encoding.pe', 'initial': [100, 100, 256], 'current': [160, 160, 256]}]
    assert dict(Counter(int(t) for k, t in state.items() if k.endswith('num_batches_tracked'))) == {19416: 33}
    previous = torch.load(arm / 'epoch08_RAW.pth', map_location='cpu', weights_only=True)
    assert torch.equal(previous['model']['encoder.pos_encoding.pe'], state['encoder.pos_encoding.pe'])
    names = ['encoder.backbone.patch_embed.proj.weight', 'decoder.class_heads.5.weight',
             'decoder.bbox_heads.5.layers.2.weight', 'decoder.reference_point.weight',
             'decoder.tgt_embed.weight', 'decoder.query_pos.weight']
    selected = {k: {'changed_from_initial': not torch.equal(state[k], initial[k]),
                    'max_abs_delta': float((state[k] - initial[k]).abs().max())} for k in names}
    assert all(v['changed_from_initial'] for v in selected.values())
    del previous, initial, ck, state
    opt = torch.load(arm / optimizer_file['file'], map_location='cpu', weights_only=True)
    assert opt['epoch'] == 12 and opt['optimizer_updates'] == 9708 and opt['protocol_sha256'] == PROTO
    optimizer = opt['optimizer']
    steps = Counter(int(row['step']) for row in optimizer['state'].values())
    assert dict(steps) == {9708: exit_record['counts']['active_gradient_parameter_tensors_first_update']}
    assert all(bool(torch.isfinite(t).all()) for row in optimizer['state'].values()
               for t in row.values() if isinstance(t, torch.Tensor) and t.is_floating_point())
    assert all(bool((row['exp_avg_sq'] >= 0).all()) for row in optimizer['state'].values())
    optimizer_audit = {'active_parameter_states': len(optimizer['state']), 'step_histogram': dict(steps),
                       'all_floating_state_tensors_finite': True, 'second_moments_nonnegative': True,
                       'parameter_groups': [{'params': len(g['params']), 'lr': g['lr'],
                                              'weight_decay': g['weight_decay'], 'betas': g['betas']}
                                             for g in optimizer['param_groups']],
                       'CPU_and_CUDA_RNG_state_saved': 'CPU_RNG' in opt and 'CUDA_RNG' in opt}
    del opt, optimizer
    assert not torch.cuda.is_initialized()
    events = [json.loads(line) for line in (arm / 'events.jsonl').read_bytes().splitlines()]
    event_counts = dict(Counter(row['event'] for row in events))
    assert len(events) == 59162
    assert event_counts == {'model_forward_entered': 19867, 'model_forward_completed': 19867,
                            'optimizer_step_attempt': 9708, 'optimizer_step_returned': 9708,
                            'training_epoch_completed': 12}
    assert events[-1]['model_counter'] == exit_record['model_counter']
    assert [row['epoch'] for row in events if row['event'] == 'training_epoch_completed'] == list(range(1, 13))
    audit = {'status': 'PASS_CPU_ONLY_TERMINAL_WEIGHTS_OPTIMIZER_AND_COUNTS',
             'utc': datetime.now(timezone.utc).isoformat(), 'raw_checkpoint': raw,
             'optimizer_checkpoint': optimizer_file, 'all_model_floating_tensors_finite': True,
             'model_state_tensors': 2011, 'shape_differences': shapes, 'BN_counter_histogram': {19416: 33},
             'PE_exactly_equal_to_B8': True, 'selected_tensors': selected,
             'optimizer_audit': optimizer_audit, 'event_counts': event_counts,
             'source_bound_protocol_unchanged': True, 'CUDA_initialized': False,
             'additional_detector_instances': 0, 'additional_model_forwards': 0,
             'additional_optimizer_updates': 0, 'additional_COCO_evaluate_calls': 0}
    snapshot = {'utc': datetime.now(timezone.utc).isoformat(), 'supervisor': supervisor,
                'worker': exit_record, 'jobs': jobs, 'GPU_compute_processes': gpu,
                'only_B_formal_directory_exists': True, 'frozen_inputs_unchanged': True}
    E.mkdir(exist_ok=True)
    archive = E / 'B12_terminal_small_v1.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(arm.iterdir()):
            if f.is_file() and f.suffix != '.pth':
                z.write(f, 'arm_B/' + f.name)
        for n in ['protocol_FROZEN_v1.json', 'pilot_design_FROZEN_v1.json', 'formal_runtime_binding.json',
                  'launch_receipt.json', 'supervisor_exit.json', 'supervisor_state.json',
                  'train_B.log', 'supervisor.log']:
            z.write(P / n, n)
        z.writestr('B12_CPU_terminal_audit_v1.json', json.dumps(audit, indent=2))
        z.writestr('B12_terminal_process_snapshot_v1.json', json.dumps(snapshot, indent=2))
    for row in [raw, optimizer_file]:
        target = E / ('B12_' + row['file'])
        assert not target.exists()
        os.link(arm / row['file'], target)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        files = {i.filename: {'bytes': i.file_size, 'sha256': hashlib.sha256(z.read(i)).hexdigest()}
                 for i in z.infolist()}
    print(json.dumps({'status': 'EXPORTED_TERMINAL_B12_NO_RETRY', 'archive': str(archive),
                      'bytes': archive.stat().st_size, 'sha256': sha(archive), 'files': files,
                      'raw_checkpoint': raw, 'optimizer_checkpoint': optimizer_file,
                      'additional_model_forwards': 0, 'additional_optimizer_updates': 0}), flush=True)


if __name__ == '__main__':
    main()
