"""One-off CPU/file-only export. Never construct or run a detector."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json
import hashlib
import os
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(2)

P = Path('/root/uav_a57_20261102_v1')
E = Path('/root/uav_a54_20261101_v1/A57_exports')


def load(p):
    return json.loads(p.read_text('utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda: f.read(4 << 20), b''):
            h.update(v)
    return h.hexdigest()


def main():
    import sys
    sys.path.insert(0, str(P))
    from supervise_A57 import validate, identity, owned_alive
    validate()
    s = load(P / 'supervisor_state.json')
    jobs = [dict(j, current_identity=identity(j['pid']), owned_live=owned_alive(j))
            for j in [s['supervisor']] + s['children']]
    assert s['status'] == 'RUNNING' and s['active_arm'] == 'B'
    assert all(j['owned_live'] for j in jobs)
    assert {(j['pid'], j['start_ticks']) for j in jobs} == {(17829, 26463967), (17830, 26464019)}
    assert not (P / 'arm_M').exists() and not (P / 'arm_Q').exists()
    arm = P / 'arm_B'
    manifest = load(arm / 'run_manifest.json')
    cp = next(c for c in manifest['checkpoints'] if c['file'] == 'epoch08_RAW.pth')
    weight = arm / cp['file']
    assert weight.stat().st_size == cp['bytes'] and sha(weight) == cp['sha256']
    ck = torch.load(weight, map_location='cpu', weights_only=True)
    initial = torch.load(P / 'common_initialization.pth', map_location='cpu', weights_only=True)
    init = initial['model']
    current = ck['model']
    assert ck['epoch'] == 8 and ck['optimizer_updates'] == 6472 and ck['arm'] == 'B'
    assert ck['protocol_sha256'] == 'd233e4a817a997a7a0c4ce203a098e3e0794d0622921fc6333659e374fc13a37'
    assert ck['initialization_sha256'] == '984c6a2413f08caf8db96c869f43a402bfd0449cb3a3dca6b439b73cfe508040'
    assert set(init) == set(current) and len(current) == 2011
    assert all(bool(torch.isfinite(t).all()) for t in current.values() if t.is_floating_point())
    shapes = [{'name': k, 'initial': list(init[k].shape), 'current': list(v.shape)}
              for k, v in current.items() if init[k].shape != v.shape]
    assert shapes == [{'name': 'encoder.pos_encoding.pe', 'initial': [100, 100, 256], 'current': [160, 160, 256]}]
    selected_names = ['encoder.backbone.patch_embed.proj.weight', 'decoder.class_heads.5.weight',
                      'decoder.bbox_heads.5.layers.2.weight', 'decoder.reference_point.weight',
                      'decoder.tgt_embed.weight', 'decoder.query_pos.weight']
    selected = {k: {'changed_from_initial': not torch.equal(current[k], init[k]),
                    'max_abs_delta': float((current[k] - init[k]).abs().max())}
                for k in selected_names}
    assert all(v['changed_from_initial'] for v in selected.values())
    bn = Counter(int(t) for k, t in current.items() if k.endswith('num_batches_tracked'))
    assert dict(bn) == {12944: 33}
    previous = torch.load(arm / 'epoch04_RAW.pth', map_location='cpu', weights_only=True)
    assert torch.equal(previous['model']['encoder.pos_encoding.pe'], current['encoder.pos_encoding.pe'])
    assert not torch.cuda.is_initialized()
    audit = {'status': 'PASS_CPU_DATA_ONLY_CHECKPOINT_AUDIT',
             'utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': cp,
             'epoch': 8, 'optimizer_updates': 6472, 'state_tensors': len(current),
             'all_floating_tensors_finite': True, 'shape_differences': shapes,
             'positional_cache_exactly_equal_to_B4': True, 'BN_counter_histogram': dict(bn),
             'selected_tensors': selected, 'detector_instances_constructed': 0,
             'additional_model_forwards': 0, 'additional_optimizer_updates': 0,
             'additional_COCO_evaluate_calls': 0, 'CUDA_initialized': False}
    del previous, initial, init, ck, current
    events = []
    for line in (arm / 'events.jsonl').read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):
            break
        row = json.loads(line)
        events.append(line)
        if row['event'] == 'model_forward_completed' and row['model_counter']['completions'] == 13218:
            break
    rows = [json.loads(line) for line in events]
    assert len(rows) == 39388
    counts = Counter(row['event'] for row in rows)
    assert counts == {'model_forward_entered': 13218, 'model_forward_completed': 13218,
                      'optimizer_step_attempt': 6472, 'optimizer_step_returned': 6472,
                      'training_epoch_completed': 8}
    assert rows[-1]['model_counter'] == {'entries': 13218, 'completions': 13218,
                                        'decoder_entries_including_recomputation': 156972}
    assert [row['epoch'] for row in rows if row['event'] == 'training_epoch_completed'] == list(range(1, 9))
    names = ['epoch08_RAW_metrics.json', 'epoch08_RAW_precision_recall.npz',
             'epoch08_RAW_predictions.json', 'epoch08_RAW_GT.json', 'epoch08_RAW_COCO_stdout.txt',
             'epoch08_train.json', 'run_manifest.json']
    names += ['epoch%02d_flow.json' % n for n in range(5, 9)]
    names += ['epoch%02d_train.json' % n for n in range(5, 8)]
    snapshot = {'utc': datetime.now(timezone.utc).isoformat(), 'supervisor': s, 'jobs': jobs,
                'progress': load(arm / 'progress.json'),
                'metrics': load(arm / 'epoch08_RAW_metrics.json'), 'checkpoint': cp,
                'event_prefix': dict(counts), 'frozen_inputs_unchanged': True,
                'purpose': 'B8 observation only; B12 learning gate remains unchanged; no formal M/Q yet'}
    E.mkdir(exist_ok=True)
    target = E / 'B_epoch08_small_results_v1.zip'
    assert not target.exists() and not (E / 'B_epoch08_RAW.pth').exists()
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(arm / name, 'arm_B/' + name)
        z.write(P / 'protocol_FROZEN_v1.json', 'protocol_FROZEN_v1.json')
        z.writestr('B_epoch08_evaluation_events_prefix.jsonl', b''.join(events))
        z.writestr('B_epoch08_snapshot.json', json.dumps(snapshot, indent=2))
        z.writestr('B_epoch08_CPU_checkpoint_audit_v1.json', json.dumps(audit, indent=2))
    os.link(weight, E / 'B_epoch08_RAW.pth')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        files = {i.filename: {'bytes': i.file_size, 'sha256': hashlib.sha256(z.read(i)).hexdigest()}
                 for i in z.infolist()}
    print(json.dumps({'status': 'EXPORTED_B8_OBSERVATION_ONLY', 'archive': str(target),
                      'bytes': target.stat().st_size, 'sha256': sha(target), 'files': files,
                      'checkpoint': cp, 'additional_model_forwards': 0,
                      'additional_optimizer_updates': 0, 'additional_COCO_evaluate_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
