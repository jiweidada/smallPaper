"""CPU archive, tensor, optimizer and recorded-flow checks; no detector/evaluator call."""
from pathlib import Path, PurePosixPath
import argparse
import collections
import hashlib
import json
import sys
import zipfile

P = Path(__file__).resolve().parent
PROTOCOL_SHA = 'da3ae8ac399483307eeb7e221627d0bdaef4222021dde468cb41459d674915a9'
INITIAL_SHA = '582fd6c57aae3725afe7af469024cd0b259a63a5a3694d36a2b1ce56ad2bcd96'
NATIVE_INIT = '2c6e0f9c6765fc7ba069c2292e0884bb86924d2232a952499fcd251cc57b47e6'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text('utf-8'))


def main(arm):
    output = P/f'{arm}_completed_local_verification.json'
    assert not output.exists()
    ready = load(P/f'{arm}_bundle_ready.json')
    bundle = P/f'{arm}_completed_bundle_v1.zip'
    assert bundle.stat().st_size == ready['bytes'] and sha(bundle) == ready['sha256']
    dest = P/f'completed_{arm}_v1'
    assert not dest.exists()
    dest.mkdir()
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read('bundle_manifest.json'))
        assert len(z.namelist()) == len(set(z.namelist())) == len(manifest['files'])+1
        assert set(z.namelist()) == set(manifest['files']) | {'bundle_manifest.json'}
        assert len(manifest['files']) == ready['payload_files']
        for name, expected in manifest['files'].items():
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts
            target = (dest/Path(*rel.parts)).resolve()
            assert target.is_relative_to(dest.resolve())
            target.parent.mkdir(exist_ok=True, parents=True)
            h, size = hashlib.sha256(), 0
            with z.open(name) as source, target.open('xb') as target_file:
                for data in iter(lambda: source.read(1 << 20), b''):
                    target_file.write(data)
                    h.update(data)
                    size += len(data)
            assert size == expected['bytes'] and h.hexdigest() == expected['sha256'], name
    (dest/'bundle_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    folder = dest/f'arm_{arm}'
    result, initial = load(folder/'result.json'), load(folder/'initialization.json')
    process = load(dest/f'{arm}_completed_process_snapshot.json')
    assert process['child_exit_code'] == 0 and not process['timed_out']
    assert result['status'] == 'COMPLETED' and result['epochs'] == 12
    assert result['optimizer_updates'] == 9708 and result['full_model_forwards'] == 10159
    assert result['model_parameters'] == 9877256 and result['extra_parameters'] == 74144
    assert sha(dest/'protocol_FROZEN_v1.json') == result['protocol_sha256'] == initial['protocol_sha256'] == PROTOCOL_SHA
    metrics = load(folder/'epoch12_EMA_metrics.json')
    assert metrics == result['endpoint_metrics'] == ready['endpoint']
    assert load(dest/f'{arm}_completed_source_audit.json')['status'] == 'PASS'
    baseline = P.parent/'A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B'
    keys = ['order_sha256', 'first_eight_actual_input_sha256',
            'first_eight_pre_forward_rng_sha256', 'all_batches_targets_including_ignore_tiles_sha256']
    flows = {}
    for epoch in range(1, 13):
        a = load(folder/f'epoch{epoch:02d}_input_flow.json')
        b = load(baseline/f'epoch{epoch:02d}_input_flow.json')
        assert a['optimizer_updates'] == b['optimizer_updates'] == 809*epoch
        flows[str(epoch)] = {k: a[k] == b[k] for k in keys}
        assert all(flows[str(epoch)].values())
    sys.path.insert(0, str(P.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
    import torch
    torch.set_num_threads(4)
    initial_path = P.parent/'A43_ECDet三组蒸馏最小验证_20261031_v1/common_init_10class.pth'
    assert sha(initial_path) == initial['common_init_sha256'] == result['initialization_sha256'] == NATIVE_INIT
    expected = torch.load(initial_path, map_location='cpu', weights_only=True)['model']
    init_path = dest/'new_branches_untrained_initial_state.pth'
    init_receipt = load(dest/'new_branches_untrained_initial_state_receipt.json')
    assert sha(init_path) == init_receipt['file_sha256']
    extra = torch.load(init_path, map_location='cpu', weights_only=True)['states']
    digest = hashlib.sha256()
    new_names = []
    for i, state in enumerate(extra):
        for name, tensor in state.items():
            digest.update(f'{i}/{name}'.encode())
            digest.update(tensor.contiguous().numpy().tobytes())
            full = f'decoder.decoder.layers.{i}.cross_attn.a56_contrast.{name}'
            expected[full] = tensor
            new_names.append(full)
    assert len(extra) == 4 and len(new_names) == 12
    assert sum(expected[n].numel() for n in new_names) == 74144
    assert digest.hexdigest() == init_receipt['tensor_sha256'] == initial['new_module_initialization_sha256'] == INITIAL_SHA
    checkpoint = torch.load(folder/'epoch12.pth', map_location='cpu', weights_only=True)
    assert checkpoint['arm'] == arm and checkpoint['epoch'] == 12
    assert checkpoint['optimizer_updates'] == checkpoint['ema']['updates'] == 9708
    assert checkpoint['protocol_sha256'] == PROTOCOL_SHA
    states = {}
    for label, state in [('raw', checkpoint['model']), ('EMA', checkpoint['ema']['module'])]:
        assert state.keys() == expected.keys()
        changed = 0
        structural = []
        for name, tensor in state.items():
            assert tensor.shape == expected[name].shape and tensor.dtype == expected[name].dtype, name
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                assert name.endswith('.anchors') and torch.equal(tensor, expected[name]), name
                structural.append(name)
            changed += int(not torch.equal(tensor, expected[name]))
        bn = ['encoder.fpn_blocks.0.cv4.norm', 'encoder.fpn_blocks.1.cv4.norm',
              'encoder.pan_blocks.0.cv4.norm', 'encoder.pan_blocks.1.cv4.norm']
        assert all(int(state[n+'.num_batches_tracked']) == 9708 for n in bn)
        delta = {n: float((state[n]-expected[n]).abs().max()) for n in new_names}
        assert changed > 0 and any(v > 0 for v in delta.values())
        states[label] = {'changed_state_tensors': changed, 'new_parameter_max_abs_changes': delta,
                         'four_BN_counters': 9708, 'unchanged_nonfinite_anchor_buffers': structural}
    histogram = collections.Counter(int(v['step']) for v in checkpoint['optimizer']['state'].values())
    assert dict(histogram) == {9708: 482+12}, dict(histogram)
    for row in checkpoint['optimizer']['state'].values():
        assert all(torch.isfinite(row[k]).all() for k in ['exp_avg', 'exp_avg_sq'])
    verified = {'status': 'PASS', 'arm': arm, 'epochs': 12, 'optimizer_updates': 9708,
                'full_model_forwards': 10159, 'bundle_sha256': ready['sha256'],
                'payload_files_verified': len(manifest['files']), 'endpoint_metrics': metrics,
                'checkpoint_state_checks': states, 'optimizer_step_histogram': dict(histogram),
                'recorded_flow_checks_against_B': flows, 'new_detector_forwards': 0,
                'new_optimizer_updates': 0, 'scope': 'CPU archive and recorded-state verification; no prediction rerun or independent-test claim'}
    with output.open('x', encoding='utf-8') as f:
        json.dump(verified, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in verified.items()
                      if k not in ['checkpoint_state_checks', 'recorded_flow_checks_against_B']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['M', 'Q'], required=True)
    main(parser.parse_args().arm)
