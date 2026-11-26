"""One-shot CUDA functional replay of saved candidate tensors; no detector forward."""
import argparse
import ast
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback
import typing
import zipfile
import numpy as np
import torch
import torch.nn.functional as F


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_function(z, path, name, namespace):
    source = z.read(path).decode('utf8')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, (path, name)
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(tree, path, 'exec'), namespace)
    return namespace[name]


class SamplingCapture:
    def __init__(self):
        self.calls = []

    def grid_sample(self, value, grid, **kwargs):
        assert kwargs == dict(mode='bilinear', padding_mode='zeros', align_corners=False), kwargs
        result = F.grid_sample(value, grid, **kwargs)
        self.calls.append((grid.detach(), result.detach()))
        return result


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads(Path(args.protocol).read_text(encoding='utf8'))
    controls = []
    payload = {}
    counters = dict(functional_attention_replays=0, grid_sample_calls=0,
                    new_detector_forwards=0, evaluator_calls=0, optimizer_updates=0, epochs_added=0)
    write(out/'start.json', dict(utc=utc(), protocol_sha256=sha(args.protocol), pid=__import__('os').getpid(),
                               argv=sys.argv, counters=counters))
    try:
        assert p['schema'] == 'saved_sampling_cuda_functional_replay_v1'
        assert sha(__file__) == p['replay_script_sha256']
        assert sha(args.inputs) == p['inputs_sha256']
        assert Path(args.inputs).stat().st_size == p['inputs_bytes']
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 1
        device = torch.device('cuda:0')
        torch.set_grad_enabled(False)
        capture = SamplingCapture()
        with zipfile.ZipFile(args.inputs) as z:
            mr = z.read('manifest.json')
            assert sha_bytes(mr) == p['inputs_manifest_sha256']
            manifest = json.loads(mr)
            for name, entry in manifest['files'].items():
                raw = z.read(name)
                assert len(raw) == entry['bytes'] and sha_bytes(raw) == entry['sha256'], name
            source_p = json.loads(z.read('source_protocol_v4.json'))
            assert source_p['reconstruction_control']['cross_attention_relative_L2_tolerance'] == p['control']['relative_L2']
            assert source_p['reconstruction_control']['cross_attention_max_abs_tolerance'] == p['control']['max_abs']
            namespace = dict(torch=torch, F=capture, List=typing.List)
            load_function(z, 'native/engine/deim/box_ops.py', 'box_xyxy_to_cxcywh', namespace)
            distance_box = load_function(z, 'native/engine/deim/dfine_utils.py', 'distance2bbox', namespace)
            core = load_function(z, 'native/engine/deim/utils.py', 'deformable_attention_core_func_v2', namespace)
            for seed in p['seeds']:
                for batch in p['batches']:
                    prefix = f'seed{seed}/capture/batch_{batch:03d}/'
                    record = json.loads(z.read(prefix+'record.json'))
                    assert record['metadata']['seed'] == seed and record['metadata']['arm'] == 'target_protect'
                    assert record['metadata']['checkpoint_sha256'] == source_p['sources'][str(seed)]['checkpoint_sha256']
                    def event(name, invocation=0):
                        matches = [e for e in record['invocation_events'] if e['module'] == name and e['invocation'] == invocation]
                        assert len(matches) == 1
                        return matches[0]['output']
                    def tensor(desc):
                        raw = z.read(prefix+desc['blob_file'])
                        assert sha_bytes(raw) == desc['blob_sha256']
                        a = np.load(io.BytesIO(raw), allow_pickle=False)
                        assert list(a.shape) == desc['shape'] and str(a.dtype) == desc['storage_numpy_dtype']
                        assert np.isfinite(a).all()
                        return torch.from_numpy(a.copy()).to(device)
                    features = [tensor(event(f'decoder.input_proj.{i}')) for i in range(3)]
                    assert [list(f.shape) for f in features] == [[4,256,80,80],[4,256,40,40],[4,256,20,20]]
                    distances = [tensor(event('decoder.integral', i)) for i in range(3)]
                    stages = []
                    for iid in record['metadata']['image_ids']:
                        with np.load(io.BytesIO(z.read(f'seed{seed}/stages/image_{iid:06d}.npz')), allow_pickle=False) as n:
                            stages.append({k:n[k].copy() for k in n.files})
                    def stage_tensor(key):
                        return torch.from_numpy(np.stack([s[key] for s in stages])).to(device)
                    pre = stage_tensor('pre_after_first_layer_boxes')
                    refs = [stage_tensor('encoder_initial_boxes'), distance_box(pre,distances[0],4.), distance_box(pre,distances[1],4.)]
                    final = stage_tensor('final_boxes')
                    final_reconstructed = distance_box(pre, distances[2], 4.)
                    final_error = (final_reconstructed-final).abs().max().item()
                    # Source order of _get_encoder_input and TransformerDecoder.value_op.
                    memory = torch.cat([f.flatten(2).permute(0,2,1) for f in features], dim=1)
                    value = memory.reshape(4,8400,8,32).permute(0,2,3,1).split([6400,1600,400], dim=-1)
                    for layer in range(3):
                        offsets = tensor(event(f'decoder.decoder.layers.{layer}.cross_attn.sampling_offsets')).reshape(4,300,8,12,2)
                        logits = tensor(event(f'decoder.decoder.layers.{layer}.cross_attn.attention_weights')).reshape(4,300,8,12)
                        assert offsets.dtype == logits.dtype == pre.dtype == torch.float32
                        weights = F.softmax(logits, dim=-1)
                        scale = torch.tensor([1/n for n in [3,6,3] for _ in range(n)], dtype=offsets.dtype, device=device).unsqueeze(-1)
                        ref_input = refs[layer].unsqueeze(2)
                        offset = offsets * scale * ref_input[:,:,None,:,2:] * .5
                        locations = ref_input[:,:,None,:,:2] + offset
                        before = len(capture.calls)
                        reconstructed = core(value, [[80,80],[40,40],[20,20]], locations, weights, [3,6,3])
                        actual = tensor(event(f'decoder.decoder.layers.{layer}.cross_attn'))
                        torch.cuda.synchronize()
                        calls = capture.calls[before:]
                        assert len(calls) == 3
                        counters['functional_attention_replays'] += 1
                        counters['grid_sample_calls'] += 3
                        delta = reconstructed.double()-actual.double()
                        relative = (torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(actual.double()).clamp(min=1e-30)).item()
                        max_abs = delta.abs().max().item()
                        control = dict(seed=seed,batch=batch,layer=layer,final_box_max_abs=final_error,
                                       cross_attention_max_abs=max_abs,cross_attention_relative_L2=relative,
                                       bitwise_equal=bool(torch.equal(reconstructed,actual)),
                                       passed=bool(final_error<=p['control']['final_box_abs'] and max_abs<=p['control']['max_abs'] and relative<=p['control']['relative_L2']))
                        controls.append(control)
                        key=f'seed{seed}_batch{batch:03d}_layer{layer}'
                        # Export source arithmetic products, even on failed controls; analysis must require all controls pass.
                        samples=torch.cat([c[1] for c in calls],dim=-1).reshape(4,8,32,300,12).permute(0,3,1,4,2)
                        for name, t in [('locations',locations),('weights',weights),('samples',samples),('reference',refs[layer]),('reconstructed',reconstructed),('actual',actual)]:
                            payload[key+'_'+name]=t.cpu().numpy()
                        print(json.dumps(control), flush=True)
                        capture.calls.clear()
                    del features, distances, memory, value
        passed=all(c['passed'] for c in controls) and len(controls)==12
        np.savez_compressed(out/'replayed_arrays.npz',**payload)
        report=dict(status='PASS_SAVED_CUDA_FUNCTIONAL_REPLAY' if passed else 'FAIL_SAVED_CUDA_FUNCTIONAL_REPLAY_CONTROL',
                    utc=utc(), controls=controls,counters=counters,protocol_sha256=sha(args.protocol),
                    input_archive_sha256=sha(args.inputs),script_sha256=sha(__file__),
                    torch_version=torch.__version__,cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(0),
                    source_models='previous target_protect EMA12, not BASE and not new FDR endpoint2',
                    seed1_replayed_on_seed2_gpu=True,causal_root_established=False,
                    original_backend_identity_not_inferred_from_current_extension_availability=True)
        write(out/'report.json',report)
        write(out/'manifest.json',dict(files={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in out.iterdir() if f.is_file()}))
        return 0 if passed else 2
    except BaseException:
        write(out/'failure.json',dict(utc=utc(),traceback=traceback.format_exc(),controls=controls,counters=counters))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True)
    sys.exit(main(parser.parse_args()))
