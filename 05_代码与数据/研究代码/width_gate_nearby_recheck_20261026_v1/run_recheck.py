"""CPU, aggregate-only recheck of FDR boundaries on criterion-2 nearby queries."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from PIL import Image

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent
EN = ROOT.parents[1]
MAIN = EN/'持续寻找研究主线_20261020_v2'
WORK = EN/'code/unit_instance_completion_20261021_v1'
CACHE = WORK/'342_codex_stage3_20261022_v1'
NATIVE = EN/'code/dn_pre_frozen_diagnostic_20261020_v1/native'
VAL = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val')
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(NATIVE))
import stage1_cache_v3 as S
import engine.deim.dfine_decoder as dec
from engine.deim.dfine_decoder import Integral

START = time.monotonic()
LOG = None
CAP = {}
ORIGINAL_D2B = dec.distance2bbox


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode('utf-8'))
        t = value.detach().cpu().contiguous()
        h.update(str(t.dtype).encode('ascii'))
        h.update(str(tuple(t.shape)).encode('ascii'))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def dump(name, value):
    with (ROOT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def log(message):
    elapsed = time.monotonic()-START
    line = f'{elapsed:.1f}s {message}'
    print(line, flush=True)
    LOG.write(line+'\n')
    LOG.flush()
    (ROOT/'progress.json').write_text(json.dumps({'pid': os.getpid(), 'elapsed_seconds': elapsed,
                                                'message': message}), encoding='utf-8')
    if elapsed > 7200:
        raise TimeoutError('Two-hour hard budget exceeded')


def capture_d2b(points, distance, reg_scale):
    CAP['ref'] = points.detach()
    CAP['dist'] = distance.detach()
    CAP['rs'] = abs(float(reg_scale))
    return ORIGINAL_D2B(points, distance, reg_scale)


def capture_integral(module, inputs, output):
    CAP['corners'] = inputs[0].detach()
    CAP['project'] = inputs[1].detach()


def edges(boxes):
    b = boxes.double()*640.
    return torch.cat((b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2), dim=1)


def iou_matrix(a, b):
    lo = torch.maximum(a[:, None, :2], b[None, :, :2])
    hi = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (hi-lo).clamp(min=0)
    inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1])
    bb = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+bb[None]-inter+1e-12)


def quantiles(t):
    t = t.double().flatten()
    assert torch.isfinite(t).all()
    ps = torch.tensor([0., .1, .25, .5, .75, .9, 1.], dtype=torch.float64)
    values = torch.quantile(t, ps)
    return dict(zip(('min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'), map(float, values)))


def mass_summary(m):
    all_event = (m >= .5).any(dim=1)
    x_event = (m[:, [0, 2]] >= .5).any(dim=1)
    return {'mass_quantiles_all_edges': quantiles(m),
            'mass_quantiles_horizontal_edges': quantiles(m[:, [0, 2]]),
            'events_any_four_edges': int(all_event.sum()),
            'rate_any_four_edges': float(all_event.double().mean()),
            'events_horizontal_only': int(x_event.sum()),
            'rate_horizontal_only': float(x_event.double().mean()),
            'side_rates_LTRB': (m >= .5).double().mean(dim=0).tolist()}


def seed_run(cp, seed, input_records):
    tick = time.monotonic()
    checkpoint = Path(cp['path'])
    cp_sha_before = file_sha(checkpoint)
    assert cp_sha_before == cp['expected_SHA256_from_288']
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert int(ck['last_epoch']) == 23
    model_branch_sha = state_sha(ck['model'])
    ema_branch_sha = state_sha(ck['ema']['module'])
    del ck
    model = S.load_real_model(str(checkpoint))
    assert not model.training and all(p.device.type == 'cpu' for p in model.parameters())
    before = state_sha(model.state_dict())
    assert before == model_branch_sha
    integrals = [m for m in model.modules() if isinstance(m, Integral)]
    assert len(integrals) == 1
    hook = integrals[0].register_forward_hook(capture_integral)
    columns = {key: [] for key in ('endpoint', 'outer', 'negative_tail', 'positive_tail',
                                  'hist', 'refw_px', 'gtw_px', 'ref_gt_ratio', 'pred_gt_ratio',
                                  'pred_ref_ratio', 'width_error_px', 'center_distance_px',
                                  'central_step_px', 'outer_gap_px', 'required', 'x_reachable',
                                  'all_reachable')}
    counts = {'images': 0, 'all_gt': 0, 'tiny': 0, 'Geo0': 0, 'matched_criterion2': 0,
              'any_class_IoU_ge_0_1': 0, 'same_class_IoU_ge_0_1': 0,
              'overlap_ge_0_1_wrong_class_only': 0, 'any_class_IoU_lt_0_1': 0}
    box_diff = logits_diff = recon_diff = 0.
    reg_scales, up_values = set(), set()
    project = None
    files = sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt'))
    assert len(files) == 548
    log(f'seed{seed} checkpoint model branch loaded; epoch=23, EMA branch equal={before == ema_branch_sha}')
    try:
        for k, f in enumerate(files, 1):
            digest = file_sha(f)
            input_records.append({'path': str(f), 'sha256': digest, 'bytes': f.stat().st_size})
            e = torch.load(f, map_location='cpu', weights_only=True)
            assert str(e['seed']) == str(seed)
            assert e['checkpoint_sha256'] == cp_sha_before
            pb_cache = e['final300_boxes'].double()
            lg_cache = e['final300_logits'].double()
            gb = e['gt_boxes_640'].double()
            gl = e['gt_labels'].long()
            ts = e['gt_short_side_640_all'].double()
            image_path = VAL/'images'/e['file_name']
            assert file_sha(image_path) == e['source_image_sha256']
            CAP.clear()
            with Image.open(image_path) as image:
                pb, lg = S.global_forward_640(model, image.convert('RGB'))
            pb, lg = pb.double(), lg.double()
            diff_pb = float((pb-pb_cache).abs().max())
            diff_lg = float((lg-lg_cache).abs().max())
            assert diff_pb == 0 and diff_lg == 0, (seed, k, diff_pb, diff_lg)
            box_diff, logits_diff = max(box_diff, diff_pb), max(logits_diff, diff_lg)
            ref, dist = CAP['ref'][0].double(), CAP['dist'][0].double()
            project = CAP['project'].double()
            rs = CAP['rs']
            reg_scales.add(rs)
            up_values.update(float(v) for name, value in model.state_dict().items()
                             if name.endswith('.up') for v in value.flatten())
            r = project.numel()-1
            assert r == 32 and rs == 4.
            mass = torch.softmax(CAP['corners'][0].view(-1, 4, r+1).double(), dim=-1)
            assert mass.shape == (300, 4, 33) and ref.shape == (300, 4)
            x1 = ref[:, 0]-(rs/2+dist[:, 0])*ref[:, 2]/rs
            x2 = ref[:, 0]+(rs/2+dist[:, 2])*ref[:, 2]/rs
            y1 = ref[:, 1]-(rs/2+dist[:, 1])*ref[:, 3]/rs
            y2 = ref[:, 1]+(rs/2+dist[:, 3])*ref[:, 3]/rs
            reconstruction = torch.stack(((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1), dim=1)
            err = float((reconstruction-pb).abs().max())
            assert err <= 1e-6
            recon_diff = max(recon_diff, err)
            I = iou_matrix(edges(pb_cache), gb)
            tiny = ts < 4
            geo0 = tiny & (I.max(dim=0).values < .5)
            same_class = lg_cache.argmax(dim=1)[:, None] == gl[None, :]
            eligible = same_class & (I >= .1)
            matched_iou, query = torch.where(eligible, I, torch.full_like(I, -1.)).max(dim=0)
            matched = geo0 & (matched_iou >= .1)
            any_overlap = (I >= .1).any(dim=0)
            same_overlap = eligible.any(dim=0)
            counts['images'] += 1
            counts['all_gt'] += int(ts.numel())
            counts['tiny'] += int(tiny.sum())
            counts['Geo0'] += int(geo0.sum())
            counts['matched_criterion2'] += int(matched.sum())
            counts['any_class_IoU_ge_0_1'] += int((geo0 & any_overlap).sum())
            counts['same_class_IoU_ge_0_1'] += int((geo0 & same_overlap).sum())
            counts['overlap_ge_0_1_wrong_class_only'] += int((geo0 & any_overlap & ~same_overlap).sum())
            counts['any_class_IoU_lt_0_1'] += int((geo0 & ~any_overlap).sum())
            g = matched.nonzero().flatten()
            if g.numel():
                q = query[g]
                assert (pb[q, 2:] > 0).all() and (ref[q, 2:] > 0).all()
                pm = mass[q]
                columns['endpoint'].append(pm[:, :, 0]+pm[:, :, -1])
                negative = pm[:, :, 0]+pm[:, :, 1]
                positive = pm[:, :, -2]+pm[:, :, -1]
                columns['negative_tail'].append(negative)
                columns['positive_tail'].append(positive)
                columns['outer'].append(negative+positive)
                columns['hist'].append(pm.argmax(dim=-1))
                rw = ref[q, 2]*640
                rh = ref[q, 3]*640
                gt = gb[g]
                gw = gt[:, 2]-gt[:, 0]
                pw = pb[q, 2]*640
                columns['refw_px'].append(rw)
                columns['gtw_px'].append(gw)
                columns['ref_gt_ratio'].append(rw/gw)
                columns['pred_gt_ratio'].append(pw/gw)
                columns['pred_ref_ratio'].append(pw/rw)
                columns['width_error_px'].append((pw-gw).abs())
                gc = (gt[:, :2]+gt[:, 2:])/2
                columns['center_distance_px'].append(torch.linalg.vector_norm(pb[q, :2]*640-gc, dim=1))
                columns['central_step_px'].append(rw/rs*(project[17]-project[16]))
                columns['outer_gap_px'].append(rw/rs*(project[32]-project[31]))
                cx, cy = ref[q, 0]*640, ref[q, 1]*640
                required = torch.stack(((cx-gt[:, 0])/(rw/rs)-rs/2,
                                        (cy-gt[:, 1])/(rh/rs)-rs/2,
                                        (gt[:, 2]-cx)/(rw/rs)-rs/2,
                                        (gt[:, 3]-cy)/(rh/rs)-rs/2), dim=1)
                reachable = (required >= project.min()) & (required <= project.max())
                columns['required'].append(required)
                columns['x_reachable'].append(reachable[:, [0, 2]].all(dim=1))
                columns['all_reachable'].append(reachable.all(dim=1))
            if k % 64 == 0 or k == len(files):
                log(f'seed{seed} {k}/548, criterion2 matched={counts["matched_criterion2"]}')
    finally:
        hook.remove()
    assert counts['all_gt'] == 38759 and counts['tiny'] == 4952
    assert counts['Geo0'] == {1: 3715, 2: 3727}[seed]
    assert counts['matched_criterion2'] == {1: 1243, 2: 1307}[seed]
    after = state_sha(model.state_dict())
    assert after == before
    cp_sha_after = file_sha(checkpoint)
    assert cp_sha_after == cp_sha_before
    data = {key: torch.cat(value) for key, value in columns.items()}
    histogram = torch.bincount(data['hist'].flatten(), minlength=33)
    n_edges = int(histogram.sum())
    result = {
        'seed': seed, 'counts': counts,
        'checkpoint': {'path': str(checkpoint), 'state_key': 'model', 'last_epoch': 23,
                       'epochs_completed': 24, 'file_sha256_before': cp_sha_before,
                       'file_sha256_after': cp_sha_after, 'state_dict_sha256_before': before,
                       'state_dict_sha256_after': after, 'ema_state_dict_sha256': ema_branch_sha,
                       'model_branch_equals_ema': before == ema_branch_sha},
        'integrity': {'boxes_max_abs_diff_vs_cache': box_diff, 'logits_max_abs_diff_vs_cache': logits_diff,
                      'max_normalized_box_reconstruction_error': recon_diff,
                      'actual_decoder_module': str(Path(dec.__file__).resolve()),
                      'reg_scale': sorted(reg_scales), 'up': sorted(up_values), 'project': project.tolist()},
        'endpoint': mass_summary(data['endpoint']),
        'outer_two_each_end': mass_summary(data['outer']),
        'negative_tail': mass_summary(data['negative_tail']),
        'positive_tail': mass_summary(data['positive_tail']),
        'argmax': {'edge_count': n_edges, 'bin_counts': histogram.tolist(),
                   'bin_fractions': (histogram.double()/n_edges).tolist()},
        'geometry_quantiles': {key: quantiles(data[key]) for key in (
            'refw_px', 'gtw_px', 'ref_gt_ratio', 'pred_gt_ratio', 'pred_ref_ratio',
            'width_error_px', 'center_distance_px', 'central_step_px', 'outer_gap_px')},
        'required_offsets_horizontal_quantiles': quantiles(data['required'][:, [0, 2]]),
        'GT_horizontal_edges_within_support_count': int(data['x_reachable'].sum()),
        'GT_horizontal_edges_within_support_fraction': float(data['x_reachable'].double().mean()),
        'GT_all_edges_within_support_fraction': float(data['all_reachable'].double().mean()),
        'elapsed_seconds': time.monotonic()-tick,
    }
    dump(f'seed{seed}_aggregates.json', result)
    log(f'seed{seed} complete: endpoint={result["endpoint"]["events_any_four_edges"]}/{counts["matched_criterion2"]}; outer={result["outer_two_each_end"]["events_any_four_edges"]}/{counts["matched_criterion2"]}')
    return result


def main():
    global LOG
    protocol_bytes = (ROOT/'protocol.json').read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    (ROOT/'frozen_protocol.sha256').open('x', encoding='utf-8').write(protocol_sha+'  protocol.json\n')
    LOG = (ROOT/'run.log').open('x', encoding='utf-8')
    input_paths = [MAIN/'327_训练实施输入索引.json', MAIN/'348b_tiny_geo0_oracle.py',
                   MAIN/'348b_tiny_geo0_oracle.json', MAIN/'348c_axis_decomposition.py',
                   MAIN/'348c_axis_decomposition.json', MAIN/'360_线程B_宽度门槛诊断_20261026_v2.md',
                   Path(S.__file__), Path(dec.__file__), NATIVE/'engine/deim/dfine_utils.py']
    records = [{'path': str(p), 'sha256': file_sha(p), 'bytes': p.stat().st_size} for p in input_paths]
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                'script_sha256': file_sha(__file__), 'protocol_sha256': protocol_sha,
                'python_executable': sys.executable, 'torch_version': torch.__version__,
                'torch_threads': torch.get_num_threads(), 'device': 'cpu',
                'loaded_state': 'checkpoint[model]; not EMA', 'expected_last_epoch': 23}
    dump('run_metadata.json', metadata)
    dec.distance2bbox = capture_d2b
    try:
        log('Protocol frozen; starting two-seed CPU recheck on criterion-2 queries')
        index = json.loads((MAIN/'327_训练实施输入索引.json').read_text(encoding='utf-8'))
        results = {}
        for seed in (1, 2):
            cp = next(c for c in index['checkpoints'] if c['seed'] == seed)
            results[str(seed)] = seed_run(cp, seed, records)
        for record in records:
            assert file_sha(record['path']) == record['sha256'], record['path']
        dump('input_manifest.json', {'files': records, 'rehash_unchanged': True})
        output = {'protocol_sha256': protocol_sha, 'seeds': results,
                  'execution': {'models_loaded': 2, 'CPU_forwards': 1096, 'training': False,
                                'test_access': False, 'GPU': False, 'remote': False,
                                'wall_seconds': time.monotonic()-START},
                  'interpretation': 'Supplementary criterion-2 diagnosis. No causal identification, method claim, or efficacy gate.'}
        dump('aggregate_results.json', output)
        size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        assert size < 20_000_000
        log(f'COMPLETED, aggregate and metadata bytes={size}')
    except Exception as exc:
        dump('failure.json', {'type': type(exc).__name__, 'message': str(exc),
                             'elapsed_seconds': time.monotonic()-START})
        raise
    finally:
        dec.distance2bbox = ORIGINAL_D2B
        LOG.close()
    dump('delivery_manifest.json', {'files': [
        {'path': p.name, 'bytes': p.stat().st_size, 'sha256': file_sha(p)}
        for p in sorted(ROOT.iterdir()) if p.is_file()], 'self_not_hashed': True})


if __name__ == '__main__':
    main()
