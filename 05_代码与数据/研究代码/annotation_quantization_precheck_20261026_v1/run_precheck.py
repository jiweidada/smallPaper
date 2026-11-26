"""Frozen, CPU-only cached-prediction test of synthetic annotation quantization."""
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import differential_evolution
from scipy.stats import rankdata, spearmanr
import torch

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
LOG = None
P = None


def dump(name, value):
    with (ROOT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def log(message):
    line = f'{time.monotonic()-START:.1f}s {message}'
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()
    (ROOT / 'progress.json').write_text(json.dumps({
        'pid': os.getpid(), 'elapsed_seconds': time.monotonic()-START,
        'message': message, 'utc': datetime.now(timezone.utc).isoformat()
    }, ensure_ascii=False), encoding='utf-8')
    if P is not None and time.monotonic()-START > P['timeout_seconds']:
        raise TimeoutError('Frozen 1200-second timeout exceeded')


def xywh_edges(a):
    return np.concatenate((a[..., :2], a[..., :2]+a[..., 2:]), axis=-1)


def iou_pairs(a, b):
    wh = np.maximum(0., np.minimum(a[..., 2:], b[..., 2:])-np.maximum(a[..., :2], b[..., :2]))
    inter = wh[..., 0]*wh[..., 1]
    aa = np.prod(a[..., 2:]-a[..., :2], axis=-1)
    ab = np.prod(b[..., 2:]-b[..., :2], axis=-1)
    return inter / np.maximum(aa+ab-inter, 1e-300)


def axis_vertices(lower, upper, pl, pr):
    """Vertices of xywh cell cut by left/right-edge overlap break lines.

    Horizontal cell variables are x and w. Break lines: x=pl, x=pr,
    x+w=pl, x+w=pr, plus four cell boundary lines. Within each polygon,
    IoU with the other axis fixed is linear-fractional, so an optimum
    occurs at a vertex. Enumerate 20 nonparallel line intersections.
    Each axis's partition is independent of the other axis's values.
    Sequential axis optimization therefore yields a Cartesian vertex pair.
    """
    v = np.column_stack((lower[:, 0], upper[:, 0], pl, pr))
    w = np.column_stack((lower[:, 1], upper[:, 1]))
    r = np.column_stack((pl, pr))
    ax = np.broadcast_to(v[:, :, None], (len(v), 4, 2)).reshape(-1, 8)
    aw = np.broadcast_to(w[:, None, :], (len(v), 4, 2)).reshape(-1, 8)
    bx = ax.copy()
    bw = (r[:, None, :]-v[:, :, None]).reshape(-1, 8)
    cw = np.broadcast_to(w[:, :, None], (len(v), 2, 2)).reshape(-1, 4)
    cx = (r[:, None, :]-w[:, :, None]).reshape(-1, 4)
    x = np.concatenate((ax, bx, cx), axis=1)
    width = np.concatenate((aw, bw, cw), axis=1)
    valid = ((x >= lower[:, 0, None]-1e-10) & (x <= upper[:, 0, None]+1e-10)
             & (width >= lower[:, 1, None]-1e-10) & (width <= upper[:, 1, None]+1e-10))
    overlap = np.maximum(0., np.minimum(pr[:, None], x+width)-np.maximum(pl[:, None], x))
    return width, overlap, valid


def max_cell_iou(pred, lower, upper):
    w, ix, vx = axis_vertices(lower[:, [0, 2]], upper[:, [0, 2]], pred[:, 0], pred[:, 2])
    h, iy, vy = axis_vertices(lower[:, [1, 3]], upper[:, [1, 3]], pred[:, 1], pred[:, 3])
    inter = ix[:, :, None]*iy[:, None, :]
    pa = np.prod(pred[:, 2:]-pred[:, :2], axis=1)
    union = pa[:, None, None]+w[:, :, None]*h[:, None, :]-inter
    valid = vx[:, :, None] & vy[:, None, :]
    values = np.divide(inter, union, out=np.full_like(inter, -np.inf), where=valid & (union > 0))
    return values.max(axis=(1, 2))


def expected_cell_target(pred, lower, upper, order, gamma, progress=True):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u = (nodes+1.)/2.
    w2 = np.outer(weights/2., weights/2.).reshape(-1)
    out = np.empty(len(pred), dtype=np.float64)
    for start in range(0, len(pred), 32):
        end = min(start+32, len(pred))
        p, lo, hi = pred[start:end], lower[start:end], upper[start:end]
        node = lo[:, :, None]+(hi-lo)[:, :, None]*u[None, None, :]
        x, y, w, h = [node[:, i] for i in range(4)]
        rw = np.broadcast_to(w[:, None, :], (len(p), order, order)).reshape(len(p), -1)
        rh = np.broadcast_to(h[:, None, :], (len(p), order, order)).reshape(len(p), -1)
        ix = np.maximum(0., np.minimum(p[:, 2, None, None], x[:, :, None]+w[:, None, :])
                        -np.maximum(p[:, 0, None, None], x[:, :, None])).reshape(len(p), -1)
        iy = np.maximum(0., np.minimum(p[:, 3, None, None], y[:, :, None]+h[:, None, :])
                        -np.maximum(p[:, 1, None, None], y[:, :, None])).reshape(len(p), -1)
        inter = ix[:, :, None]*iy[:, None, :]
        area = np.prod(p[:, 2:]-p[:, :2], axis=1)
        union = area[:, None, None]+rw[:, :, None]*rh[:, None, :]-inter
        target = np.power(inter/union, gamma)
        out[start:end] = np.einsum('nij,i,j->n', target, w2, w2, optimize=False)
        if progress and start % 1024 == 0:
            log(f'integration order={order}: {end}/{len(pred)}')
    return out


def math_audit():
    rng = np.random.default_rng(2026102601)
    lo = np.column_stack((rng.uniform(2, 10, (16, 2)), rng.uniform(2, 8, (16, 2))))
    hi = lo+1.
    reference = lo+rng.random((16, 4))
    pred_xywh = reference.copy()
    pred_xywh[:, :2] += rng.uniform(-1.5, 1.5, (16, 2))
    pred_xywh[:, 2:] *= rng.uniform(.7, 1.3, (16, 2))
    pred = xywh_edges(pred_xywh)
    exact = max_cell_iou(pred, lo, hi)
    checks = []
    for i in range(len(lo)):
        opt = differential_evolution(
            lambda z: -float(iou_pairs(pred[i], xywh_edges(z))),
            list(zip(lo[i], hi[i])), seed=20261026+i,
            popsize=12, maxiter=600, tol=1e-10, polish=True,
        )
        numerical = -float(opt.fun)
        gap = float(exact[i]-numerical)
        assert -1e-8 <= gap < 2e-5, (i, exact[i], numerical)
        checks.append({'case': i, 'vertex_max': float(exact[i]), 'DE_max': numerical, 'gap': gap})
    direct = iou_pairs(pred, xywh_edges(reference))**P['gamma']
    zero_max = max_cell_iou(pred, reference, reference)**P['gamma']
    zero_exp = expected_cell_target(pred, reference, reference, 8, P['gamma'], False)
    assert np.max(np.abs(zero_max-direct)) < 1e-12
    assert np.max(np.abs(zero_exp-direct)) < 1e-12
    scale = np.array([.4, .8, .4, .8])
    trans = np.array([13., -5., 13., -5.])
    affine_error = np.max(np.abs(iou_pairs(pred, xywh_edges(reference))
                         -iou_pairs(pred*scale+trans, xywh_edges(reference)*scale+trans)))
    assert affine_error < 1e-12
    result = {'independent_optimizer_checks': checks,
              'zero_width_set_max_error': float(np.max(np.abs(zero_max-direct))),
              'zero_width_set_expectation_error': float(np.max(np.abs(zero_exp-direct))),
              'affine_iou_error': float(affine_error)}
    dump('math_audit.json', result)
    log('Independent numerical optimizer, zero-noise and affine checks passed')


def load_pairs():
    annotation_path = Path(P['annotation_file'])
    raw = annotation_path.read_bytes()
    assert sha(raw) == P['annotation_sha256'], 'Annotation file hash mismatch'
    annotations = {int(a['id']): a for a in json.loads(raw)['annotations']}
    manifest = []
    data = {}
    counts = {}
    for seed in P['seeds']:
        cache = Path(P['input_cache'])/f'342_cache_val_v2_seed{seed}'
        files = sorted(cache.glob('342_*.pt'))
        assert len(files) == P['expected_images_per_seed']
        refs, preds, meta, phases, queries = [], [], [], [], []
        count = {'total_gt': 0, 'cached_tiny_gt': 0, 'after_minimum_width_height': 0, 'eligible_gt': 0}
        for file_index, file in enumerate(files):
            raw = file.read_bytes()
            manifest.append({'path': str(file), 'size': len(raw), 'sha256': sha(raw)})
            d = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            image_id = int(d['image_id'])
            ow, oh = map(int, d['orig_wh'])
            ids = np.asarray(d['gt_annotation_ids'], dtype=np.int64)
            g640 = d['gt_boxes_640'].double().numpy()
            assert len(g640) == len(ids)
            count['total_gt'] += len(ids)
            if not len(ids):
                continue
            ref = np.asarray([np.rint(annotations[int(a)]['bbox']) for a in ids], dtype=np.float64)
            assert all(int(annotations[int(a)]['image_id']) == image_id for a in ids)
            ref_edges = xywh_edges(ref)
            scale4 = np.array([640/ow, 640/oh, 640/ow, 640/oh])
            assert np.max(np.abs(g640/scale4-ref_edges)) < .01
            tiny = np.min(g640[:, 2:]-g640[:, :2], axis=1) < 4.
            count['cached_tiny_gt'] += int(tiny.sum())
            minimum = (ref[:, 2] >= 2) & (ref[:, 3] >= 2)
            count['after_minimum_width_height'] += int((tiny & minimum).sum())
            interior = ((ref[:, 0] >= 2) & (ref[:, 1] >= 2)
                        & (ref_edges[:, 2] <= ow-2) & (ref_edges[:, 3] <= oh-2))
            eligible = tiny & minimum & interior
            count['eligible_gt'] += int(eligible.sum())
            b = d['final300_boxes'].double().numpy()*640.
            pred640 = np.concatenate((b[:, :2]-b[:, 2:]/2., b[:, :2]+b[:, 2:]/2.), axis=1)
            pred = pred640/scale4
            assert len(pred) == 300 and np.isfinite(pred).all()
            assert (pred[:, 2:] > pred[:, :2]).all()
            selected = np.flatnonzero(eligible)
            if len(selected):
                matrix = iou_pairs(ref_edges[selected, None, :], pred[None, :, :])
                best = matrix.argmax(axis=1)
                refs.append(ref[selected])
                preds.append(pred[best])
                queries.extend(best.tolist())
                for j in selected:
                    aid = int(ids[j])
                    meta.append((image_id, aid))
                    phases.append(np.random.default_rng(np.random.SeedSequence([20261026, image_id, aid])).random(4))
            if (file_index+1) % 100 == 0 or file_index+1 == len(files):
                log(f'loaded seed{seed}: {file_index+1}/{len(files)}, eligible={count["eligible_gt"]}')
        arrays = {'reference': np.concatenate(refs), 'prediction': np.concatenate(preds),
                  'ids': np.asarray(meta, dtype=np.int64), 'phase': np.asarray(phases),
                  'query_index': np.asarray(queries, dtype=np.int64)}
        assert len(np.unique(arrays['ids'], axis=0)) == len(arrays['ids'])
        assert count['total_gt'] == 38759 and count['cached_tiny_gt'] == 4952
        if seed != P['seeds'][0]:
            first = data[P['seeds'][0]]
            for key in ('reference', 'ids', 'phase'):
                assert np.array_equal(first[key], arrays[key]), f'Seed mismatch: {key}'
        np.savez_compressed(ROOT/f'pairs_seed{seed}.npz', **arrays)
        data[seed], counts[f'seed{seed}'] = arrays, count
    dump('input_manifest.json', {'annotation': {'path': str(annotation_path), 'sha256': P['annotation_sha256']},
                                 'cache_files': manifest, 'counts': counts})
    common = data[P['seeds'][0]]
    dump('cohort.json', {'counts': counts,
                        'eligible_images': len(np.unique(common['ids'][:, 0])),
                        'cohort_id_bytes_sha256': sha(common['ids'].tobytes()),
                        'seed_pairing_identical': True})
    return data, counts


def metrics(target, reference):
    error = target-reference
    mse = float(np.mean(error**2))
    return {'MSE': mse, 'RMSE': math.sqrt(mse), 'MAE': float(np.mean(np.abs(error))),
            'Spearman': float(spearmanr(target, reference).statistic)}


def correlation_rows(a):
    ranks = rankdata(a, axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(ranks*ranks, axis=1))
    return np.sum(ranks[1:]*ranks[0], axis=1)/(norms[1:]*norms[0])


def run_targets(data):
    cases = {}
    csv_file = (ROOT/'per_gt_targets.csv').open('x', encoding='utf-8', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['seed', 'rule', 'image_id', 'annotation_id', 'query_index',
                     'reference_target', 'point', 'center', 'max', 'expected8', 'expected12'])
    for seed, d in data.items():
        ref, pred, phase = d['reference'], d['prediction'], d['phase']
        reference = iou_pairs(pred, xywh_edges(ref))**P['gamma']
        for rule in P['quantization']['rules']:
            log(f'begin seed{seed} {rule}: {len(ref)} GT')
            if rule == 'floor':
                q = np.floor(ref-phase)+phase
                lower, upper = q, q+1.
            else:
                q = np.floor(ref-phase+.5)+phase
                lower, upper = q-.5, q+.5
            assert (ref >= lower-1e-10).all() and (ref <= upper+1e-10).all()
            assert (lower[:, 2:] > 0).all()
            point = iou_pairs(pred, xywh_edges(q))**P['gamma']
            center = iou_pairs(pred, xywh_edges((lower+upper)/2.))**P['gamma']
            maximum = max_cell_iou(pred, lower, upper)**P['gamma']
            expected8 = expected_cell_target(pred, lower, upper, 8, P['gamma'])
            expected12 = expected_cell_target(pred, lower, upper, 12, P['gamma'])
            for target in (point, center, reference, expected8, expected12):
                assert (maximum+1e-9 >= target).all()
            arrays = {'reference': reference, 'point': point, 'center': center, 'max': maximum,
                      'expected': expected12, 'expected8': expected8}
            case_key = f'seed{seed}_{rule}'
            np.savez_compressed(ROOT/f'targets_{case_key}.npz', **arrays)
            diff = np.abs(expected8-expected12)
            numeric = {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
                       'rho_abs': abs(float(spearmanr(expected8, reference).statistic)
                                      -float(spearmanr(expected12, reference).statistic))}
            numeric['passed'] = bool(
                numeric['max_abs'] <= P['gates']['numeric_max_abs_expected8_vs12']
                and numeric['mean_abs'] <= P['gates']['numeric_mean_abs_expected8_vs12']
                and numeric['rho_abs'] <= P['gates']['numeric_abs_rho_expected8_vs12'])
            cases[case_key] = {'seed': seed, 'rule': rule, 'arrays': arrays,
                               'metrics': {arm: metrics(arrays[arm], reference) for arm in P['arms']},
                               'numeric_audit': numeric, 'comparisons': {}}
            for i, (image_id, aid) in enumerate(d['ids']):
                writer.writerow([seed, rule, image_id, aid, d['query_index'][i], reference[i],
                                 point[i], center[i], maximum[i], expected8[i], expected12[i]])
            log(f'completed targets {case_key}; integration audit={numeric["passed"]}')
    csv_file.close()
    return cases


def evaluate(cases, data):
    common = data[P['seeds'][0]]
    images = common['ids'][:, 0]
    unique_images = np.unique(images)
    groups = [np.flatnonzero(images == image) for image in unique_images]
    rng = np.random.default_rng(P['bootstrap']['seed'])
    draws = {key: {arm: [] for arm in ('point', 'center')} for key in cases}
    for repeat in range(P['bootstrap']['repeats']):
        sampled = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups[g] for g in sampled])
        for key, case in cases.items():
            a = case['arrays']
            values = np.stack([a['reference'], a['point'], a['center'], a['expected']])[:, index]
            rho = correlation_rows(values)
            errors = np.mean((values[1:]-values[0])**2, axis=1)
            for k, arm in enumerate(('point', 'center')):
                draws[key][arm].append([float(errors[k]-errors[2]), float(rho[2]-rho[k])])
        if (repeat+1) % 200 == 0:
            log(f'paired image bootstrap {repeat+1}/{P["bootstrap"]["repeats"]}')
    for key, case in cases.items():
        e = case['metrics']['expected']
        for arm in ('point', 'center'):
            base = case['metrics'][arm]
            relative = (base['MSE']-e['MSE'])/base['MSE'] if base['MSE'] > 0 else -1.
            rho_advantage = e['Spearman']-base['Spearman']
            intervals = np.percentile(np.asarray(draws[key][arm]), [2.5, 97.5], axis=0)
            gates = {'MSE_reduction_at_least_5pct': relative >= P['gates']['minimum_relative_MSE_reduction'],
                     'rho_increase_at_least_0.001': rho_advantage >= P['gates']['minimum_rho_increase'],
                     'MSE_advantage_CI_strictly_positive': intervals[0, 0] > 0,
                     'rho_advantage_CI_strictly_positive': intervals[0, 1] > 0}
            case['comparisons'][arm] = {
                'relative_MSE_reduction': float(relative), 'rho_advantage': float(rho_advantage),
                'MSE_advantage_CI95': intervals[:, 0].tolist(),
                'rho_advantage_CI95': intervals[:, 1].tolist(),
                'gates': {k: bool(v) for k, v in gates.items()}, 'passed': bool(all(gates.values()))}
        case['passed'] = bool(all(c['passed'] for c in case['comparisons'].values()))
        del case['arrays']
    sample_pass = len(images) >= P['gates']['sample_min_gt'] and len(unique_images) >= P['gates']['sample_min_images']
    numeric_pass = all(c['numeric_audit']['passed'] for c in cases.values())
    effect_pass = all(c['passed'] for c in cases.values())
    verdict = ('insufficient' if not (sample_pass and numeric_pass)
               else 'pass_synthetic_target_precheck' if effect_pass
               else 'stop_current_expected_version')
    return {'id': P['id'], 'verdict': verdict, 'eligible_gt_per_seed': len(images),
            'eligible_images': len(unique_images), 'sample_gate': bool(sample_pass),
            'numeric_gate': bool(numeric_pass), 'effect_gates': bool(effect_pass),
            'cases': cases, 'model_forwards': 0, 'weights_loaded': 0, 'training_started': False,
            'scope': P['interpretation'], 'elapsed_seconds': time.monotonic()-START}


def report(result, counts):
    lines = ['368｜已知量化机制下的零训练目标质量预检',
             '## Material Passport', '- Origin Skill: academic-research-suite / experiment-agent',
             '- Origin Mode: run', '- Origin Date: '+datetime.now(timezone.utc).isoformat(),
             '- Verification Status: UNVERIFIED (未作独立完整复跑；数学校验与数值精度检查见附件)',
             '- Version Label: quantization_precheck_v1', '',
             '实际结论：'+result['verdict'],
             f'样本：每 seed {result["eligible_gt_per_seed"]} 个 GT，{result["eligible_images"]} 张图。',
             '原始 tiny cohort 为4952；按预先冻结的正宽高和图像边界条件筛选，未按 IoU 选样。',
             json.dumps(counts, ensure_ascii=False), '',
             '参考是原整数标注框；本次重新施加已知随机相位量化，不称物理真框。',
             '使用缓存的最佳参考 IoU query，任意类别，所有目标共用。下面不代表 AP 或召回率。',
             '主候选：均匀 xywh 集合期望 E[IoU^1.5]；最大值仅描述。', '']
    for key, case in result['cases'].items():
        lines += [key, 'arm          MSE             RMSE          MAE           Spearman']
        for arm, m in case['metrics'].items():
            lines.append(f'{arm:10} {m["MSE"]:.10f}  {m["RMSE"]:.8f}  {m["MAE"]:.8f}  {m["Spearman"]:.8f}')
        for arm, c in case['comparisons'].items():
            lines += [f'expected vs {arm}: MSE下降 {100*c["relative_MSE_reduction"]:.4f}%，rho差 {c["rho_advantage"]:+.8f}',
                      f'  MSE改善95%CI={c["MSE_advantage_CI95"]}；rho改善95%CI={c["rho_advantage_CI95"]}',
                      '  各门槛：'+json.dumps(c['gates'], ensure_ascii=False),
                      '  对照通过='+str(c['passed'])]
        lines += ['8/12阶精度：'+json.dumps(case['numeric_audit'], ensure_ascii=False), '']
    lines += [f'样本门槛={result["sample_gate"]}；数值门槛={result["numeric_gate"]}；全部效应门槛={result["effect_gates"]}',
              '执行：仅 CPU 读取缓存、几何计算和图像配对 bootstrap。加载模型0，前向0，训练0。',
              '停止边界：失败后不换主候选、不改样本或取整规则。通过也只进入方法定义审查，不自动训练。',
              '局限：合成附加量化的目标保真度；真实量化器未知，均匀积分不是已识别的真实后验；两seed共享图像，不能当两份独立数据；不证明 AP、物理定位改善或创新性。']
    with (ROOT/'368_零训练预检结果.txt').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')


def main():
    global LOG, P
    torch.set_num_threads(2)
    protocol_bytes = (ROOT/'protocol.json').read_bytes()
    P = json.loads(protocol_bytes)
    with (ROOT/'frozen_protocol.sha256').open('x', encoding='utf-8') as f:
        f.write(sha(protocol_bytes)+'  protocol.json\n')
    LOG = (ROOT/'run.log').open('x', encoding='utf-8')
    dump('run_metadata.json', {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
          'command': sys.argv, 'python_executable': sys.executable, 'python': platform.python_version(),
          'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
          'protocol_sha256': sha(protocol_bytes), 'script_sha256': sha(Path(__file__).read_bytes()),
          'device': 'CPU', 'torch_threads': 2})
    try:
        log('Protocol frozen; starting independent mathematical implementation checks')
        math_audit()
        data, counts = load_pairs()
        cases = run_targets(data)
        result = evaluate(cases, data)
        dump('results.json', result)
        report(result, counts)
        log('COMPLETED: '+result['verdict'])
    except Exception as e:
        dump('failure.json', {'type': type(e).__name__, 'message': str(e),
                             'elapsed_seconds': time.monotonic()-START})
        raise
    finally:
        LOG.close()
    manifest = [{'path': p.name, 'size': p.stat().st_size, 'sha256': sha(p.read_bytes())}
                for p in sorted(ROOT.iterdir()) if p.is_file()]
    dump('delivery_manifest.json', {'files': manifest, 'self_not_hashed': True})


if __name__ == '__main__':
    main()
