"""128 + 135 preregistered cached PMF audit; corrected native FP32 interface.

Written without reading 127 PMF quality. Execute only by the authorized root.
The common-translation compatibility is axis-factorized, not a new 2D copula.
No model/optimizer/forward/backward. Original gate and method are unchanged.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time
import traceback
import types

sys.dont_write_bytecode = True
# Use the project's existing CPU runtime without importing model packages.
_CPU_RUNTIME = pathlib.Path(__file__).resolve().parents[1] / 'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'
sys.path.insert(0, str(_CPU_RUNTIME))
import numpy as np
import psutil
import torch

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[1]
NATIVE = WORK / '英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/deim'
ORIGINAL_PROTOCOL = ROOT / '128_C0共同平移PMF_兼容质量事前协议.txt'
PROTOCOL = ROOT / '135_C0共同平移PMF_FP32接口事前勘误.txt'
CHECKPOINT_SHA = '9d6c0dd7a9cd6cb56465c4eab9416c0e89547c7e45df936e559fa12b921f275e'
AXIS = np.array([-2., -1., 0., 1., 2.], dtype=np.float64)
OFFSETS = np.array([(x, y) for x in AXIS for y in AXIS], dtype=np.float64)
ZERO_INDEX = 12
MOMENT_TOL = 1e-10
STRICT_TOL = 1e-10
BOUNDARY_TOL = 1e-12
MAX_BISECT = 160
MAX_BRACKET = 50
MAX_PARAMETER = 1e12


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return '-inf' if value == -math.inf else ('+inf' if value == math.inf else 'NaN')
    return value


def save(path, obj):
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def resource_limits():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    process = psutil.Process()
    allowed = process.cpu_affinity()
    require({13, 15}.issubset(allowed), 'required CPU affinity unavailable')
    process.cpu_affinity([13, 15])
    affinity = process.cpu_affinity()
    require(sorted(affinity) == [13, 15], 'unexpected CPU affinity')
    available_ram = psutil.virtual_memory().available / 2**30
    require(available_ram > 8, 'RAM guard')
    priority = 'unchanged'
    if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        priority = 'below_normal'
    return {'affinity': affinity, 'priority': priority, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'torch': torch.__version__,
            'numpy': np.__version__, 'available_RAM_GiB': available_ram}


def readable_mass(log_mass):
    if log_mass == -math.inf:
        return 0.
    value = math.exp(log_mass)
    return value if value > 0. else 'positive_below_float64_range'


def scalar(value, name):
    arr = torch.as_tensor(value).detach().cpu().reshape(-1)
    require(arr.numel() == 1, name + ' must be scalar')
    result = float(arr.item())
    require(math.isfinite(result), name + ' nonfinite')
    return result


def array(value, name, shape=None, integer=False):
    tensor = torch.as_tensor(value).detach().cpu()
    if shape is not None:
        require(tuple(tensor.shape) == tuple(shape), name + ' unexpected shape ' + str(tuple(tensor.shape)))
    result = tensor.numpy().astype(np.int64 if integer else np.float64, copy=True)
    require(np.all(np.isfinite(result)), name + ' nonfinite')
    if integer:
        require(np.array_equal(tensor.numpy(), result), name + ' is not integral')
    return result


def logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isneginf(values)):
        return -math.inf
    require(not np.any(np.isnan(values)) and not np.any(np.isposinf(values)), 'invalid log weights')
    top = float(np.max(values))
    return top + math.log(float(np.exp(values - top).sum()))


def normalized(logits):
    z = logsumexp(logits)
    require(math.isfinite(z), 'empty probability support')
    return np.asarray(logits, dtype=np.float64) - z


def moments(logp):
    p = np.exp(logp)
    mean = float(np.dot(p, AXIS))
    second = float(np.dot(p, AXIS * AXIS))
    positive = p > 0
    entropy = -float(np.dot(p[positive], logp[positive]))
    return {'sum': float(p.sum()), 'mean': mean, 'variance': second - mean * mean, 'entropy': entropy}


def zero_mean_projection(logits, support):
    """Min-KL projection with exact structural boundary treatment."""
    support = np.asarray(support, dtype=bool)
    require(support.shape == (5,) and support[2], 'axis support must contain zero')
    require(np.all(np.isfinite(np.asarray(logits)[support])), 'supported logit nonfinite')
    d = AXIS[support]
    if not (np.any(d < 0) and np.any(d > 0)):
        answer = np.full(5, -math.inf)
        answer[2] = 0.
        return answer, {'kind': 'structural_delta_zero', 'lambda': None, 'iterations': 0}
    base = np.where(support, np.asarray(logits, dtype=np.float64), -math.inf)

    def at(lam):
        lp = normalized(base + lam * AXIS)
        return moments(lp)['mean'], lp

    mean0, lp0 = at(0.)
    if abs(mean0) <= 1e-12:
        return lp0, {'kind': 'finite', 'lambda': 0., 'iterations': 0}
    low, high = -1., 1.
    for _ in range(MAX_BRACKET):
        ml, _ = at(low)
        mh, _ = at(high)
        if ml <= 0. <= mh:
            break
        if ml > 0.:
            low *= 2.
        if mh < 0.:
            high *= 2.
        require(max(abs(low), abs(high)) <= MAX_PARAMETER, 'zero-mean bracket exceeded')
    else:
        raise RuntimeError('zero-mean bracket unavailable')
    for iteration in range(MAX_BISECT):
        middle = (low + high) / 2.
        mm, lp = at(middle)
        if abs(mm) <= 1e-12:
            break
        if mm < 0.:
            low = middle
        else:
            high = middle
    else:
        raise RuntimeError('zero-mean bisection did not converge')
    stats = moments(lp)
    require(abs(stats['mean']) <= MOMENT_TOL, 'zero-mean moment failed')
    require(abs(stats['sum'] - 1.) <= 1e-12, 'zero-mean normalization failed')
    return lp, {'kind': 'finite', 'lambda': middle, 'iterations': iteration + 1}


def maximum_entropy_same_moments(logq, support):
    """exp(alpha*d+beta*d^2), matching mean zero and Q variance."""
    support = np.asarray(support, dtype=bool)
    target = moments(logq)
    d = AXIS[support]
    if not (np.any(d < 0) and np.any(d > 0)):
        require(abs(target['variance']) <= 1e-15 and abs(target['mean']) <= 1e-15,
                'structural boundary target is not delta zero')
        answer = np.full(5, -math.inf)
        answer[2] = 0.
        return answer, {'kind': 'structural_delta_zero', 'alpha': None, 'beta': None}
    variance = target['variance']
    max_variance = -float(d.min() * d.max())
    require(variance > BOUNDARY_TOL and max_variance - variance > BOUNDARY_TOL,
            'nonstructural variance numerically on boundary; failclosed')

    def at(beta):
        lp, projection = zero_mean_projection(beta * AXIS * AXIS, support)
        return moments(lp)['variance'], lp, projection

    v0, lp0, meta0 = at(0.)
    if abs(v0 - variance) <= 1e-12:
        answer, beta, meta, iterations = lp0, 0., meta0, 0
    else:
        low, high = -1., 1.
        for _ in range(MAX_BRACKET):
            vl, _, _ = at(low)
            vh, _, _ = at(high)
            if vl <= variance <= vh:
                break
            if vl > variance:
                low *= 2.
            if vh < variance:
                high *= 2.
            require(max(abs(low), abs(high)) <= MAX_PARAMETER, 'maxentropy bracket exceeded')
        else:
            raise RuntimeError('maxentropy variance bracket unavailable')
        for iteration in range(MAX_BISECT):
            beta = (low + high) / 2.
            vm, answer, meta = at(beta)
            if abs(vm - variance) <= 1e-12:
                break
            if vm < variance:
                low = beta
            else:
                high = beta
        else:
            raise RuntimeError('maxentropy bisection did not converge')
        iterations = iteration + 1
    stats = moments(answer)
    require(abs(stats['mean']) <= MOMENT_TOL, 'maxentropy mean failed')
    require(abs(stats['variance'] - variance) <= MOMENT_TOL, 'maxentropy variance failed')
    require(abs(stats['sum'] - 1.) <= 1e-12, 'maxentropy normalization failed')
    require(stats['entropy'] + 1e-9 >= target['entropy'], 'maxentropy entropy is lower than Q')
    return answer, {'kind': 'finite', 'alpha': meta['lambda'], 'beta': beta,
                    'outer_iterations': iterations, 'variance': variance,
                    'theoretical_max_variance': max_variance}


def solver_self_checks():
    """Deterministic mathematical checks; no model and no random samples."""
    full = np.ones(5, dtype=bool)
    lp, _ = zero_mean_projection(np.zeros(5), full)
    require(np.max(np.abs(np.exp(lp) - .2)) < 1e-12, 'uniform self-check')
    one_sided = np.array([False, False, True, True, True])
    lp, meta = zero_mean_projection(np.arange(5, dtype=np.float64), one_sided)
    require(meta['kind'] == 'structural_delta_zero' and lp[2] == 0., 'boundary self-check')
    mp, _ = maximum_entropy_same_moments(lp, one_sided)
    require(np.array_equal(lp, mp), 'boundary maxentropy self-check')
    asymmetric = np.array([False, True, True, True, True])
    lp, _ = zero_mean_projection(np.array([0., -1.2, .7, -.3, -2.]), asymmetric)
    mp, _ = maximum_entropy_same_moments(lp, asymmetric)
    require(abs(moments(mp)['mean']) <= MOMENT_TOL, 'asymmetric mean self-check')
    require(abs(moments(mp)['variance'] - moments(lp)['variance']) <= MOMENT_TOL,
            'asymmetric variance self-check')
    return {'uniform': True, 'one_sided_exact_delta': True, 'asymmetric_two_parameter': True}


def load_native_math():
    # Bypass native package __init__ and all model registrations.
    name = '_audit136_native_math'
    package = types.ModuleType(name)
    package.__path__ = [str(NATIVE)]
    sys.modules[name] = package
    for module_name in ('box_ops', 'dfine_utils'):
        spec = importlib.util.spec_from_file_location(name + '.' + module_name, NATIVE / (module_name + '.py'))
        require(spec is not None and spec.loader is not None, 'native math import unavailable')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[name + '.dfine_utils']


def xyxy(boxes):
    return np.concatenate((boxes[..., :2] - boxes[..., 2:] / 2.,
                           boxes[..., :2] + boxes[..., 2:] / 2.), axis=-1)


def iou(a, b):
    aa, bb = xyxy(a), xyxy(b)
    inter_wh = np.maximum(np.minimum(aa[..., 2:], bb[..., 2:]) - np.maximum(aa[..., :2], bb[..., :2]), 0.)
    intersection = inter_wh[..., 0] * inter_wh[..., 1]
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - intersection
    require(np.all(union > 0.), 'nonpositive IoU union')
    return intersection / union


def match_capacity_one(edges):
    """Deterministic Kuhn: ascending q, ascending target-array index."""
    nq, ng = edges.shape
    adjacency = [np.flatnonzero(edges[q]).tolist() for q in range(nq)]
    owner = [-1] * ng

    def augment(q, seen):
        for g in adjacency[q]:
            if seen[g]:
                continue
            seen[g] = True
            if owner[g] < 0 or augment(owner[g], seen):
                owner[g] = q
                return True
        return False

    for q in range(nq):
        augment(q, [False] * ng)
    pairs = sorted((q, g) for g, q in enumerate(owner) if q >= 0)
    require(len({q for q, _ in pairs}) == len(pairs), 'matching duplicated query')
    return pairs


def allowed_classes(pred_logits):
    logits = torch.as_tensor(pred_logits).detach().cpu()
    require(tuple(logits.shape) == (300, 10), 'class logits shape')
    # Preserve the cached dtype and the same native flattened topk rule as 121.
    _, flat = logits.sigmoid().flatten().topk(300)
    allow = np.zeros((300, 10), dtype=bool)
    flat = flat.numpy()
    allow[flat // 10, flat % 10] = True
    return allow, flat.tolist()


def physical_four_lens64(ref64, target_xyxy64, reg_scale):
    """Unchanged independent FP64 physical-support calculation."""
    require(ref64.dtype == target_xyxy64.dtype == torch.float64, 'physical geometry must be FP64')
    rs = abs(float(reg_scale))
    return torch.stack(((ref64[:, 0] - target_xyxy64[:, 0]) / (ref64[:, 2] / rs + 1e-16) - .5 * rs,
                        (ref64[:, 1] - target_xyxy64[:, 1]) / (ref64[:, 3] / rs + 1e-16) - .5 * rs,
                        (target_xyxy64[:, 2] - ref64[:, 0]) / (ref64[:, 2] / rs + 1e-16) - .5 * rs,
                        (target_xyxy64[:, 3] - ref64[:, 1]) / (ref64[:, 3] / rs + 1e-16) - .5 * rs), dim=-1)


def audit_native_interpolation(raw32, indices, wr, wl, project32, reg_max):
    """Audit actual native FP32 input, separating clamping from precision loss."""
    require(raw32.dtype == project32.dtype == torch.float32, 'native raw/nodes must be FP32')
    require(indices.dtype == wr.dtype == wl.dtype == torch.float32, 'native indices/weights must be FP32')
    shape = tuple(raw32.shape)
    ids = indices.long().reshape(shape)
    wr, wl = wr.reshape(shape), wl.reshape(shape)
    require(bool(torch.isfinite(raw32).all()), 'native four_lens nonfinite')
    require(bool(torch.all(ids >= 0) and torch.all(ids < reg_max)), 'native interpolation bin invalid')
    require(bool(torch.all(wr >= 0.) and torch.all(wl >= 0.)), 'negative interpolation weight')
    require(float(torch.max(torch.abs(wr + wl - 1.))) <= 1e-12, 'native weights do not sum to 1')
    nodes64 = project32.double()
    clipped_raw32 = raw32.clamp(min=project32.min(), max=project32.max())
    target64 = clipped_raw32.double()
    rebuilt64 = wl.double() * nodes64[ids] + wr.double() * nodes64[ids + 1]
    interpolation_error = float(torch.max(torch.abs(rebuilt64 - target64)))
    require(interpolation_error <= 5e-7, 'native FP32 interpolation reconstruction failed')
    exact_right = (target64 - nodes64[ids]) / (nodes64[ids + 1] - nodes64[ids])
    weight_error = float(torch.max(torch.abs(wr.double() - exact_right)))
    require(weight_error <= 1e-7, 'native FP32 weights differ from same-input two-node interpolation')
    inside32 = (raw32 >= project32.min()) & (raw32 <= project32.max())
    return ids, wr, wl, {'interpolation_max_error': interpolation_error,
                         'weight_max_error': weight_error,
                         'native_outside_edge_count': int((~inside32).sum()),
                         'native_endpoint_clamp_max': float(torch.max(torch.abs(raw32.double() - target64)))}


def encode_native_fp32(native, ref64, candidate_cxcywh64, reg_max, up32, scale32):
    """Keep FP64 support; invoke the actual production target encoder in FP32."""
    require(ref64.dtype == candidate_cxcywh64.dtype == torch.float64, 'geometry input must be FP64')
    require(up32.dtype == scale32.dtype == torch.float32 and tuple(up32.shape) == (1,)
            and tuple(scale32.shape) == (1,), 'native up/scale must be FP32 tensor[1]')
    project32 = native.weighting_function(reg_max, up32, scale32)
    require(project32.dtype == torch.float32 and tuple(project32.shape) == (reg_max + 1,), 'native node contract')
    project64 = project32.double()
    target64 = torch.cat((candidate_cxcywh64[:, :2] - candidate_cxcywh64[:, 2:] / 2.,
                          candidate_cxcywh64[:, :2] + candidate_cxcywh64[:, 2:] / 2.), dim=-1)
    raw64 = physical_four_lens64(ref64, target64, float(scale32.item()))
    physical_edge_support = (raw64 >= project64.min()) & (raw64 <= project64.max())
    ref32 = ref64.float()
    box32 = candidate_cxcywh64.float()
    box_ops = sys.modules[native.__package__ + '.box_ops']
    target32 = box_ops.box_cxcywh_to_xyxy(box32)
    captured = []
    original_translate = native.translate_gt

    def observe_translate(gt, rm, rs, up):
        captured.append(gt.detach().clone())
        return original_translate(gt, rm, rs, up)

    native.translate_gt = observe_translate
    try:
        indices, wr, wl = native.bbox2distance(ref32, target32, reg_max, scale32, up32)
    finally:
        native.translate_gt = original_translate
    require(len(captured) == 1, 'bbox2distance must call translate_gt exactly once')
    raw32 = captured[0]
    require(tuple(raw32.shape) == tuple(raw64.shape) and raw32.dtype == torch.float32,
            'native four_lens capture is not the production FP32 tensor')
    ids, wr, wl, audit = audit_native_interpolation(raw32, indices, wr, wl, project32, reg_max)
    native_edge_support = (raw32 >= project32.min()) & (raw32 <= project32.max())
    physical_point_support = physical_edge_support.all(-1)
    native_point_support = native_edge_support.all(-1)
    audit.update({'four_lens32_minus64_max': float(torch.max(torch.abs(raw32.double() - raw64))),
                  'cxcywh_quantization_max': float(torch.max(torch.abs(box32.double() - candidate_cxcywh64))),
                  'xyxy_quantization_max': float(torch.max(torch.abs(target32.double() - target64))),
                  'physical64_inside_native32_outside_edges': int((physical_edge_support & ~native_edge_support).sum()),
                  'physical64_outside_native32_inside_edges': int((~physical_edge_support & native_edge_support).sum()),
                  'physical64_inside_native32_outside_points': int((physical_point_support & ~native_point_support).sum()),
                  'physical64_outside_native32_inside_points': int((~physical_point_support & native_point_support).sum()),
                  'native_ref_dtype': str(ref32.dtype), 'native_target_dtype': str(target32.dtype),
                  'native_four_lens_dtype': str(raw32.dtype), 'native_weight_dtype': str(wr.dtype)})
    return {'indices': ids, 'wr': wr, 'wl': wl, 'support': physical_point_support.numpy(),
            'raw64': raw64, 'raw32': raw32, 'project32': project32, 'audit': audit}


def native_encoding_self_checks(native):
    """Fixed synthetic checks BEFORE reading/loading the 127 cache."""
    rm = 32
    up = torch.tensor([.5], dtype=torch.float32)
    scale = torch.tensor([4.], dtype=torch.float32)
    nodes = native.weighting_function(rm, up, scale)
    require(nodes.dtype == torch.float32 and tuple(nodes.shape) == (33,), 'synthetic node shape/dtype')
    require(bool(torch.all(nodes[1:] > nodes[:-1])), 'synthetic nodes not increasing')
    ref32 = torch.tensor([[.5, .5, .25, .125]], dtype=torch.float32)
    box64 = torch.tensor([[.503, .497, .2, .1]], dtype=torch.float64)
    encoded = encode_native_fp32(native, ref32.double(), box64, rm, up, scale)
    require(bool(encoded['support'][0]), 'synthetic interior is not physically supported')
    # Preserve a regression check for the rejected old API, without using it in analysis.
    rejected = None
    target64 = torch.cat((box64[:, :2] - box64[:, 2:] / 2., box64[:, :2] + box64[:, 2:] / 2.), dim=-1)
    try:
        native.bbox2distance(ref32.double(), target64, rm, scale, up)
    except RuntimeError as exc:
        rejected = str(exc)
        require('dtype' in rejected.lower() and 'destination' in rejected.lower(),
                'unexpected exception in synthetic FP64 rejection check')
    require(rejected is not None, 'known FP64 native-interface rejection no longer reproduced; review contract')
    # Direct interpolation checks include exact support endpoints and outside values.
    direct32 = torch.stack((nodes[0] - 1., nodes[0], (nodes[15] + nodes[16]) / 2.,
                            nodes[16], nodes[-1], nodes[-1] + 1.))
    ix, wr, wl = native.translate_gt(direct32, rm, scale, up)
    _, _, _, direct_audit = audit_native_interpolation(direct32, ix, wr, wl, nodes, rm)
    endpoint_support = ((direct32.double() >= nodes[0].double()) &
                        (direct32.double() <= nodes[-1].double())).tolist()
    require(endpoint_support == [False, True, True, True, True, False], 'synthetic endpoint support wrong')
    # A valid box with opposite horizontal edges at the two support endpoints.
    distance32 = torch.stack((nodes[0], nodes[16], nodes[-1], nodes[16]))[None, :]
    endpoint_box64 = native.distance2bbox(ref32, distance32, scale).double()
    endpoint = encode_native_fp32(native, ref32.double(), endpoint_box64, rm, up, scale)
    require(bool(endpoint['support'][0]), 'exact synthetic endpoint box rejected')
    outside64 = endpoint_box64.clone()
    outside64[:, 0] += 2. ** -30
    rounded = encode_native_fp32(native, ref32.double(), outside64, rm, up, scale)
    require(not bool(rounded['support'][0]), 'FP64 outside point incorrectly admitted by native clipping')
    require(rounded['audit']['physical64_outside_native32_inside_points'] == 1,
            'synthetic rounded-inside counterexample not reproduced')
    return {'FP32_nodes': True, 'corrected_FP64_geometry_to_FP32_encoder': encoded['audit'],
            'direct_FP64_native_call_expected_rejection': rejected,
            'direct_endpoint_audit': direct_audit, 'direct_endpoint_support': endpoint_support,
            'exact_endpoint_box': endpoint['audit'], 'outside64_rounds_inside32_but_is_rejected': rounded['audit']}


def translation_energy(record, q, native, reg_max, up, reg_scale):
    boxes = torch.as_tensor(record['pred_boxes']).detach().cpu().double()
    refs = torch.as_tensor(record['ref_points_initial']).detach().cpu().double()
    logits = torch.as_tensor(record['final_corners']).detach().cpu().double()[q].reshape(4, reg_max + 1)
    native_scale = torch.tensor([reg_scale], dtype=torch.float32)
    b = boxes[q].repeat(25, 1)
    b[:, :2] += torch.from_numpy(OFFSETS) / 640.
    ref = refs[q].repeat(25, 1)
    encoded = encode_native_fp32(native, ref, b, reg_max, up, native_scale)
    support = encoded['support']
    require(bool(support[ZERO_INDEX]), 'native final d=0 outside support')
    indices, wr, wl = encoded['indices'], encoded['wr'], encoded['wl']
    logp = torch.log_softmax(logits, dim=-1)
    edge = torch.arange(4)[None, :].expand(25, 4)
    energy = (wl * logp[edge, indices] + wr * logp[edge, indices + 1]).sum(dim=-1).numpy()
    energy[~support] = -math.inf
    grid = energy.reshape(5, 5)
    support_grid = support.reshape(5, 5)
    sx, sy = support_grid[:, 2], support_grid[2, :]
    require(np.array_equal(support_grid, sx[:, None] & sy[None, :]), 'physical support does not factorize')
    additive = grid[:, 2, None] + grid[2, None, :] - grid[2, 2]
    factor_error = float(np.max(np.abs(additive[support_grid] - grid[support_grid])))
    require(factor_error <= 1e-9, 'common-translation energy does not factorize')
    return {'energy': energy, 'support': support, 'energy_x': grid[:, 2].copy(),
            'energy_y': grid[2, :].copy(), 'support_x': sx.copy(), 'support_y': sy.copy(),
            'interpolation_max_error': encoded['audit']['interpolation_max_error'],
            'weight_max_error': encoded['audit']['weight_max_error'],
            'precision_audit': encoded['audit'],
            'factorization_max_error': factor_error}


def priors_from_energy(energy):
    axes = []
    for axis in ('x', 'y'):
        support = energy['support_' + axis]
        q, qmeta = zero_mean_projection(energy['energy_' + axis], support)
        m, mmeta = maximum_entropy_same_moments(q, support)
        axes.append({'Q': q, 'M': m, 'Q_solver': qmeta, 'M_solver': mmeta,
                     'Q_moments': moments(q), 'M_moments': moments(m)})
    logq = (axes[0]['Q'][:, None] + axes[1]['Q'][None, :]).reshape(-1)
    logm = (axes[0]['M'][:, None] + axes[1]['M'][None, :]).reshape(-1)
    logu = np.full(25, -math.inf)
    logu[energy['support']] = -math.log(int(energy['support'].sum()))
    for name, lp in [('Q', logq), ('M', logm), ('U', logu)]:
        require(abs(float(np.exp(lp).sum()) - 1.) <= 1e-12, name + ' joint normalization failed')
        require(np.all(np.isneginf(lp[~energy['support']])), name + ' has mass outside physical support')
    return {'logQ': logq, 'logM': logm, 'logU': logu, 'axes': axes}


def summarize(rows):
    scored = [r for r in rows if r['included_in_mean']]
    scores = {}
    for arm in ('Q', 'U', 'M'):
        logs = [r['log_mass'][arm] for r in scored]
        mean = None if not logs else (-math.inf if any(x == -math.inf for x in logs) else float(np.mean(logs)))
        scores[arm] = {'mean_log_mass': mean, 'zero_mass_pairs': sum(x == -math.inf for x in logs)}
    qv, uv, mv = (scores[k]['mean_log_mass'] for k in ('Q', 'U', 'M'))
    gate = bool(qv is not None and math.isfinite(qv)
                and qv > uv + STRICT_TOL and qv > mv + STRICT_TOL)
    return {'continuous_recoverable_main_pairs': len(rows), 'scored_pairs': len(scored),
            'no_25grid_correction_pairs': sum(not r['included_in_mean'] for r in rows),
            'grid_correction_but_none_in_physical_support': sum(
                r['included_in_mean'] and r['supported_correcting_offsets'] == 0 for r in rows),
            'scores': scores, 'strict_numerical_tolerance': STRICT_TOL, 'gate': gate}


def analyze(cache, native):
    require(isinstance(cache, dict) and 'records' in cache, 'cache schema: records missing')
    require(cache['checkpoint_sha256'] == CHECKPOINT_SHA, 'unexpected checkpoint declaration')
    require(cache['weight_key'] == 'model_online', 'unexpected weight key')
    reg_max = scalar(cache['reg_max'], 'reg_max')
    require(reg_max == 32., 'reg_max must be 32')
    reg_max = int(reg_max)
    reg_scale = scalar(cache['reg_scale'], 'reg_scale')
    up_value = scalar(cache['up'], 'up')
    require(abs(reg_scale) > 0. and abs(up_value) > 0., 'invalid native scale/up')
    # C0 native eval is FP32 and has not been convert_to_deploy()'d. Rebuild
    # actual FP32 nodes first, then use FP64 for cached geometry/energy analysis.
    up = torch.tensor([up_value], dtype=torch.float32)
    records = cache['records']
    require(isinstance(records, list) and len(records) == 4, 'exactly four records required')
    require(len({int(r['image_index']) for r in records}) == 4, 'duplicate image_index')
    records = sorted(records, key=lambda r: int(r['image_index']))
    results, main_rows = [], []
    gt_count = sub4_count = 0
    reconstruction_max = 0.
    reproduction = {key: 0 for key in ('original_class05', 'bound_class05', 'original_class075',
                    'bound_class075', 'original_geo05', 'bound_geo05', 'original_geo075', 'bound_geo075')}
    project = native.weighting_function(reg_max, up, torch.tensor([reg_scale], dtype=torch.float32)).double()
    require(tuple(project.shape) == (33,) and bool(torch.all(project[1:] > project[:-1])), 'invalid native nodes')
    for record in records:
        require(psutil.virtual_memory().available / 2**30 > 8, 'RAM guard')
        index, event = int(record['image_index']), str(record['event'])
        ihash = record['input_sha256']
        require(isinstance(ihash, str) and len(ihash) == 64 and all(c in '0123456789abcdef' for c in ihash.lower()),
                'invalid input_sha256')
        boxes = array(record['pred_boxes'], 'pred_boxes', (300, 4))
        array(record['pred_logits'], 'pred_logits', (300, 10))
        refs = array(record['ref_points_initial'], 'ref_points_initial', (300, 4))
        corners = array(record['final_corners'], 'final_corners', (300, 132))
        require(torch.as_tensor(record['final_corners']).dtype == torch.float32, 'C0 native corners must be FP32')
        selected = array(record['selected_indices'], 'selected_indices', (300,), integer=True)
        require(np.all((selected >= 0) & (selected < 8400)), 'selected index outside P3/P4/P5')
        gt = array(record['target']['boxes'], 'target.boxes')
        require(gt.ndim == 2 and gt.shape[1] == 4, 'GT shape')
        labels = array(record['target']['labels'], 'target.labels', (len(gt),), integer=True)
        require(np.all((labels >= 0) & (labels < 10)), 'invalid GT class')
        raw_small = torch.as_tensor(record['original_sub4']).detach().cpu()
        require(raw_small.dtype == torch.bool and tuple(raw_small.shape) == (len(gt),), 'original_sub4 must be bool')
        small = raw_small.numpy().copy()
        require(np.all(boxes[:, 2:] > 0.) and np.all(refs[:, 2:] > 0.) and np.all(gt[:, 2:] > 0.),
                'nonpositive box dimensions')
        gt_count += len(gt)
        sub4_count += int(small.sum())
        expected_distance = (torch.softmax(torch.from_numpy(corners).reshape(300, 4, 33), dim=-1) * project).sum(-1)
        rebuilt = native.distance2bbox(torch.from_numpy(refs), expected_distance, reg_scale).numpy()
        rec_error = float(np.max(np.abs(rebuilt - boxes)))
        reconstruction_max = max(reconstruction_max, rec_error)
        require(rec_error <= 1e-6, 'corners/ref do not reconstruct cached final boxes')
        allow, flat_top300 = allowed_classes(record['pred_logits'])
        class_edges = allow[:, labels]
        original = iou(boxes[:, None, :], gt[None, :, :])
        candidates = np.broadcast_to(boxes[:, None, :], (300, len(gt), 4)).copy()
        p3 = selected < 6400
        candidates[p3, :, :2] = np.clip(gt[None, :, :2], boxes[p3, None, :2] - 2. / 640.,
                                                     boxes[p3, None, :2] + 2. / 640.)
        bound = iou(candidates, gt[None, :, :])
        require(np.all(bound + 1e-12 >= original), 'continuous bound below original')
        for suffix, threshold in [('05', .5), ('075', .75)]:
            for arm, matrix in [('original', original), ('bound', bound)]:
                reproduction[arm + '_class' + suffix] += len(match_capacity_one((matrix >= threshold)[:, small] & class_edges[:, small]))
                reproduction[arm + '_geo' + suffix] += len(match_capacity_one((matrix >= threshold)[:, small]))
        matching = match_capacity_one((bound >= .5) & class_edges)
        recoverable = (original < .5) & (bound >= .5) & small[None, :]
        true_recoverable = recoverable & class_edges
        all_edges = np.argwhere(true_recoverable)
        image_main = [(q, g) for q, g in matching if true_recoverable[q, g]]
        energy_cache, prior_cache = {}, {}
        geometric_rows = []
        for q, g in all_edges.tolist():
            if q not in energy_cache:
                energy_cache[q] = translation_energy(record, q, native, reg_max, up, reg_scale)
            candidate_boxes = np.broadcast_to(boxes[q], (25, 4)).copy()
            candidate_boxes[:, :2] += OFFSETS / 640.
            discrete = iou(candidate_boxes, gt[g]) >= .5
            geometric_rows.append({'query': q, 'gt_index': g, 'original_iou': float(original[q, g]),
                                   'bound_iou': float(bound[q, g]), 'correcting_25grid_offsets': int(discrete.sum()),
                                   'supported_correcting_offsets': int((discrete & energy_cache[q]['support']).sum()),
                                   'in_main_matching': (q, g) in image_main})
        for q, g in image_main:
            energy = energy_cache[q]
            candidate_boxes = np.broadcast_to(boxes[q], (25, 4)).copy()
            candidate_boxes[:, :2] += OFFSETS / 640.
            discrete_iou = iou(candidate_boxes, gt[g])
            correcting = discrete_iou >= .5
            row = {'image_index': index, 'event': event, 'query': q, 'gt_index': g,
                   'original_iou': float(original[q, g]), 'bound_iou': float(bound[q, g]),
                   'included_in_mean': bool(correcting.any()), 'correcting_25grid_offsets': int(correcting.sum()),
                   'supported_correcting_offsets': int((correcting & energy['support']).sum()),
                   'grid_iou': discrete_iou, 'physical_support': energy['support'], 'energy': energy['energy'],
                   'native_interpolation_max_error': energy['interpolation_max_error'],
                   'native_weight_max_error': energy['weight_max_error'],
                   'precision_audit': energy['precision_audit'],
                   'factorization_max_error': energy['factorization_max_error']}
            if row['included_in_mean']:
                if q not in prior_cache:
                    prior_cache[q] = priors_from_energy(energy)
                priors = prior_cache[q]
                log_mass = {arm: logsumexp(priors['log' + arm][correcting]) for arm in ('Q', 'U', 'M')}
                row.update({'log_mass': log_mass, 'mass': {k: readable_mass(v) for k, v in log_mass.items()},
                            'log_priors': {arm: priors['log' + arm] for arm in ('Q', 'U', 'M')},
                            'axis_solvers': priors['axes']})
            else:
                row.update({'reason': 'continuous_recoverable_but_no_25grid_correction', 'log_mass': None, 'mass': None})
            main_rows.append(row)
        image_rows = [r for r in main_rows if r['image_index'] == index]
        results.append({'image_index': index, 'event': event, 'input_sha256': ihash, 'GT': len(gt),
                        'sub4': int(small.sum()), 'P3_queries': int(p3.sum()), 'reconstruction_max_error': rec_error,
                        'native_flat_top300': flat_top300, 'all_GT_maximum_matching': matching,
                        'all_GT_class_bound_matching_size': len(matching),
                        'all_GT_geometry_bound_matching_size': len(match_capacity_one(bound >= .5)),
                        'all_recoverable_trueclass_sub4_edges': geometric_rows,
                        'all_recoverable_geometry_sub4_edge_count': int(recoverable.sum()),
                        'main_summary': summarize(image_rows)})
    require(gt_count == 902 and sub4_count == 453, 'wrong four-image GT panel')
    expected = {'original_class05': 51, 'bound_class05': 89, 'original_class075': 10, 'bound_class075': 40,
                'original_geo05': 66, 'bound_geo05': 105, 'original_geo075': 15, 'bound_geo075': 49}
    require(reproduction == expected, '121 reproduction mismatch: ' + str(reproduction))
    return {'state': 'complete', 'checkpoint_sha256': CHECKPOINT_SHA, 'weight_key': 'model_online',
            'reg_max': reg_max, 'up': up_value, 'reg_scale': reg_scale, 'offsets_pixels': OFFSETS,
            'GT': gt_count, 'sub4': sub4_count, 'final_box_reconstruction_max_error': reconstruction_max,
            'reproduction_121': reproduction, 'images': results, 'main_pairs': main_rows,
            'primary': summarize(main_rows), 'scope': 'Single C0 development panel; compatibility premise only, not AP, training permission, or established novelty.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=pathlib.Path, default=ROOT / '127_C0_attention_template_registration/01_cache.pt')
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / '137_C0_pmf_translation_prior_fp32')
    args = parser.parse_args()
    args.out.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    files = {'cache': args.cache, 'script': pathlib.Path(__file__), 'protocol': PROTOCOL,
             'original_protocol': ORIGINAL_PROTOCOL,
             'native_dfine_utils': NATIVE / 'dfine_utils.py', 'native_box_ops': NATIVE / 'box_ops.py'}
    before = {}
    result = {'state': 'error', 'primary': {'gate': False}}
    resources = None
    checks = native_checks = None
    cache_loaded = False
    stage = 'resources'
    try:
        resources = resource_limits()
        before = {key: sha(path) for key, path in files.items() if key != 'cache'}
        stage = 'synthetic_solver_checks_before_cache'
        checks = solver_self_checks()
        native = load_native_math()
        stage = 'synthetic_native_encoding_checks_before_cache'
        native_checks = native_encoding_self_checks(native)
        stage = 'cache_load'
        before['cache'] = sha(args.cache)
        cache = torch.load(args.cache, map_location='cpu', weights_only=False)
        cache_loaded = True
        stage = 'cached_analysis'
        result = analyze(cache, native)
        stage = 'complete'
    except Exception as exc:
        result = {'state': 'error', 'primary': {'gate': False}, 'error': str(exc),
                  'traceback': traceback.format_exc(), 'scope': 'Failclosed; no inference about method utility.'}
    finally:
        # If a synthetic check failed, do not read the cache even just for hashing.
        after = {key: sha(files[key]) for key in before if files[key].exists()}
        unchanged = bool(before and before == after)
        if result['state'] == 'complete':
            unchanged = unchanged and set(before) == set(files)
        if not unchanged:
            result['state'] = 'error'
            result.setdefault('primary', {})['gate'] = False
            result['hash_error'] = 'Missing source or source file changed during analysis'
        result.update({'source_paths': {k: str(v.resolve()) for k, v in files.items()},
                       'sha256_before': before, 'sha256_after': after, 'sources_unchanged': unchanged,
                       'model_forwards': 0, 'model_backwards': 0, 'optimizer_steps': 0, 'new_training': False,
                       'resources': resources,
                       'stage': stage, 'cache_content_loaded': cache_loaded,
                       'solver_self_checks': checks, 'native_encoding_self_checks': native_checks,
                       'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                       'seconds': time.monotonic() - started})
        save(args.out / '00_results.json', result)
        print(json.dumps(jsonable({'state': result['state'], 'primary': result['primary'],
                                  'seconds': result['seconds'], 'output': str(args.out / '00_results.json')}),
                         ensure_ascii=False, allow_nan=False))
    if result['state'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
