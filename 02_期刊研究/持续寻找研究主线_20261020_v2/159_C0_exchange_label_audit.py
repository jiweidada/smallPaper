"""158: fixed 16x16 C0 exchange-label audit, cache only. Root reviews before run.

No model/optimizer/backward. Native matcher and MAL source methods are isolated
without changing their mathematics. Archived project COCO Python methods provide
greedy/PR semantics; this is an augmented four-image panel, never validation AP.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import logging
import pathlib
import sys
import time
import traceback
import types
from typing import Dict

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[1]
HELPER = ROOT / '136_C0_pmf_translation_prior_fp32.py'
HELPER_SHA = '10bc190cce57688cd8c04b31190fc356244cc2afd6df5b6c74f8bed4eca6ab26'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('frozen helper changed')
_spec = importlib.util.spec_from_file_location('_audit159_helpers', HELPER)
if _spec is None or _spec.loader is None:
    raise RuntimeError('helper unavailable')
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
np, torch, psutil = H.np, H.torch, H.psutil
require, sha, save, array = H.require, H.sha, H.save, H.array
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import scipy

PROTOCOL = ROOT / '158_排序交换标签核查_事前协议.txt'
NATIVE = H.NATIVE
RECIPE = WORK / '英文小论文/相位机制论文_20261018/data/source_snapshot/common.yml'
EVAL_SOURCE = WORK / '英文小论文/data/00_实验计划与证据索引/评价器旧二进制编译分支静态追溯_20261008_v1/archived_package/cocoeval.py'
WRAPPER = WORK / '英文小论文/code/p2_candidate_screen24_20261007_v1/native_reference_B/engine/data/dataset/coco_eval.py'
CACHE_SHA = 'f2fb2120ccd670f846a0a70651a3fabbf9bcfb636c94d43d33f8dc34eceac79c'
CHECKPOINT_SHA = '9d6c0dd7a9cd6cb56465c4eab9416c0e89547c7e45df936e559fa12b921f275e'
EXPECTED = {
    'helper': HELPER_SHA,
    'recipe': '25c158ef1cae63595512e2a57f5a60507cc06978b68724edb705923d44ea3157',
    'matcher': 'b20d4501bbaf1430daf63f1c82394762695a5b793a472bbba152383841983200',
    'criterion': '16cdd80b2197b8def8ee3f5f19f6311f5221c3da08abacb99f50d346610a2c30',
    'box_ops': '398fc107fc003b0fabaf36154594d50a8e0aa54195056b343031960c0aff0975',
    'dfine_utils': 'ad87ec5a50b1970e76b06f7242298ffa332beaf78c828ac3e8d41ac656ec5f6c',
    'postprocessor': '434fef8b5120b6f5273e40a2bfc6a31409fcb6bc743bcc8abbebf35268a7bcbe',
    'evaluator': '288eab6c1a82fe9c94f1eede61559aadb48dfe77940b0aee2ea587e355097948',
    'wrapper': 'fe04d0919fcac44ce0633503287d1cdb154ac391647eecdd3f3fb64849a1121d',
}
THRESHOLDS = np.linspace(.5, .95, 10)
PR_TOL = 1e-12
LABELS = ('d_q', 'd_i', 'd_fix', 'd_sep_q', 'd_sep_i', 'mal_descent', 'quality_descent')


def memory_guard():
    require(psutil.virtual_memory().available / 2**30 > 8., 'RAM guard')


def forbidden(*args, **kwargs):
    raise RuntimeError('mask/model/visualization path is outside this cached audit')


def load_sources():
    native_math = H.load_native_math()
    box = sys.modules[native_math.__package__ + '.box_ops']
    namespace = {'__name__': '_audit159_native', 'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'np': np, 'os': os,
                 'Dict': Dict, 'linear_sum_assignment': linear_sum_assignment,
                 'register': lambda: (lambda cls: cls), 'logger': logging.getLogger('audit159'),
                 'VIS_MATCH': False, 'point_sample': forbidden}
    for key in ('batch_dice_loss', 'batch_sigmoid_ce_loss', 'box_cxcywh_to_xyxy',
                'generalized_box_iou', 'box_iou', 'generalized_box_inner_iou'):
        namespace[key] = getattr(box, key)
    matcher_tree = ast.parse((NATIVE / 'matcher.py').read_text(encoding='utf-8-sig'))
    matcher_class = next(node for node in matcher_tree.body if isinstance(node, ast.ClassDef) and node.name == 'HungarianMatcher')
    exec(compile(ast.Module(body=[matcher_class], type_ignores=[]), str(NATIVE / 'matcher.py'), 'exec'), namespace)
    criterion_tree = ast.parse((NATIVE / 'deim_criterion.py').read_text(encoding='utf-8-sig'))
    criterion_class = next(node for node in criterion_tree.body if isinstance(node, ast.ClassDef) and node.name == 'DEIMCriterion')
    names = {'loss_labels_mal', '_get_src_permutation_idx', '_vfl_weighted_bce_loss', '_get_vfl_weight_warmup_factor'}
    methods = [node for node in criterion_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in methods} == names, 'native MAL methods missing')
    shell = copy.deepcopy(criterion_class)
    shell.name, shell.bases, shell.keywords, shell.body, shell.decorator_list = 'NativeMALMethods', [], [], methods, []
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    exec(compile(module, str(NATIVE / 'deim_criterion.py'), 'exec'), namespace)
    # Exact archived COCO Python class; no mask or compiled evaluator is called.
    package_name = '_audit159_project_coco'
    package = types.ModuleType(package_name)
    package.__path__ = [str(EVAL_SOURCE.parent)]
    sys.modules[package_name] = package
    mask = types.ModuleType(package_name + '.mask')
    for name in ('iou', 'encode', 'decode', 'area', 'toBbox', 'frPyObjects'):
        setattr(mask, name, forbidden)
    sys.modules[mask.__name__] = mask
    spec = importlib.util.spec_from_file_location(package_name + '.cocoeval', EVAL_SOURCE)
    require(spec is not None and spec.loader is not None, 'COCO source unavailable')
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    return namespace['HungarianMatcher'], namespace['NativeMALMethods'], box, evaluator.COCOeval


def read_configuration():
    recipe = yaml.safe_load(RECIPE.read_text(encoding='utf-8-sig'))
    cc = recipe['DEIMCriterion']
    mc = copy.deepcopy(cc['matcher'])
    require(mc.pop('type') == 'HungarianMatcher', 'matcher type')
    require(recipe['num_classes'] == 10 and recipe['use_focal_loss'] is True
            and recipe['remap_mscoco_category'] is False, 'class/postprocess config')
    require(recipe['PostProcessor']['num_top_queries'] == 300 and recipe['score_threshold'] == 0., 'native top300/threshold')
    require(recipe['evaluator']['max_dets'] == 100 and not recipe['evaluator']['using_tiny_metrice']
            and not recipe['evaluator']['using_vtiny_metrice'], 'COCO configuration changed')
    require(mc['weight_dict'] == {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}
            and mc['alpha'] == .25 and mc['gamma'] == 2. and not mc.get('change_matcher', False)
            and not mc.get('mask_point_sample_ratio'), 'native matcher configuration changed')
    require(cc['gamma'] == 1.5 and cc['alpha'] == .75 and cc.get('mal_alpha') is None
            and cc.get('no_weight_vfl_epoch', -1) == -1 and cc['weight_dict']['loss_mal'] == 1., 'native MAL configuration changed')
    return recipe, mc


def new_mal(cls, recipe):
    obj = cls()
    cfg = recipe['DEIMCriterion']
    obj.num_classes, obj.gamma, obj.alpha = 10, cfg['gamma'], cfg['alpha']
    obj.mal_alpha = cfg.get('mal_alpha')
    obj.no_weight_vfl_epoch = cfg.get('no_weight_vfl_epoch', -1)
    obj.no_weight_vfl_warmup_iter = cfg.get('no_weight_vfl_warmup_iter', 500)
    obj.epoch = 94
    return obj


def mal_targets_and_gradients(logits, boxes, targets, own, native_box, gamma):
    """Closed-form derivative of the actual stop-weight final ordinary MAL."""
    denominator = float(sum(len(target['labels']) for target in targets))
    require(denominator > 0., 'MAL denominator')
    target0 = torch.zeros_like(logits)
    positive = torch.zeros_like(logits, dtype=torch.bool)
    own_quality = []
    for b, (queries, gtids) in enumerate(own):
        overlaps = native_box.box_iou(native_box.box_cxcywh_to_xyxy(boxes[b, queries]),
                                     native_box.box_cxcywh_to_xyxy(targets[b]['boxes'][gtids]))[0].diag()
        classes = targets[b]['labels'][gtids]
        target0[b, queries, classes] = overlaps
        positive[b, queries, classes] = True
        own_quality.append(overlaps)
    probabilities = logits.sigmoid().detach()
    target = target0.pow(gamma)
    weight = torch.where(positive, torch.ones_like(logits), probabilities.pow(gamma)).detach()
    grad = weight * (probabilities - target) / denominator
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction='sum') / denominator
    # Descriptive dense max-IoU alternative, intentionally not called native MAL.
    quality = torch.zeros_like(logits)
    for b, target_dict in enumerate(targets):
        overlaps = native_box.box_iou(native_box.box_cxcywh_to_xyxy(boxes[b]),
                                     native_box.box_cxcywh_to_xyxy(target_dict['boxes']))[0]
        for c in range(10):
            mask = target_dict['labels'] == c
            if bool(mask.any()):
                quality[b, :, c] = overlaps[:, mask].max(dim=1).values
    quality_weight = torch.where(quality > 0., torch.ones_like(logits), probabilities.pow(gamma)).detach()
    quality_grad = quality_weight * (probabilities - quality.pow(gamma)) / denominator
    return grad, quality_grad, loss, own_quality, {'denominator': denominator, 'gamma': gamma,
            'mal_alpha': None, 'positive_slots': int(positive.sum()), 'zero_IoU_positive_slots': int((positive & (target0 == 0.)).sum())}


def pixel_xywh(boxes):
    # Same FP32 arithmetic order as PostProcessor -> CocoEvaluator.prepare.
    from torchvision.ops import box_convert
    values = torch.as_tensor(boxes).detach().cpu()
    require(values.dtype == torch.float32 and values.ndim == 2 and values.shape[1] == 4, 'FP32 box contract')
    xyxy = box_convert(values, in_fmt='cxcywh', out_fmt='xyxy') * 640.
    x1, y1, x2, y2 = xyxy.unbind(1)
    answer = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1).double().numpy()
    require(np.all(np.isfinite(answer)) and np.all(answer[:, 2:] > 0.), 'invalid pixel box')
    return answer


def xywh_iou(a, b):
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    wh = np.maximum(hi - lo, 0.)
    intersection = wh[..., 0] * wh[..., 1]
    union = np.prod(a[:, 2:], axis=1)[:, None] + np.prod(b[:, 2:], axis=1)[None, :] - intersection
    require(np.all(union > 0.), 'invalid IoU union')
    result = intersection / union
    require(np.all((result >= 0.) & (result <= 1. + 1e-12)), 'IoU range')
    return np.clip(result, 0., 1.)


def matching_count(edges):
    if edges.shape[0] == 0 or edges.shape[1] == 0 or not edges.any():
        return 0
    matched = maximum_bipartite_matching(csr_matrix(edges), perm_type='column')
    return int(np.count_nonzero(matched >= 0))


def capacity_counts(overlaps, gt_labels, slots, subset=None, verify=False):
    """Both capacities see identical (q,c) slots and no per-class100 cap."""
    gtids = np.arange(len(gt_labels)) if subset is None else np.flatnonzero(subset)
    matrix, labels = overlaps[:, gtids], gt_labels[gtids]
    slots = np.asarray(slots, dtype=np.int64)
    permit = np.zeros((len(overlaps), 10), dtype=bool)
    permit[slots // 10, slots % 10] = True
    query_class = permit[:, labels]
    instance_class = (slots % 10)[:, None] == labels[None, :]
    qcounts, icounts = [], []
    for threshold in THRESHOLDS:
        qedges = (matrix >= threshold) & query_class
        iedges = (matrix[slots // 10] >= threshold) & instance_class
        qn, inum = matching_count(qedges), matching_count(iedges)
        require(qn <= inum, 'candidate capacity below query capacity')
        if verify:
            require(qn == len(H.match_capacity_one(qedges)) and inum == len(H.match_capacity_one(iedges)), 'independent matching mismatch')
        qcounts.append(qn)
        icounts.append(inum)
    return {'q': np.array(qcounts, dtype=np.int64), 'i': np.array(icounts, dtype=np.int64)}


def native_selection(logits):
    flat = torch.as_tensor(logits).detach().cpu().reshape(-1)
    require(flat.dtype == torch.float32 and flat.numel() == 3000, 'FP32 native logits contract')
    probability = flat.sigmoid()
    _, selected = probability.topk(300)
    return selected.numpy().copy(), probability.numpy().copy()


def swap_action(record, incoming, outgoing):
    z = record['logits'].clone().reshape(-1)
    old = z[incoming].clone()
    z[incoming] = z[outgoing]
    z[outgoing] = old
    selected, probabilities = native_selection(z)
    expected = set(record['selected'].tolist()) - {outgoing} | {incoming}
    realized = set(selected.tolist()) == expected
    return selected, probabilities, realized


class PanelEvaluator:
    def __init__(self, evaluator_class, records):
        self.records = {r['image_index']: r for r in records}
        self.evaluator = evaluator_class(None, None, iouType='bbox', using_tiny_metrice=False,
                                         using_vtiny_metrice=False, max_dets=100)
        p = self.evaluator.params
        p.imgIds, p.catIds = sorted(self.records), list(range(10))
        p.areaRng, p.areaRngLbl = [[0., 1e10]], ['all']
        require(np.array_equal(p.iouThrs, THRESHOLDS) and p.maxDets == [1, 10, 100]
                and p.useCats == 1 and len(p.recThrs) == 101, 'project COCO parameters')
        for index, record in self.records.items():
            for c in range(10):
                gtids = np.flatnonzero(record['labels'] == c)
                self.evaluator._gts[index, c] = [
                    {'id': index * 10000 + int(g) + 1, 'ignore': 0, 'iscrowd': 0,
                     'area': float(np.prod(record['gt_pixel'][g, 2:]))} for g in gtids]
        self.baseline = {index: self.image_categories(record, record['selected'], record['probabilities'], range(10))
                         for index, record in self.records.items()}

    def image_categories(self, record, selected, probabilities, categories):
        index = record['image_index']
        result = {}
        for c in sorted(set(int(x) for x in categories)):
            slots = np.asarray(selected)[np.asarray(selected) % 10 == c]
            # Input order is the actual global native topk order; mergesort
            # preserves its ties exactly as the project Python evaluator does.
            order = np.argsort(-probabilities[slots], kind='mergesort')
            sorted_slots = slots[order][:100]
            gtids = np.flatnonzero(record['labels'] == c)
            self.evaluator._dts[index, c] = [
                {'id': index * 10000 + int(slot) + 1, 'score': float(probabilities[slot]),
                 'area': float(np.prod(record['pred_pixel'][slot // 10, 2:]))} for slot in slots]
            self.evaluator.ious[index, c] = record['overlaps'][sorted_slots // 10][:, gtids]
            result[c] = self.evaluator.evaluateImg(index, c, [0., 1e10], 100)
        return result

    def action_categories(self, record, selected, probabilities, realized, incoming, outgoing):
        # A native topk tie may reorder an otherwise unchanged class.  COCO's
        # stable score sort must inherit the actual new native order, so reuse
        # baseline only when both the class slot order and probabilities agree.
        categories = set()
        for c in range(10):
            before = record['selected'][record['selected'] % 10 == c]
            after = np.asarray(selected)[np.asarray(selected) % 10 == c]
            if not np.array_equal(before, after) or not np.array_equal(record['probabilities'][before], probabilities[after]):
                categories.add(c)
        result = dict(self.baseline[record['image_index']])
        result.update(self.image_categories(record, selected, probabilities, categories))
        return result

    @staticmethod
    def greedy_counts(categories):
        tp = np.zeros(10, dtype=np.int64)
        fp = np.zeros(10, dtype=np.int64)
        kept = 0
        for evaluation in categories.values():
            if evaluation is None:
                continue
            matches = evaluation['dtMatches'] > 0
            ignore = np.asarray(evaluation['dtIgnore'], dtype=bool)
            tp += ((matches) & (~ignore)).sum(axis=1)
            fp += ((~matches) & (~ignore)).sum(axis=1)
            kept += len(evaluation['dtIds'])
        return {'TP': tp, 'FP': fp, 'kept_after_perclass100': kept}

    def panel(self, replacements):
        source = {index: replacements.get(index, self.baseline[index]) for index in sorted(self.records)}
        self.evaluator.evalImgs = [source[index][c] for c in range(10) for index in sorted(self.records)]
        self.evaluator._paramsEval = copy.deepcopy(self.evaluator.params)
        with contextlib.redirect_stdout(io.StringIO()):
            self.evaluator.accumulate()
        precision = self.evaluator.eval['precision'][:, :, :, 0, -1].copy()
        recall = self.evaluator.eval['recall'][:, :, 0, -1].copy()
        valid = precision >= 0.
        require(valid.any(), 'no evaluated panel GT')
        mean = float(precision[valid].mean())
        threshold_ap = [float(precision[t][valid[t]].mean()) if valid[t].any() else None for t in range(10)]
        class_ap = [float(precision[:, :, c][valid[:, :, c]].mean()) if valid[:, :, c].any() else None for c in range(10)]
        counts = [self.greedy_counts(source[index]) for index in sorted(source)]
        report = {'mean_panel_PR': mean, 'panel_PR50': threshold_ap[0], 'panel_PR75': threshold_ap[5],
                  'per_threshold_panel_PR': threshold_ap, 'per_class_panel_PR': class_ap,
                  'TP_by_threshold': sum((item['TP'] for item in counts), np.zeros(10, dtype=np.int64)),
                  'FP_by_threshold': sum((item['FP'] for item in counts), np.zeros(10, dtype=np.int64)),
                  'kept_after_perclass100': sum(item['kept_after_perclass100'] for item in counts),
                  'classes_with_GT': int(sum(value is not None for value in class_ap))}
        return report, precision, recall


def choose(rows, label):
    candidates = [row for row in rows if row['realizes_named_exchange'] and row['labels'][label] > 0.]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row['logit_gap'], row['j_flat'], row['i_flat']))


def self_checks(matcher_class, mal_class, native_box, evaluator_class, recipe, matcher_config):
    checks = {}
    # Same query, two distinct classes, two identical GTs: query cap1, candidate cap2.
    overlaps = np.array([[1., 1.]])
    caps = capacity_counts(overlaps, np.array([0, 1]), np.array([0, 1]), verify=True)
    require(np.all(caps['q'] == 1) and np.all(caps['i'] == 2), 'capacity-definition synthetic example')
    checks['same_query_different_class_capacity'] = True
    # Rank-function exchange interaction: S={A}, A and B both see GT1.
    # Adding B alone gives0, deleting A loses1, but replacing A by B gives0.
    base = matching_count(np.array([[True]], dtype=bool))
    add = matching_count(np.array([[True], [True]], dtype=bool))
    remove = matching_count(np.zeros((0, 1), dtype=bool))
    exchange = matching_count(np.array([[True]], dtype=bool))
    require(exchange - base == 0 and (add - base) - (base - remove) == -1, 'exchange vs separated increments')
    checks['exchange_not_equal_separate_add_remove'] = True
    z = torch.arange(3000, dtype=torch.float32).reshape(300, 10) / 1000. - 3.
    selected, probability = native_selection(z)
    outside = np.setdiff1d(np.arange(3000), selected)
    record = {'logits': z, 'selected': selected}
    _, _, realized = swap_action(record, int(outside[-1]), int(selected[-1]))
    require(realized, 'native score swap realization')
    checks['native_logit_swap'] = True
    boxes = torch.tensor([[[.25, .25, .1, .1], [.75, .75, .1, .1]]], dtype=torch.float32)
    logits = torch.full((1, 2, 10), -4., dtype=torch.float32)
    logits[0, 0, 0], logits[0, 1, 1] = 4., 4.
    targets = [{'boxes': boxes[0].clone(), 'labels': torch.tensor([0, 1])}]
    matcher = matcher_class(**matcher_config, use_focal_loss=True)
    own = matcher({'pred_boxes': boxes, 'pred_logits': logits}, targets, epoch=94)['indices']
    require(own[0][0].tolist() == [0, 1] and own[0][1].tolist() == [0, 1], 'native synthetic matcher')
    grad, _, formula_loss, _, _ = mal_targets_and_gradients(logits, boxes, targets, own, native_box, 1.5)
    obj = new_mal(mal_class, recipe)
    native_loss = obj.loss_labels_mal({'pred_boxes': boxes, 'pred_logits': logits}, targets, own, 2.)['loss_mal']
    require(abs(float(native_loss - formula_loss)) <= 1e-6, 'native MAL synthetic formula')
    p = logits.sigmoid().detach().double()
    y = torch.zeros_like(p)
    y[0, 0, 0], y[0, 1, 1] = 1., 1.
    weight = p.pow(1.5)
    weight[0, 0, 0], weight[0, 1, 1] = 1., 1.
    direction = torch.zeros_like(p)
    direction[0, 0, 0], direction[0, 1, 0] = 1., -1.
    f = lambda zz: torch.nn.functional.binary_cross_entropy_with_logits(zz, y, weight=weight, reduction='sum') / 2.
    eps = 1e-5
    numerical = float((f(logits.double() + eps * direction) - f(logits.double() - eps * direction)) / (2 * eps))
    analytic = float(((logits.double().sigmoid() - y) * weight * direction).sum() / 2.)
    require(abs(numerical - analytic) < 1e-9 and torch.isfinite(grad).all(), 'stop-weight MAL derivative')
    checks['native_MAL_and_finite_difference'] = True
    # Direct source greedy: first DT chooses GT1, so second is stranded, although
    # maximum matching could assign first->GT2 and second->GT1.
    ev = evaluator_class(None, None, iouType='bbox', max_dets=100)
    ev._gts[0, 0] = [{'id': 1, 'ignore': 0, 'iscrowd': 0, 'area': 1.}, {'id': 2, 'ignore': 0, 'iscrowd': 0, 'area': 1.}]
    ev._dts[0, 0] = [{'id': 1, 'score': .9, 'area': 1.}, {'id': 2, 'score': .8, 'area': 1.}]
    ev.ious[0, 0] = np.array([[.9, .8], [.9, 0.]])
    result = ev.evaluateImg(0, 0, [0., 1e10], 100)
    require(int((result['dtMatches'][0] > 0).sum()) == 1 and matching_count(ev.ious[0, 0] >= .5) == 2, 'COCO greedy vs maximum')
    checks['COCO_greedy_is_not_maximum_matching'] = True
    ev._gts[0, 0] = []
    ev._dts[0, 0] = [{'id': i + 1, 'score': 1. - i / 1000., 'area': 1.} for i in range(101)]
    ev.ious[0, 0] = np.zeros((100, 0))
    require(len(ev.evaluateImg(0, 0, [0., 1e10], 100)['dtIds']) == 100, 'COCO per-class cap')
    checks['COCO_perclass100'] = True
    # A matched ignored GT and an unmatched out-of-area DT are both ignored.
    ev._gts[0, 0] = [{'id': 1, 'ignore': 1, 'iscrowd': 0, 'area': 1.}]
    ev._dts[0, 0] = [{'id': 1, 'score': .9, 'area': 1.}, {'id': 2, 'score': .8, 'area': 4.}]
    ev.ious[0, 0] = np.array([[1.], [0.]])
    ignored = ev.evaluateImg(0, 0, [0., 2.], 100)
    require(bool(np.asarray(ignored['gtIgnore']).all()) and bool(np.asarray(ignored['dtIgnore']).all()), 'COCO ignore/area handling')
    checks['COCO_ignore_and_current_area'] = True
    # One perfect detection/GT gives all101 precision samples equal1.
    ev.params.imgIds, ev.params.catIds = [0], [0]
    ev.params.areaRng, ev.params.areaRngLbl = [[0., 1e10]], ['all']
    ev._gts[0, 0] = [{'id': 1, 'ignore': 0, 'iscrowd': 0, 'area': 1.}]
    ev._dts[0, 0] = [{'id': 1, 'score': .9, 'area': 1.}]
    ev.ious[0, 0] = np.ones((1, 1))
    ev.evalImgs = [ev.evaluateImg(0, 0, [0., 1e10], 100)]
    ev._paramsEval = copy.deepcopy(ev.params)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.accumulate()
    require(np.max(np.abs(ev.eval['precision'] - 1.)) < 1e-12, 'COCO 101-point precision')
    checks['COCO_101point_PR'] = True
    example = [{'realizes_named_exchange': True, 'labels': {'x': 1.}, 'logit_gap': .2, 'j_flat': 4, 'i_flat': 8},
               {'realizes_named_exchange': True, 'labels': {'x': 1.}, 'logit_gap': .2, 'j_flat': 3, 'i_flat': 9}]
    require(choose(example, 'x')['j_flat'] == 3 and choose([], 'x') is None, 'choice tie/NA')
    checks['minimum_gap_tie_and_NA'] = True
    return checks


def prepare_cache(cache, matcher_class, mal_class, native_box, recipe, matcher_config):
    require(cache['checkpoint_sha256'] == CHECKPOINT_SHA and cache['weight_key'] == 'model_online', 'cache identity')
    raw = cache['records']
    require(isinstance(raw, list) and len(raw) == 4 and len({int(r['image_index']) for r in raw}) == 4, 'four-image panel')
    raw = sorted(raw, key=lambda r: int(r['image_index']))
    targets, records, logits_list, boxes_list = [], [], [], []
    for item in raw:
        z = torch.as_tensor(item['pred_logits']).detach().cpu().clone()
        b = torch.as_tensor(item['pred_boxes']).detach().cpu().clone()
        gt = torch.as_tensor(item['target']['boxes']).detach().cpu().clone()
        labels = torch.as_tensor(item['target']['labels']).detach().cpu().long().clone()
        require(z.shape == (300, 10) and b.shape == (300, 4) and z.dtype == b.dtype == gt.dtype == torch.float32,
                'native cache FP32 shapes')
        require(gt.ndim == 2 and gt.shape[1] == 4 and labels.shape == (len(gt),)
                and bool(((labels >= 0) & (labels < 10)).all()), 'GT schema')
        require(bool(torch.isfinite(z).all() and torch.isfinite(b).all() and torch.isfinite(gt).all())
                and bool((b[:, 2:] > 0).all() and (gt[:, 2:] > 0).all()), 'cache values')
        small = torch.as_tensor(item['original_sub4']).detach().cpu()
        require(small.dtype == torch.bool and small.shape == (len(gt),), 'sub4 mask')
        flags = {}
        for field in ('iscrowd', 'ignore'):
            flags[field] = field in item['target']
            if flags[field]:
                require(not bool(torch.as_tensor(item['target'][field]).bool().any()), 'nonzero crowd/ignore not supported by ordinary panel contract')
        selected, probabilities = native_selection(z)
        cached_selected = array(item['natural_top300_flat_indices'], 'cached native top300', shape=(300,), integer=True)
        require(np.array_equal(selected, cached_selected), 'recomputed native top300 order differs from cached natural_top300_flat_indices')
        require(bool((probabilities[selected] > 0.).all()), 'native score_threshold=0 would remove a selected slot')
        complement = torch.from_numpy(probabilities.copy())
        complement[torch.from_numpy(selected)] = -float('inf')
        _, outside = complement.topk(16)
        pred_pixel, gt_pixel = pixel_xywh(b), pixel_xywh(gt)
        overlaps = xywh_iou(pred_pixel, gt_pixel)
        normalized64 = H.iou(b.double().numpy()[:, None], gt.double().numpy()[None])
        record = {'image_index': int(item['image_index']), 'event': str(item['event']), 'input_sha256': item['input_sha256'],
                  'logits': z, 'boxes': b, 'gt_boxes': gt, 'labels': labels.numpy(), 'small': small.numpy().copy(),
                  'pred_pixel': pred_pixel, 'gt_pixel': gt_pixel, 'overlaps': overlaps, 'selected': selected,
                  'probabilities': probabilities, 'inside': selected[284:300].copy(), 'outside': outside.numpy().copy(),
                  'metadata_flags_present': flags, 'pixel_vs_normalized64_IoU_max_difference': float(np.max(np.abs(overlaps - normalized64)))}
        records.append(record)
        targets.append({'boxes': gt, 'labels': labels})
        logits_list.append(z)
        boxes_list.append(b)
    require(sum(len(r['labels']) for r in records) == 902 and sum(int(r['small'].sum()) for r in records) == 453, 'panel counts')
    logits, boxes = torch.stack(logits_list), torch.stack(boxes_list)
    matcher = matcher_class(**matcher_config, use_focal_loss=True)
    own = matcher({'pred_boxes': boxes, 'pred_logits': logits}, targets, epoch=94)['indices']
    grad, quality_grad, formula_loss, own_iou, meta = mal_targets_and_gradients(logits, boxes, targets, own, native_box, 1.5)
    obj = new_mal(mal_class, recipe)
    actual_loss = obj.loss_labels_mal({'pred_boxes': boxes, 'pred_logits': logits}, targets, own, 902.)['loss_mal']
    error = abs(float(actual_loss - formula_loss))
    require(error <= 1e-5 * max(1., abs(float(actual_loss))), 'native/formula MAL loss mismatch')
    meta.update({'native_loss_mal': float(actual_loss), 'formula_loss_mal': float(formula_loss), 'loss_abs_error': error,
                 'matcher_call_count': 1, 'matcher_batch_size': 4, 'MAL_method_calls_real_cache': 1})
    for position, record in enumerate(records):
        queries, gids = own[position]
        require(len(queries) == min(300, len(record['labels'])) and len(set(queries.tolist())) == len(queries)
                and len(set(gids.tolist())) == len(gids), 'own assignment contract')
        u = np.zeros(3000, dtype=np.float64)
        # Fixed quality uses the same pixel-IoU threshold geometry as H, with
        # native own assignment held fixed. Native MAL itself keeps native FP32 IoU.
        qnp, gnp = queries.numpy(), gids.numpy()
        pairs_iou = record['overlaps'][qnp, gnp]
        u[qnp * 10 + record['labels'][gnp]] = (pairs_iou[:, None] >= THRESHOLDS).mean(axis=1)
        record.update({'own_pairs': [[int(q), int(g)] for q, g in zip(qnp, gnp)], 'fixed_u': u,
                       'native_own_IoU': own_iou[position].numpy(), 'mal_grad': grad[position].numpy().astype(np.float64).reshape(-1),
                       'quality_grad': quality_grad[position].numpy().astype(np.float64).reshape(-1)})
    return records, meta


def audit_image(record, evaluator):
    memory_guard()
    slots, overlaps, gt_labels = record['selected'], record['overlaps'], record['labels']
    base = capacity_counts(overlaps, gt_labels, slots, verify=True)
    base_small = capacity_counts(overlaps, gt_labels, slots, subset=record['small'])
    additions, removals = {}, {}
    for incoming in record['outside'].tolist():
        additions[incoming] = capacity_counts(overlaps, gt_labels, np.append(slots, incoming))
    for outgoing in record['inside'].tolist():
        removals[outgoing] = capacity_counts(overlaps, gt_labels, slots[slots != outgoing])
    baseline_greedy = evaluator.greedy_counts(evaluator.baseline[record['image_index']])
    rows = []
    for j_rank, outgoing in enumerate(record['inside'].tolist(), 285):
        memory_guard()
        for i_rank, incoming in enumerate(record['outside'].tolist(), 301):
            intended_slots = np.where(slots == outgoing, incoming, slots)
            full = capacity_counts(overlaps, gt_labels, intended_slots)
            selected, probability, realized = swap_action(record, incoming, outgoing)
            actual = full if realized else capacity_counts(overlaps, gt_labels, selected)
            categories = evaluator.action_categories(record, selected, probability, realized, incoming, outgoing)
            greedy = evaluator.greedy_counts(categories)
            delta = {kind: full[kind] - base[kind] for kind in ('q', 'i')}
            sep = {kind: (additions[incoming][kind] - base[kind]) - (base[kind] - removals[outgoing][kind]) for kind in ('q', 'i')}
            d_mal = float(record['mal_grad'][incoming] - record['mal_grad'][outgoing])
            d_quality = float(record['quality_grad'][incoming] - record['quality_grad'][outgoing])
            labels = {'d_q': float(delta['q'].mean()), 'd_i': float(delta['i'].mean()),
                      'd_fix': float(record['fixed_u'][incoming] - record['fixed_u'][outgoing]),
                      'd_sep_q': float(sep['q'].mean()), 'd_sep_i': float(sep['i'].mean()),
                      'mal_descent': -d_mal, 'quality_descent': -d_quality}
            rows.append({'image_index': record['image_index'], 'event': record['event'], 'i_flat': incoming, 'j_flat': outgoing,
                         'i_query_class': [incoming // 10, incoming % 10], 'j_query_class': [outgoing // 10, outgoing % 10],
                         'i_rank': i_rank, 'j_rank': j_rank,
                         'z_i': float(record['logits'].reshape(-1)[incoming]), 'z_j': float(record['logits'].reshape(-1)[outgoing]),
                         'logit_gap': float(record['logits'].reshape(-1)[outgoing].double() - record['logits'].reshape(-1)[incoming].double()),
                         'same_query': incoming // 10 == outgoing // 10, 'same_class': incoming % 10 == outgoing % 10,
                         'realizes_named_exchange': realized, 'actual_added_slots': sorted(set(selected.tolist()) - set(slots.tolist())),
                         'actual_removed_slots': sorted(set(slots.tolist()) - set(selected.tolist())),
                         'labels': labels, 'd_query_by_threshold': delta['q'], 'd_instance_by_threshold': delta['i'],
                         'd_sep_query_by_threshold': sep['q'], 'd_sep_instance_by_threshold': sep['i'],
                         'add_query_by_threshold': additions[incoming]['q'] - base['q'],
                         'remove_query_by_threshold': base['q'] - removals[outgoing]['q'],
                         'add_instance_by_threshold': additions[incoming]['i'] - base['i'],
                         'remove_instance_by_threshold': base['i'] - removals[outgoing]['i'],
                         'actual_H_counts': actual, 'mal_directional_derivative': d_mal,
                         'quality_directional_derivative': d_quality,
                         'actual_greedy_TP_delta_by_threshold': greedy['TP'] - baseline_greedy['TP'],
                         'actual_greedy_FP_delta_by_threshold': greedy['FP'] - baseline_greedy['FP'],
                         'actual_kept_after_perclass100': greedy['kept_after_perclass100'],
                         'actual_dropped_by_perclass100': 300 - greedy['kept_after_perclass100']})
    require(len(rows) == 256, 'fixed exchange count')
    descriptor = {'image_index': record['image_index'], 'event': record['event'], 'input_sha256': record['input_sha256'],
                  'GT': len(gt_labels), 'original_sub4': int(record['small'].sum()),
                  'baseline_H_counts': base, 'baseline_sub4_capacity_diagnostic': base_small, 'baseline_greedy': baseline_greedy,
                  'inside_flat_ids': record['inside'], 'outside_flat_ids': record['outside'], 'native_top300': slots,
                  'native_top300_matches_cached_order': True,
                  'own_Hungarian_pairs': record['own_pairs'], 'fixed_u_nonzero': [[int(i), float(record['fixed_u'][i])] for i in np.flatnonzero(record['fixed_u'])],
                  'top300_adjacent_probability_ties': int((np.diff(record['probabilities'][slots]) == 0.).sum()),
                  'inside_outside_equal_probability_pairs': int((record['probabilities'][record['inside']][:, None] == record['probabilities'][record['outside']][None]).sum()),
                  'rank300_rank301_probability_tie': bool(record['probabilities'][slots[-1]] == record['probabilities'][record['outside'][0]]),
                  'metadata_flags_present': record['metadata_flags_present'],
                  'pixel_vs_normalized64_IoU_max_difference': record['pixel_vs_normalized64_IoU_max_difference'],
                  'IoU_near_threshold_counts': [int((np.abs(overlaps - t) <= 1e-10).sum()) for t in THRESHOLDS]}
    return rows, descriptor


def pair_key(row):
    return (row['image_index'], row['j_flat'], row['i_flat'])


def label_summary(rows):
    realized = [row for row in rows if row['realizes_named_exchange']]
    positives = {label: {pair_key(row) for row in realized if row['labels'][label] > 0.} for label in LABELS}
    output = {'all_rows': len(rows), 'realized_rows': len(realized), 'unrealized_rows': len(rows) - len(realized),
              'same_query_rows': sum(row['same_query'] for row in rows), 'labels': {}, 'disagreements': {}}
    for label in LABELS:
        output['labels'][label] = {'positive': len(positives[label]),
            'negative': sum(row['labels'][label] < 0. for row in realized),
            'zero': sum(row['labels'][label] == 0. for row in realized),
            'positive_exchange_ids': sorted(positives[label])}
    for candidate, simple in (('d_q', 'd_fix'), ('d_q', 'd_sep_q'), ('d_i', 'd_fix'), ('d_i', 'd_sep_i'),
                              ('d_i', 'd_q'), ('d_i', 'mal_descent'), ('d_i', 'quality_descent')):
        a, b = positives[candidate], positives[simple]
        output['disagreements'][candidate + '_vs_' + simple] = {
            'positive_sets_identical': a == b, 'candidate_only': sorted(a - b), 'simple_only': sorted(b - a),
            'both_positive': len(a & b), 'neither_positive': len(realized) - len(a | b)}
    for candidate in ('d_q', 'd_i'):
        positive_rows = [row for row in realized if row['labels'][candidate] > 0.]
        output['labels'][candidate]['positive_but_mean_greedy_TP_negative'] = sum(
            np.mean(row['actual_greedy_TP_delta_by_threshold']) < 0. for row in positive_rows)
        output['labels'][candidate]['positive_but_greedy_TP75_negative'] = sum(
            row['actual_greedy_TP_delta_by_threshold'][5] < 0 for row in positive_rows)
    return output, positives


def selected_panels(records, all_rows, evaluator):
    baseline, precision, recall = evaluator.panel({})
    arrays = {'identity_precision': precision, 'identity_recall': recall}
    report = {'identity': baseline, 'rules': {}}
    choices = {}
    row_by_image = {r['image_index']: [row for row in all_rows if row['image_index'] == r['image_index']] for r in records}
    for label in LABELS:
        selections, replacements, selected_ids = [], {}, []
        choices[label] = {}
        for record in records:
            index = record['image_index']
            selected_row = choose(row_by_image[index], label)
            choices[label][index] = selected_row
            if selected_row is None:
                selections.append({'image_index': index, 'status': 'NA_no_positive_realized_exchange', 'selected_pair': None})
                continue
            incoming, outgoing = selected_row['i_flat'], selected_row['j_flat']
            selected, probability, realized = swap_action(record, incoming, outgoing)
            require(realized, 'selected action no longer realizes named exchange')
            categories = evaluator.action_categories(record, selected, probability, True, incoming, outgoing)
            replacements[index] = categories
            one_panel, p, rec = evaluator.panel({index: categories})
            key = label + '_image' + str(index)
            arrays[key + '_precision'], arrays[key + '_recall'] = p, rec
            tiny = capacity_counts(record['overlaps'], record['labels'], selected, subset=record['small'])
            selected_ids.append(pair_key(selected_row))
            selections.append({'image_index': index, 'status': 'selected', 'selected_pair': selected_row,
                               'single_exchange_panel': one_panel, 'single_exchange_PR_delta': one_panel['mean_panel_PR'] - baseline['mean_panel_PR'],
                               'sub4_capacity_diagnostic_after': tiny})
        if replacements:
            combined, p, rec = evaluator.panel(replacements)
            arrays[label + '_joint_precision'], arrays[label + '_joint_recall'] = p, rec
            joint = {'status': 'executed_deterministic_score_swaps', 'active_images': sorted(replacements),
                     'NA_images_preserved_identity': [r['image_index'] for r in records if r['image_index'] not in replacements],
                     'panel': combined, 'PR_delta_identity': combined['mean_panel_PR'] - baseline['mean_panel_PR'],
                     'PR75_delta_identity': combined['panel_PR75'] - baseline['panel_PR75']}
        else:
            joint = {'status': 'NA_no_positive_in_any_image', 'active_images': [], 'panel': None,
                     'PR_delta_identity': None, 'PR75_delta_identity': None}
        report['rules'][label] = {'selections': selections, 'selected_exchange_ids': selected_ids, 'joint': joint}
    return report, arrays, choices


def gate_report(summary, positives, selected, choices):
    reports = {}
    for label, sep in (('d_q', 'd_sep_q'), ('d_i', 'd_sep_i')):
        controls = ('d_fix', sep, 'mal_descent', 'quality_descent')
        label_extra = positives[label] != positives['d_fix'] and positives[label] != positives[sep]
        witnesses = []
        for index, row in choices[label].items():
            if row is None:
                continue
            control_pairs = {pair_key(choices[control][index]) for control in controls if choices[control][index] is not None}
            if pair_key(row) not in control_pairs:
                witnesses.append(pair_key(row))
        joint = selected['rules'][label]['joint']
        pr_positive = None if joint['PR_delta_identity'] is None else joint['PR_delta_identity'] > PR_TOL
        differences = {}
        for control in controls:
            other = selected['rules'][control]['joint']
            differences[control] = None if joint['panel'] is None or other['panel'] is None else {
                'mean_panel_PR_difference': joint['panel']['mean_panel_PR'] - other['panel']['mean_panel_PR'],
                'panel_PR75_difference': joint['panel']['panel_PR75'] - other['panel']['panel_PR75']}
        reports[label] = {'positive_label_extra': bool(label_extra), 'selection_extra': bool(witnesses),
                          'selection_extra_exchange_ids': witnesses, 'joint_PR_strictly_positive': pr_positive,
                          'comparisons_with_simple': differences,
                          'gate': bool(label_extra and witnesses and pr_positive is True)}
    return {'primary_candidate': 'd_i', 'label_premise_gate': reports['d_i']['gate'], 'families': reports,
            'decision': 'eligible_to_discuss_preregistered_shared_head_test_only' if reports['d_i']['gate'] else 'do_not_enter_shared_head_for_this_exact_label_protocol',
            'Hq_only_cannot_authorize_next_stage': True, 'training_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=pathlib.Path, default=ROOT / '127_C0_attention_template_registration/01_cache.pt')
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / '160_C0_exchange_label_audit')
    args = parser.parse_args()
    args.cache, args.out = args.cache.resolve(), args.out.resolve()
    require(args.cache == (ROOT / '127_C0_attention_template_registration/01_cache.pt').resolve(), 'cache path is fixed to original 127')
    require(args.out.parent == ROOT and args.out.name.startswith('160_') and not args.out.exists(), 'output must be a new 160_* directory directly under this protocol ROOT')
    args.out.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    files = {'script': pathlib.Path(__file__), 'protocol': PROTOCOL, 'helper': HELPER, 'recipe': RECIPE,
             'matcher': NATIVE / 'matcher.py', 'criterion': NATIVE / 'deim_criterion.py', 'box_ops': NATIVE / 'box_ops.py',
             'dfine_utils': NATIVE / 'dfine_utils.py', 'postprocessor': NATIVE / 'postprocessor.py',
             'evaluator': EVAL_SOURCE, 'wrapper': WRAPPER, 'cache': args.cache}
    before, resources, checks = {}, None, None
    result = {'state': 'error', 'primary': {'label_premise_gate': False}}
    cache_loaded, stage = False, 'resources'
    try:
        resources = H.resource_limits()
        require(resources['priority'] == 'below_normal' and torch.version.cuda is None, 'CPU/priority contract')
        before = {key: sha(path) for key, path in files.items() if key != 'cache'}
        require(all(before[key] == expected for key, expected in EXPECTED.items()), 'frozen source changed')
        recipe, matcher_config = read_configuration()
        matcher_class, mal_class, native_box, evaluator_class = load_sources()
        stage = 'synthetic_checks_before_cache'
        checks = self_checks(matcher_class, mal_class, native_box, evaluator_class, recipe, matcher_config)
        stage = 'cache_identity_and_load'
        before['cache'] = sha(args.cache)
        require(before['cache'] == CACHE_SHA, 'unexpected cache bytes')
        cache = torch.load(args.cache, map_location='cpu', weights_only=False)
        cache_loaded = True
        records, mal_meta = prepare_cache(cache, matcher_class, mal_class, native_box, recipe, matcher_config)
        del cache
        evaluator = PanelEvaluator(evaluator_class, records)
        all_rows, image_reports = [], []
        stage = 'all_1024_fixed_exchanges'
        for record in records:
            rows, descriptor = audit_image(record, evaluator)
            all_rows.extend(rows)
            image_reports.append(descriptor)
            print('cached exchanges complete for image ' + str(record['image_index']), flush=True)
        require(len(all_rows) == 1024, 'fixed four-image exchange count')
        summary, positives = label_summary(all_rows)
        stage = 'preselected_panel_PR'
        selected, arrays, choices = selected_panels(records, all_rows, evaluator)
        primary = gate_report(summary, positives, selected, choices)
        stage = 'save_outputs'
        save(args.out / '01_all_1024_pairs.json', {'thresholds': THRESHOLDS, 'rows': all_rows})
        save(args.out / '02_selected_panels.json', selected)
        np.savez_compressed(args.out / '03_selected_precision.npz', **arrays)
        artifacts = {path.name: sha(path) for path in (args.out / '01_all_1024_pairs.json', args.out / '02_selected_panels.json', args.out / '03_selected_precision.npz')}
        result = {'state': 'complete', 'primary': primary, 'GT': 902, 'original_sub4': 453, 'thresholds': THRESHOLDS,
                  'images': image_reports, 'label_summary': summary, 'native_MAL': mal_meta,
                  'matcher_config': {**matcher_config, 'use_focal_loss': True, 'epoch': 94, 'change_matcher': False},
                  'evaluator_config': {'maxDets': [1, 10, 100], 'useCats': 1, 'recall_points': 101, 'area': 'all',
                                       'ordinary_augmented_GT_only': True, 'historical_CPP_not_executed': True},
                  'artifact_sha256': artifacts, 'identity_panel': selected['identity'],
                  'scope': 'One C0 four-image development label audit. No validation AP, training, learnability or novelty proof.'}
        stage = 'complete'
    except Exception as exc:
        result = {'state': 'error', 'primary': {'label_premise_gate': False}, 'error': str(exc),
                  'traceback': traceback.format_exc(), 'scope': 'Failclosed implementation/configuration outcome, not method evidence.'}
    finally:
        after = {key: sha(files[key]) for key in before if files[key].exists()}
        unchanged = bool(before and before == after)
        if result['state'] == 'complete':
            unchanged = unchanged and set(before) == set(files)
        if not unchanged:
            result['state'] = 'error'
            result.setdefault('primary', {})['label_premise_gate'] = False
            result['hash_error'] = 'Missing or changed source'
        result.update({'source_paths': {key: str(path.resolve()) for key, path in files.items()},
                       'sha256_before': before, 'sha256_after': after, 'sources_unchanged': unchanged,
                       'resources': resources, 'scipy_version': scipy.__version__, 'stage': stage,
                       'cache_content_loaded': cache_loaded, 'synthetic_self_checks': checks,
                       'model_forwards': 0, 'model_backwards': 0, 'optimizer_steps': 0, 'new_training': False,
                       'seconds': time.monotonic() - started})
        save(args.out / '00_results.json', result)
        print(H.json.dumps(H.jsonable({'state': result['state'], 'primary': result['primary'],
                                     'output': str(args.out / '00_results.json'), 'seconds': result['seconds']}),
                           ensure_ascii=False, allow_nan=False))
    if result['state'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
