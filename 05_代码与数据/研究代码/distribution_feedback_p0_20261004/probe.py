"""Read-only CPU P0: native FDR geometry and frozen-feature diagnostics.

No optimizer, no modified detector, no candidate AP, and no GT-driven sampling.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parent / 'local_candidates_20261004_v2'
sys.path.insert(0, str(SNAPSHOT))
import bootstrap
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, box_convert
from factory import build_model, verify_snapshot
from dataset import OriginalCocoImages
from geometry import windows, make_views, targets_for_window
from engine.deim.dfine_utils import weighting_function, distance2bbox

WORKSPACE = HERE.parents[2]
CHECKPOINT = WORKSPACE / '英文小论文/data/04_冻结权重诊断_20261001/c_seed1_runtime_v1/readonly_20261004_1828/epoch_053.pth'
DATA = Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
EXPECTED_CHECKPOINT = 'cf8bcea8d38e9ee826929f1b6d161a9ceff0ac4851e3a4318e1f197b2312aa97'
EXPECTED_ANNOTATION = '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def source_hashes():
    return {str(p.relative_to(SNAPSHOT)): sha(p) for p in sorted(SNAPSHOT.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.yml'}}


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def two_bin(probability, project):
    mean = (probability * project).sum(-1)
    j = torch.searchsorted(project.contiguous(), mean.contiguous(), right=True) - 1
    j = j.clamp(0, len(project) - 2)
    a = ((mean - project[j]) / (project[j + 1] - project[j])).clamp(0, 1)
    result = torch.zeros_like(probability)
    result.scatter_(-1, j[..., None], (1 - a)[..., None])
    result.scatter_add_(-1, (j + 1)[..., None], a[..., None])
    assert torch.allclose((result * project).sum(-1), mean, atol=2e-6, rtol=0)
    return result


def geometry(prob, initial, project, reg_scale):
    """Q,4,33 edge positions in original crop pixels; no GT enters."""
    first = distance2bbox(initial, (prob * project).sum(-1), reg_scale)
    centre = initial[:, [0, 1, 0, 1]]
    size = initial[:, [2, 3, 2, 3]]
    sign = initial.new_tensor([-1, -1, 1, 1])
    edge = (centre[..., None] + sign[None, :, None] *
            (.5 * reg_scale + project) * size[..., None] / reg_scale) * 640
    points = first[:, None, None, :2].expand(-1, 4, len(project), -1).clone() * 640
    points[:, [0, 2], :, 0] = edge[:, [0, 2]]
    points[:, [1, 3], :, 1] = edge[:, [1, 3]]
    mean = (prob[..., None] * points).sum(-2)
    first_xyxy = box_convert(first, 'cxcywh', 'xyxy') * 640
    mean_edge = (prob * edge).sum(-1)
    assert torch.allclose(mean_edge, first_xyxy, atol=3e-3, rtol=0)
    return first, edge, points, mean


def quantile_points(edge, points, probability):
    order = edge.argsort(-1)
    sorted_p = probability.gather(-1, order)
    cdf = sorted_p.cumsum(-1).contiguous()
    levels = edge.new_tensor([.1, .5, .9]).expand(*edge.shape[:-1], 3).contiguous()
    indices = torch.searchsorted(cdf, levels).clamp_max(edge.shape[-1] - 1)
    native_indices = order.gather(-1, indices)
    return points.gather(-2, native_indices[..., None].expand(-1, -1, -1, 2))


def feature_read(features, pixel_points, origin, span, valid_upper=None):
    h, w = features.shape[-2:]
    norm = (pixel_points + origin) / span
    valid = ((norm >= 0) & (norm <= 1)).all(-1)
    if valid_upper is not None:
        valid &= ((pixel_points >= 0) & (pixel_points <= valid_upper)).all(-1)
    shape = norm.shape[:-1]
    grid = norm.reshape(1, -1, 1, 2) * 2 - 1
    values = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros',
                           align_corners=False)[0, :, :, 0].T.reshape(*shape, features.shape[1])
    cells = torch.floor(norm * norm.new_tensor([w, h]) - .5)
    return values, valid, cells


def feature_metrics(feature, points, mean, quantiles, quantiles_e, p, e, origin, span, valid_upper=None):
    values, valid, _ = feature_read(feature, points, origin, span, valid_upper)
    centre, centre_valid, _ = feature_read(feature, mean[..., None, :], origin, span, valid_upper)
    qp, qvalid, cells = feature_read(feature, quantiles, origin, span, valid_upper)
    qe, evalid, _ = feature_read(feature, quantiles_e, origin, span, valid_upper)
    vp, ve = p * valid, e * valid
    mass_p, mass_e = vp.sum(-1), ve.sum(-1)
    vp = vp / mass_p.clamp_min(1e-12)[..., None]
    ve = ve / mass_e.clamp_min(1e-12)[..., None]
    expected_p = (vp[..., None] * values).sum(-2)
    expected_e = (ve[..., None] * values).sum(-2)
    centre = centre.squeeze(-2)
    scale = centre.norm(dim=-1).clamp_min(1e-8)
    def rel(a, b):
        result = (a - b).norm(dim=-1) / scale
        return torch.where((mass_p >= .95) & (mass_e >= .95) & centre_valid.squeeze(-1),
                           result, torch.full_like(result, float('nan')))
    rms = ((vp[..., None] * (values - expected_p[..., None, :]).square()).sum((-2, -1))).sqrt() / scale
    result = {
        'p_mean_relative_l2': rel(expected_p, centre),
        'p_e_relative_l2': rel(expected_p, expected_e),
        'e_mean_relative_l2': rel(expected_e, centre),
        'p_feature_rms_relative': torch.where(mass_p >= .95, rms, torch.full_like(rms, float('nan'))),
        'q3_e3_relative_l2': rel(qp.mean(-2), qe.mean(-2)),
        'p_valid_mass': mass_p,
        'e_valid_mass': mass_e,
        'q3_valid_all': qvalid.all(-1),
        'q3_same_interpolation_cell': (cells == cells[..., :1, :]).all(-1).all(-1) & qvalid.all(-1),
        'q3_feature_spread_relative': torch.stack([(qp[..., i, :] - qp[..., j, :]).norm(dim=-1)
             for i, j in [(0, 1), (0, 2), (1, 2)]]).amax(0) / scale,
    }
    result['q3_e3_relative_l2'] = torch.where(qvalid.all(-1) & evalid.all(-1),
        result['q3_e3_relative_l2'], torch.full_like(mass_p, float('nan')))
    result['q3_feature_spread_relative'] = torch.where(qvalid.all(-1), result['q3_feature_spread_relative'],
                                                      torch.full_like(mass_p, float('nan')))
    return result


def matching(first, target, local_target):
    q = len(first)
    output = {'match_iou': torch.full((q,), -1.), 'match_annotation_id': torch.full((q,), -1, dtype=torch.long),
              'original_gt_area': torch.full((q,), float('nan')), 'truncated': torch.zeros(q, dtype=torch.bool),
              'gt_edges_crop_px': torch.full((q, 4), float('nan'))}
    boxes = local_target['boxes']
    if not len(boxes):
        return output
    first_xy = box_convert(first, 'cxcywh', 'xyxy')
    gt_xy = box_convert(boxes, 'cxcywh', 'xyxy')
    iou = box_iou(first_xy, gt_xy)
    iou = torch.nan_to_num(iou, nan=0.).clamp(0, 1)
    rows, cols = linear_sum_assignment(-iou.numpy())
    rows, cols = torch.tensor(rows), torch.tensor(cols)
    output['match_iou'][rows] = iou[rows, cols]
    output['match_annotation_id'][rows] = local_target['annotation_ids'][cols]
    output['gt_edges_crop_px'][rows] = gt_xy[cols] * 640
    by_id = {int(k): i for i, k in enumerate(target['annotation_ids'])}
    for row, col in zip(rows.tolist(), cols.tolist()):
        original = by_id[int(local_target['annotation_ids'][col])]
        output['original_gt_area'][row] = target['area'][original]
        original_size = target['boxes'][original, 2:] - target['boxes'][original, :2]
        crop_size = boxes[col, 2:] * 640
        output['truncated'][row] = bool((original_size - crop_size).abs().max() > 1e-3)
    return output


class Capture:
    def __init__(self, model):
        self.model = model
        self.reset()
        self.handles = [model.decoder.integral.register_forward_pre_hook(self.integral),
                        model.decoder.decoder.register_forward_hook(self.decoder),
                        model.encoder.register_forward_hook(self.encoder),
                        model.decoder.decoder.layers[1].register_forward_pre_hook(self.second_layer)]

    def reset(self):
        self.corners = []
        self.layer_outputs = None
        self.encoded = None
        self.actual_second_reference = None

    def integral(self, module, inputs):
        self.corners.append(inputs[0].detach().clone())

    def decoder(self, module, inputs, outputs):
        self.layer_outputs = outputs

    def encoder(self, module, inputs, outputs):
        self.encoded = outputs

    def second_layer(self, module, inputs):
        self.actual_second_reference = inputs[1].detach().clone()

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def run(args):
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261004)
    np.random.seed(20261004)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'traces').mkdir()
    assert sha(CHECKPOINT) == EXPECTED_CHECKPOINT
    annotations = DATA / 'annotations/val.json'
    assert sha(annotations) == EXPECTED_ANNOTATION
    verify_snapshot()
    before_sources = source_hashes()
    data = OriginalCocoImages(DATA / 'images', annotations)
    chosen = sorted(range(len(data)), key=lambda i: hashlib.sha256(
        f'distribution-p0-v1:{data.images[i]["id"]}'.encode()).hexdigest())[:args.images]
    chosen.sort(key=lambda i: data.images[i]['id'])
    images = [{**data.images[i], 'index': i, 'sha256': sha(DATA / 'images' / data.images[i]['file_name']),
               'windows': windows(data.images[i]['width'], data.images[i]['height'], stride=512)} for i in chosen]
    cfg = {'kind': 'dual_view', 'arm': 'independent', 'fusion_width': 64}
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert payload['epoch'] == 53 and payload['bindings']['config']['arm'] == 'independent'
    model = build_model(cfg).cpu().eval()
    model.load_state_dict(payload['ema']['module'], strict=True)
    assert all(torch.equal(v, payload['ema']['module'][k]) for k, v in model.state_dict().items())
    initial_model_hash = model_hash(model)
    del payload
    assert all(not m.training for m in model.modules())
    project = weighting_function(model.decoder.reg_max, model.decoder.up, model.decoder.reg_scale).detach()
    reg_scale = model.decoder.reg_scale.detach()
    write_json(output / 'manifest.json', {
        'status': 'fixed_before_model_diagnostics', 'selection': '32 lowest sha256(distribution-p0-v1:image_id), then image_id order; all fixed windows',
        'images_requested': args.images, 'images': images, 'GT_selects_images_windows_queries_or_points': False,
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'checkpoint_completed_rounds': 54, 'weights': 'EMA module', 'annotations': str(annotations),
        'annotation_sha256': EXPECTED_ANNOTATION, 'source_hashes': before_sources, 'probe_sha256': sha(__file__),
        'model_state_hash': initial_model_hash, 'model_loaded_exactly': True,
        'torch': torch.__version__, 'device': 'cpu', 'threads': 4, 'dtype': 'float32',
        'quantiles': [.1, .5, .9], 'edge_tangential_position': 'mean FDR box center',
        'fixed3_offsets': [-.25, 0, .25], 'fixed3_scale': 'initial reference width/height',
        'features': ['local encoded P3 stride8', 'global encoded P4 stride16',
                     'local encoded P4 stride16 level control', 'global encoded P3 stride8 level control'],
        'matching': 'offline class-agnostic one-to-one maximum-IoU Hungarian; strata IoU>=0, >=.1, >=.5',
        'comparison': 'full33 probability-weighted bilinear probe; mean; same-mean two-bin; fixed3 and quantile3 oracle edge distances',
        'scope_limits': ['Not a trained attention module or candidate AP.', 'Class-agnostic diagnostic matching is not COCO TP assignment.',
                         'GT nearest-candidate distance is an oracle opportunity metric, not a deployable selector.',
                         'No frozen-feature diversity measurement establishes global context effectiveness.']})
    progress = output / 'progress.jsonl'
    preflight = {}
    crop_counter = 0
    capture = Capture(model)
    for image_row in images:
        image_started = time.perf_counter()
        image, target = data[image_row['index']]
        h, w = image.shape[-2:]
        first_window = image_row['windows'][0]
        full, _, _ = make_views(image, first_window)
        # Exact native independent eval feature route; cache the global encoded P4 once.
        global_backbone, global_prediction = model.cache_global(full[None])
        global_encoded = [v.detach().clone() for v in capture.encoded]
        for window_id, window in enumerate(image_row['windows']):
            _, local, crop_norm = make_views(image, window)
            capture.reset()
            prediction = model.predict_local(local[None], crop_norm[None], global_backbone)
            assert len(capture.corners) == 3
            logits = capture.corners[0][0].reshape(300, 4, 33)
            p = logits.softmax(-1)
            initial = capture.layer_outputs[4][0].detach()
            first, edge, points, mean = geometry(p, initial, project, reg_scale)
            assert torch.allclose(first, capture.actual_second_reference[0, :, 0], atol=2e-6, rtol=0)
            last_probability = capture.corners[-1][0].reshape(300, 4, 33).softmax(-1)
            reconstructed_last = distance2bbox(initial, (last_probability * project).sum(-1), reg_scale)
            assert torch.allclose(reconstructed_last, prediction['pred_boxes'][0], atol=2e-6, rtol=0)
            e = two_bin(p, project)
            quantiles = quantile_points(edge, points, p)
            quantiles_e = quantile_points(edge, points, e)
            normal_axis = torch.tensor([0, 1, 0, 1])
            def normal_values(x):
                return x.gather(-1, normal_axis[None, :, None, None].expand(len(p), 4, x.shape[-2], 1)).squeeze(-1)
            qedge = normal_values(quantiles)
            eedge = normal_values(quantiles_e)
            mean_edge = (p * edge).sum(-1)
            fixed_edge = mean_edge[..., None] + initial[:, [2, 3, 2, 3], None] * 640 * p.new_tensor([-.25, 0, .25])
            arrays = {'p': p, 'p_two_bin': e, 'initial_boxes': initial, 'first_boxes': first,
                      'final_boxes': prediction['pred_boxes'][0], 'final_logits': prediction['pred_logits'][0],
                      'edge_positions_px': edge, 'q3_edge_px': qedge, 'e3_edge_px': eedge,
                      'mean_edge_px': mean_edge, 'entropy_bits': -(p * p.clamp_min(1e-30).log2()).sum(-1),
                      'variance_px2': (p * (edge - mean_edge[..., None]).square()).sum(-1),
                      'q3_span_px': qedge[..., 2] - qedge[..., 0],
                      'e3_span_px': eedge[..., 2] - eedge[..., 0],
                      'q3_all_identical': (qedge - qedge[..., :1]).abs().amax(-1) < 1e-5}
            arrays['two_bin_variance_px2'] = (e * (edge - mean_edge[..., None]).square()).sum(-1)
            arrays['excess_variance_px2'] = arrays['variance_px2'] - arrays['two_bin_variance_px2']
            arrays['initial_edge_size_px'] = initial[:, [2, 3, 2, 3]] * 640
            local_target = targets_for_window(target, window, (h, w))
            arrays.update(matching(first, target, local_target))
            gt = arrays['gt_edges_crop_px']
            arrays['mean_edge_error_px'] = (mean_edge - gt).abs()
            arrays['q3_oracle_edge_error_px'] = (qedge - gt[..., None]).abs().amin(-1)
            arrays['e3_oracle_edge_error_px'] = (eedge - gt[..., None]).abs().amin(-1)
            arrays['fixed3_oracle_edge_error_px'] = (fixed_edge - gt[..., None]).abs().amin(-1)
            arrays['q3_oracle_gain_over_e3_px'] = arrays['e3_oracle_edge_error_px'] - arrays['q3_oracle_edge_error_px']
            arrays['q3_oracle_gain_over_fixed3_px'] = arrays['fixed3_oracle_edge_error_px'] - arrays['q3_oracle_edge_error_px']
            local_encoded = capture.encoded
            local_stats = feature_metrics(local_encoded[0], points, mean, quantiles, quantiles_e, p, e,
                                          p.new_tensor([0., 0.]), p.new_tensor([640., 640.]),
                                          p.new_tensor([min(640, w-window[0]), min(640, h-window[1])]))
            global_stats = feature_metrics(global_encoded[1], points, mean, quantiles, quantiles_e, p, e,
                                           p.new_tensor(window[:2]), p.new_tensor([w, h]))
            arrays.update({'local_' + k: v for k, v in local_stats.items()})
            arrays.update({'global_' + k: v for k, v in global_stats.items()})
            local_p4_stats = feature_metrics(local_encoded[1], points, mean, quantiles, quantiles_e, p, e,
                                            p.new_tensor([0., 0.]), p.new_tensor([640., 640.]),
                                            p.new_tensor([min(640, w-window[0]), min(640, h-window[1])]))
            global_p3_stats = feature_metrics(global_encoded[0], points, mean, quantiles, quantiles_e, p, e,
                                             p.new_tensor(window[:2]), p.new_tensor([w, h]))
            arrays.update({'local_p4_' + k: v for k, v in local_p4_stats.items()})
            arrays.update({'global_p3_' + k: v for k, v in global_p3_stats.items()})
            arrays['q3_span_local_cells'] = arrays['q3_span_px'] / 8
            arrays['q3_span_global_cells'] = arrays['q3_span_px'] / p.new_tensor([w / 40, h / 40, w / 40, h / 40])
            if crop_counter == 0:
                captured_prediction = {k: v.clone() for k, v in prediction.items() if torch.is_tensor(v)}
                capture.close()
                plain = model.predict_local(local[None], crop_norm[None], global_backbone)
                preflight['hooks_no_prediction_change'] = all(torch.equal(v, plain[k]) for k, v in captured_prediction.items())
                assert preflight['hooks_no_prediction_change']
                # Compare a paired call with native cached evaluation, as an independent route check.
                paired = model({'global': full[None], 'local': local[None], 'crop_norm': crop_norm[None]})
                preflight['paired_cached_max_box_difference'] = float((paired['local']['pred_boxes'] - plain['pred_boxes']).abs().max())
                preflight['paired_cached_max_logit_difference'] = float((paired['local']['pred_logits'] - plain['pred_logits']).abs().max())
                assert torch.allclose(paired['local']['pred_boxes'], plain['pred_boxes'], atol=3e-5, rtol=0)
                assert torch.allclose(paired['local']['pred_logits'], plain['pred_logits'], atol=3e-4, rtol=0)
                capture = Capture(model)
                preflight.update({'ordinary_queries': 300, 'captured_layers': 3, 'native_distance_to_mean_check': True,
                                  'two_bin_preserves_native_mean': True, 'source_manifest_verified': True,
                                  'first_distribution_equals_native_second_layer_reference': True,
                                  'last_distribution_reconstructs_native_final_box': True,
                                  'local_padding_mask_excludes_original_image_exterior': True})
                write_json(output / 'preflight.json', preflight)
            trace_path = output / 'traces' / f'{image_row["id"]:04d}_{window_id:02d}.npz'
            np.savez_compressed(trace_path, **{k: v.detach().cpu().numpy() for k, v in arrays.items()},
                                window=np.array(window), image_wh=np.array([w, h]), image_id=np.array(image_row['id']))
            crop_counter += 1
        entry = {'image_id': image_row['id'], 'images_completed': images.index(image_row) + 1,
                 'crops_completed': crop_counter, 'image_seconds': time.perf_counter() - image_started,
                 'elapsed_seconds': time.perf_counter() - started}
        with progress.open('a', encoding='utf8') as f:
            f.write(json.dumps(entry) + '\n')
        print(json.dumps(entry), flush=True)
    capture.close()
    assert model_hash(model) == initial_model_hash
    assert source_hashes() == before_sources
    assert sha(CHECKPOINT) == EXPECTED_CHECKPOINT
    assert all(sha(DATA / 'images' / r['file_name']) == r['sha256'] for r in images)
    write_json(output / 'completion.json', {'status': 'completed_real_image_CPU_diagnostics', 'images': len(images),
        'crops': crop_counter, 'query_instances': crop_counter * 300, 'model_and_BN_unchanged': True,
        'snapshot_sources_unchanged': True, 'checkpoint_unchanged': True, 'images_unchanged': True,
        'elapsed_seconds': time.perf_counter() - started, 'probe_sha256': sha(__file__),
        'trace_hashes': {p.name: sha(p) for p in sorted((output / 'traces').glob('*.npz'))}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--images', type=int, default=32)
    arguments = parser.parse_args()
    if not 1 <= arguments.images <= 548:
        raise ValueError('invalid diagnostic image count')
    run(arguments)
