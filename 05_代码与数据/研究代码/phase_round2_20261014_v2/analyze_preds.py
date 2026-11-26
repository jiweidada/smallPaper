"""CPU-only paired size/AP/localization analysis of immutable COCO predictions."""
import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import pathlib
from collections import defaultdict
import numpy as np

BINS = [('lt4', 0., 4.), ('4to8', 4., 8.), ('8to16', 8., 16.), ('16to32', 16., 32.)]


def bin_name(box):
    size = math.sqrt(max(0., box[2]*box[3]))
    return next((name for name, lo, hi in BINS if lo <= size < hi), 'ge32')


def transform_data(annotation, predictions, coordinate='input640', border=0):
    """Stretch to 640 coordinates for size AP, then clip the common 32px ROI.

    Original-coordinate appendix uses the corresponding original-image margin.
    Boxes crossing the boundary are clipped in both GT and predictions; counts
    of clipped GT are reported. No additional resize follows the crop.
    """
    data = copy.deepcopy(annotation)
    scales = {}
    for image in data['images']:
        width, height = float(image['width']), float(image['height'])
        sx, sy = (640/width, 640/height) if coordinate == 'input640' else (1., 1.)
        bx, by = border*width/640*sx, border*height/640*sy
        scales[image['id']] = (sx, sy, bx, by, width*sx, height*sy)
        image['width'], image['height'] = width*sx-2*bx, height*sy-2*by
    clipped_gt = 0
    def convert(row, is_gt):
        nonlocal clipped_gt
        row = copy.deepcopy(row)
        sx, sy, bx, by, width, height = scales[row['image_id']]
        x, y, w, h = row['bbox']
        original = [x*sx, y*sy, (x+w)*sx, (y+h)*sy]
        xyxy = original[:]
        if border:
            xyxy = [max(bx, xyxy[0]), max(by, xyxy[1]), min(width-bx, xyxy[2]), min(height-by, xyxy[3])]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            return None
        changed = xyxy != original
        row['bbox'] = [xyxy[0]-bx, xyxy[1]-by, xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
        if is_gt:
            row['_clipped_by_common_roi'] = changed
            row['_original_area_for_small'] = ((row['bbox'][2]*row['bbox'][3])/(sx*sy)
                                                if border else row.get('area', w*h))
            clipped_gt += int(changed)
            row.setdefault('iscrowd', 0)
            if coordinate == 'input640' or border:
                row['area'] = row['bbox'][2]*row['bbox'][3]
            else:
                row.setdefault('area', row['bbox'][2]*row['bbox'][3])
            row.pop('segmentation', None)
        return row
    data['annotations'] = [converted for row in data['annotations'] if (converted := convert(row, True)) is not None]
    result = [converted for row in predictions if (converted := convert(row, False)) is not None]
    return data, result, clipped_gt


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0., min(ax+aw, bx+bw)-max(ax, bx))*max(0., min(ay+ah, by+bh)-max(ay, by))
    union = aw*ah + bw*bh - intersection
    return intersection/union if union > 0 else 0.


def match_gt(annotation, predictions, score_threshold, iou_threshold):
    """Greedy maximum-IoU matching, one-to-one within image and class.

    All eligible edges are ordered by IoU, then score, then stable IDs. This is
    a localization diagnostic, deliberately separate from COCO AP matching.
    """
    groups = defaultdict(list)
    for gt in annotation['annotations']:
        if not gt.get('iscrowd', 0) and not gt.get('ignore', 0):
            groups[(gt['image_id'], gt['category_id'])].append(gt)
    detections = defaultdict(list)
    for index, pred in enumerate(predictions):
        if pred['score'] >= score_threshold:
            detections[(pred['image_id'], pred['category_id'])].append((index, pred))
    matched = {}
    for group, ground_truth in groups.items():
        edges = []
        for gt in ground_truth:
            for index, pred in detections[group]:
                overlap = iou(gt['bbox'], pred['bbox'])
                if overlap >= iou_threshold:
                    edges.append((-overlap, -pred['score'], gt['id'], index, pred))
        taken = set()
        for negative_iou, _, gt_id, index, pred in sorted(edges, key=lambda edge: edge[:4]):
            if gt_id not in matched and index not in taken:
                matched[gt_id] = {'prediction': pred, 'iou': -negative_iou}
                taken.add(index)
    return matched


def localization_error(gt, prediction):
    gx, gy, gw, gh = gt['bbox']
    px, py, pw, ph = prediction['bbox']
    dx = ((px+pw/2)-(gx+gw/2))/gw
    dy = ((py+ph/2)-(gy+gh/2))/gh
    return {'center_error': math.hypot(dx, dy), 'center_dx_over_gt_width': dx,
            'center_dy_over_gt_height': dy, 'log_width_error': math.log(pw/gw),
            'log_height_error': math.log(ph/gh)}


def distribution(values):
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'p90': None, 'p95': None}
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'p95': float(np.quantile(values, .95))}


def coco_ap(annotation, predictions, custom_bins=False):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(annotation)
        gt.dataset.setdefault('info', {})
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = dict(images=copy.deepcopy(annotation['images']), categories=copy.deepcopy(annotation['categories']), annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.maxDets = [1, 10, 100]
        if custom_bins:
            evaluator.params.areaRng = [[lo*lo, float(np.nextafter(hi*hi, -np.inf))] for _, lo, hi in BINS]
            evaluator.params.areaRngLbl = [name for name, _, _ in BINS]
        evaluator.evaluate()
        evaluator.accumulate()
        if not custom_bins:
            evaluator.summarize()
            def small_at(threshold):
                values = evaluator.eval['precision'][np.isclose(evaluator.params.iouThrs, threshold), :, :, 1, -1]
                valid = values[values >= 0]
                return float(valid.mean()) if valid.size else None
            return {'metrics_raw_0_to_1': evaluator.stats.tolist(), 'small_AP50': small_at(.5), 'small_AP75': small_at(.75)}
    precision = evaluator.eval['precision']
    result = {}
    for area, (name, _, _) in enumerate(BINS):
        def mean_at(threshold=None):
            values = precision[:, :, :, area, -1]
            if threshold is not None:
                values = values[np.isclose(evaluator.params.iouThrs, threshold)]
            valid = values[values >= 0]
            return float(valid.mean()) if valid.size else None
        result[name] = {'AP': mean_at(), 'AP50': mean_at(.5), 'AP75': mean_at(.75)}
    return result


def compare(annotation, reference, candidate, score_threshold=.3, border=0, skip_ap=False):
    data, reference, clipped_gt = transform_data(annotation, reference, 'input640', border)
    _, candidate, _ = transform_data(annotation, candidate, 'input640', border)
    valid_gt = [gt for gt in data['annotations'] if not gt.get('ignore', 0) and not gt.get('iscrowd', 0)]
    counts = {name: sum(bin_name(gt['bbox']) == name for gt in valid_gt) for name, _, _ in BINS}
    result = {'coordinate_space': '640_input_pixels_before_crop', 'border_input_pixels': border,
              'boundary_policy': 'clip GT and predictions to the same fixed ROI; no resize after crop',
              'clipped_gt_count': clipped_gt, 'valid_gt_count': len(valid_gt), 'gt_counts_by_size': counts,
              'unreported_ge32_gt_count': sum(bin_name(gt['bbox']) == 'ge32' for gt in valid_gt),
              'score_threshold': score_threshold, 'matching': 'same-class, descending IoU greedy, one-to-one',
              'localization': {}, 'AP': None}
    rows = []
    for threshold in [.5, .75]:
        rm = match_gt(data, reference, score_threshold, threshold)
        cm = match_gt(data, candidate, score_threshold, threshold)
        summaries = {}
        for name in [row[0] for row in BINS] + ['coco_small_original_area']:
            states = defaultdict(int)
            errors = defaultdict(list)
            for gt in valid_gt:
                selected = gt['_original_area_for_small'] < 32*32 if name == 'coco_small_original_area' else bin_name(gt['bbox']) == name
                if not selected:
                    continue
                r, c = rm.get(gt['id']), cm.get(gt['id'])
                state = 'common' if r and c else 'new' if c else 'lost' if r else 'neither'
                states[state] += 1
                row = {'gt_id': gt['id'], 'image_id': gt['image_id'], 'category_id': gt['category_id'],
                       'size_bin': name, 'iou_threshold': threshold, 'state': state,
                       'gt_clipped_by_common_roi': gt['_clipped_by_common_roi']}
                if state == 'common':
                    for label, match in [('reference', r), ('candidate', c)]:
                        row[label+'_iou'] = match['iou']
                        for metric, value in localization_error(gt, match['prediction']).items():
                            row[label+'_'+metric] = value
                            errors[label+'_'+metric].append(abs(value) if metric.startswith('log_') else value)
                if name != 'coco_small_original_area':
                    rows.append(row)
            totals = {key: states[key] for key in ['common', 'new', 'lost', 'neither']}
            number = sum(totals.values())
            summaries[name] = {'GT': number, **totals, 'flip_rate_all_gt': (totals['new']+totals['lost'])/number if number else None,
                                'error_distributions_common_gt': {key: distribution(values) for key, values in errors.items()}}
        result['localization'][str(threshold)] = summaries
    if not skip_ap:
        result['AP'] = {'reference': coco_ap(data, reference, True), 'candidate': coco_ap(data, candidate, True)}
        result['AP']['paired_delta_pp'] = {name: {metric: (result['AP']['candidate'][name][metric]-result['AP']['reference'][name][metric])*100
                                                      if result['AP']['candidate'][name][metric] is not None and result['AP']['reference'][name][metric] is not None else None
                                                 for metric in ['AP', 'AP50', 'AP75']} for name, _, _ in BINS}
    return result, rows


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--score', type=float, default=.3)
    parser.add_argument('--border', type=int, choices=[0, 32], default=0)
    parser.add_argument('--skip-ap', action='store_true', help='Synthetic geometry checks only; no AP claims')
    args = parser.parse_args()
    destination = pathlib.Path(args.out)
    destination.mkdir(parents=True, exist_ok=False)
    annotation = json.loads(pathlib.Path(args.annotations).read_text())
    reference = json.loads(pathlib.Path(args.reference).read_text())
    candidate = json.loads(pathlib.Path(args.candidate).read_text())
    expected_ids = {image['id'] for image in annotation['images']}
    for label, predictions in [('reference', reference), ('candidate', candidate)]:
        if any(pred['image_id'] not in expected_ids for pred in predictions):
            raise ValueError('Unknown image in '+label+' predictions')
        manifest_path = pathlib.Path(args.reference if label == 'reference' else args.candidate).parent/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if not manifest.get('complete') or manifest.get('coordinate_space') != 'original_image':
                raise ValueError('Export is incomplete or uses unsupported coordinates')
            if manifest['predictions_sha256'] != digest(args.reference if label == 'reference' else args.candidate):
                raise ValueError('Prediction bytes differ from export manifest')
            if manifest['annotations_sha256'] != digest(args.annotations):
                raise ValueError('Annotation bytes differ from export manifest')
    result, rows = compare(annotation, reference, candidate, args.score, args.border, args.skip_ap)
    result['sources_sha256'] = {key: digest(value) for key, value in [('annotations', args.annotations), ('reference', args.reference), ('candidate', args.candidate)]}
    if not args.skip_ap:
        appendix = {}
        for label, predictions in [('reference', reference), ('candidate', candidate)]:
            original, predictions, _ = transform_data(annotation, predictions, 'original', args.border)
            appendix[label] = {'coco_original_coordinate': coco_ap(original, predictions), 'size_original_coordinate': coco_ap(original, predictions, True)}
            appendix[label]['gt_counts_original_size'] = {name: sum(bin_name(gt['bbox']) == name and not gt.get('iscrowd', 0) and not gt.get('ignore', 0) for gt in original['annotations']) for name, _, _ in BINS}
        result['original_coordinate_appendix'] = appendix
    with (destination/'analysis.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    fields = sorted({key for row in rows for key in row})
    with (destination/'gt_matching.csv').open('x', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'out': str(destination), 'GT': result['valid_gt_count'], 'AP_computed': not args.skip_ap}))


if __name__ == '__main__':
    main()
