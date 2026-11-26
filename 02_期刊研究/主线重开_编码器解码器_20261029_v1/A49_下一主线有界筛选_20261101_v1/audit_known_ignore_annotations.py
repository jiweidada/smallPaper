"""Annotation-only audit, matched to A43 parsing. No torch/model imports."""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics
import time

ROOT = Path('/root/uav_a49_20261101_v1')
DATA = Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')
REFERENCE = Path('/root/uav_a48_20261101_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_area(rects):
    if not rects:
        return 0.0
    xs = sorted({r[i] for r in rects for i in [0, 2]})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rects if r[0] < right and r[2] > left)
        height = 0.0
        end = -math.inf
        for lo, hi in intervals:
            height += max(0.0, hi - max(lo, end))
            end = max(end, hi)
        total += (right - left) * height
    return total


def clip(row, width, height):
    x, y, w, h = row[:4]
    rect = [max(0.0, min(width, x)), max(0.0, min(height, y)),
            max(0.0, min(width, x + w)), max(0.0, min(height, y + h))]
    return rect if rect[2] > rect[0] and rect[3] > rect[1] else None


def intersection(a, b):
    rect = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return rect if rect[2] > rect[0] and rect[3] > rect[1] else None


if __name__ == '__main__':
    output = ROOT / 'annotations_audit_v1'
    assert not output.exists(), 'Do not overwrite an audit or blindly retry.'
    results, all_rows = {}, {}
    start = time.monotonic()
    for split in ['train', 'val']:
        manifest_path = REFERENCE / (split + '_data_manifest.json')
        manifest = json.loads(manifest_path.read_text())
        annotations = DATA / ('VisDrone2019-DET-' + split) / 'annotations'
        assert len(list(annotations.glob('*.txt'))) == manifest['images']
        rows = []
        raw_categories = collections.Counter()
        verified = 0
        for expected in manifest['rows']:
            path = annotations / (Path(expected['file']).stem + '.txt')
            assert sha(path) == expected['annotation_sha256'], str(path)
            verified += 1
            width, height = expected['width'], expected['height']
            valid, ignores = [], []
            score0_valid_categories = 0
            for line in path.read_text().splitlines():
                values = [float(v.strip()) for v in line.split(',') if v.strip()]
                assert len(values) >= 8 and all(math.isfinite(v) for v in values[:8])
                score, category = values[4], int(values[5])
                raw_categories[str(category)] += 1
                rect = clip(values, width, height)
                if rect is None:
                    continue
                if category == 0:
                    ignores.append(rect)
                if score > 0 and 1 <= category <= 10:
                    valid.append(rect)
                if score <= 0 and 1 <= category <= 10:
                    score0_valid_categories += 1
            assert len(valid) == expected['valid_gt'], expected['file']
            covered = union_area(ignores)
            assert 0 <= covered <= width * height + 1e-6
            gt_ioa_half = 0
            for gt in valid:
                overlaps = [x for other in ignores if (x := intersection(gt, other)) is not None]
                gt_ioa_half += union_area(overlaps) >= .5 * (gt[2] - gt[0]) * (gt[3] - gt[1])
            rows.append({'file': expected['file'], 'valid_gt': len(valid),
                         'ignored_rectangles': len(ignores), 'image_area': width * height,
                         'ignore_union_area': covered, 'ignore_union_fraction': covered / (width * height),
                         'valid_GT_IoA_ignore_ge_half': gt_ioa_half,
                         'score0_clipped_boxes_in_classes1to10': score0_valid_categories,
                         'annotation_sha256': expected['annotation_sha256']})
        assert len(rows) == manifest['images'] and sum(r['valid_gt'] for r in rows) == manifest['valid_gt']
        fractions = [r['ignore_union_fraction'] for r in rows]
        results[split] = {'images': len(rows), 'annotations_SHA_verified': verified,
                          'valid_GT': manifest['valid_gt'],
                          'images_with_ignored_rectangles': sum(r['ignored_rectangles'] > 0 for r in rows),
                          'ignored_rectangles': sum(r['ignored_rectangles'] for r in rows),
                          'mean_per_image_ignore_union_fraction': statistics.mean(fractions),
                          'median_per_image_ignore_union_fraction': statistics.median(fractions),
                          'total_ignore_union_area_over_total_image_area': sum(r['ignore_union_area'] for r in rows) / sum(r['image_area'] for r in rows),
                          'valid_GT_IoA_ignore_ge_half': sum(r['valid_GT_IoA_ignore_ge_half'] for r in rows),
                          'score0_clipped_boxes_in_classes1to10': sum(r['score0_clipped_boxes_in_classes1to10'] for r in rows),
                          'raw_annotation_category_counts': dict(raw_categories),
                          'reference_manifest_sha256': sha(manifest_path)}
        all_rows[split] = rows
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / 'image_rows.json'
    with rows_path.open('x') as f:
        json.dump(all_rows, f, indent=2, allow_nan=False)
    result = {'status': 'COMPLETE_ANNOTATION_ONLY', 'splits': results,
              'script_sha256': sha(Path(__file__)), 'rows_sha256': sha(rows_path),
              'rows_path': str(rows_path), 'elapsed_seconds': time.monotonic() - start,
              'model_forwards': 0, 'optimizer_updates': 0, 'GPU_jobs_started': 0,
              'limits': 'Known ignore-region area is not missing-instance count or evidence of causal detection harm. Valid GT overlap is not false-negative query supervision. No natural label-omission rate estimated.'}
    with (output / 'result.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False), flush=True)
