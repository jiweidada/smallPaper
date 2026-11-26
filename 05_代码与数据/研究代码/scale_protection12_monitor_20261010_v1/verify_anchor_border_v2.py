"""Independent scalar spatial membership and GT-count recomputation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(protocol, output):
    protocol, output = Path(protocol), Path(output)
    spec = read(protocol)
    for path, expected in spec['bound_files'].items():
        assert sha(path) == expected, path
    assert sha(Path(__file__).with_name('analyze_anchor_border.py')) == spec['reader_sha256']
    root = protocol.parent
    manifest = read(root / 'analysis/manifest.json')
    for name, expected in manifest['files'].items():
        assert sha(root / 'analysis' / name) == expected
    report = read(root / 'analysis/report.json')
    assert report['protocol_sha256'] == sha(protocol)
    proofs = read(root / 'saved_buffer_CPU_readers.json')
    for seed, payload in proofs.items():
        assert payload['initial_failed_reader']['receipt']['exit_code'] == 1
        proof = payload['corrected_path_reader']
        assert proof['receipt']['exit_code'] == 0
        for level in proof['report']['levels']:
            h, w = level['height'], level['width']
            expected = [[y, x] for y in range(h) for x in range(w)
                if min((x + .5) / w, (y + .5) / h) <= .01 or
                   max((x + .5) / w, (y + .5) / h) >= .99]
            assert expected == level['invalid_row_column']
        assert all(g['mask_equal'] and g['nonfinite_pattern_equal'] and
                   g['invalid_count'] == 316 and g['positive_inf_elements'] == 1264
                   for g in proof['report']['groups'].values())
    ann = read(spec['annotations'])
    images = {int(a['id']): a for a in ann['images']}
    entries = sorted(ann['annotations'], key=lambda a: int(a['id']))
    with np.load(root / 'analysis/spatial_per_GT.npz', allow_pickle=False) as data:
        spatial = {name: data[name] for name in data.files}
    fields = ('GT', 'initial_IoU75_support', 'pre_IoU75_support', 'final_IoU75_support',
              'final_IoU75_failed', 'FDR_range_reachable', 'native_all_area_TP75')
    checked = {}
    for seed, paths in spec['seeds'].items():
        with np.load(paths['range_per_GT'], allow_pickle=False) as data:
            ranges = {name: data[name] for name in data.files}
        with np.load(paths['selection_per_GT'], allow_pickle=False) as data:
            selection = {name: data[name] for name in data.files}
        counts = {cohort: {region: {k: 0 for k in fields} for region in rows}
                  for cohort, rows in report['seeds'][seed].items()}
        for j, a in enumerate(entries):
            image = images[int(a['image_id'])]
            x, y, w, h = a['bbox']
            cx = (x + w * .5) * 640 / image['width']
            cy = (y + h * .5) * 640 / image['height']
            shortside = min(w * 640 / image['width'], h * 640 / image['height'])
            inside = 0 <= cx < 640 and 0 <= cy < 640
            # Direct pixel inequalities, independent of vectorized grid-floor analysis.
            border = inside and (cx < 8 or cy < 8 or cx >= 632 or cy >= 632)
            region = ('P3_outer_cell_centers' if border else 'P3_interior_cell_centers') if inside else 'outside_image_centers'
            assert int(a['id']) == int(spatial['annotation_id'][j]) == int(ranges['annotation_id'][j]) == int(selection['annotation_id'][j])
            assert border == bool(spatial['P3_outer_cell_center'][j]) and inside == bool(spatial['center_inside_image'][j])
            assert all(math.isclose(v, float(t), rel_tol=0, abs_tol=1e-10)
                       for v, t in zip((cx, cy), spatial['centers_640'][j]))
            cohorts = ['all', 'input_under4' if shortside < 4 else 'input_4_to_under8' if shortside < 8 else 'input_ge8']
            if a['area'] <= 1024:
                cohorts.append('COCO_small')
            m = ranges['max_iou'][j]
            values = dict(GT=1, initial_IoU75_support=int(m[0] >= .75), pre_IoU75_support=int(m[1] >= .75),
                final_IoU75_support=int(m[2] >= .75), final_IoU75_failed=int(m[2] < .75),
                FDR_range_reachable=int(ranges['any_closed_range'][j]), native_all_area_TP75=int(selection['native_TP_75'][j]))
            for cohort in cohorts:
                for key in ('all', region):
                    for field, value in values.items():
                        counts[cohort][key][field] += value
        assert counts == report['seeds'][seed]
        checked[seed] = len(entries)
    result = dict(status='PASS_SCALAR_SPATIAL_AND_ALL_COUNTS', per_seed_GT_checked=checked,
        protocol_sha256=sha(protocol), analysis_manifest_sha256=sha(root / 'analysis/manifest.json'),
        verifier_sha256=sha(__file__), actual_saved_mask_grid_positions_verified=True,
        new_model_forwards=0, new_evaluator_calls=0, new_epochs=0, new_optimizer_updates=0,
        scope='Independent grid-index and direct-pixel-membership/count checks. No extra inference or causal intervention.')
    assert 'torch' not in sys.modules
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.protocol, args.output)), flush=True)
