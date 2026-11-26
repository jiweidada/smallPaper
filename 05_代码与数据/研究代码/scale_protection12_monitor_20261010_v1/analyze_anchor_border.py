"""Spatial association using saved GT/geometry only; no causal or AP claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)


def analyze(protocol, output):
    protocol, output = Path(protocol), Path(output)
    spec = read(protocol)
    assert spec['reader_sha256'] == sha(__file__) and not output.exists()
    for path, expected in spec['bound_files'].items():
        assert sha(path) == expected, path
    ann = read(spec['annotations'])
    images = {int(i['id']): i for i in ann['images']}
    entries = sorted(ann['annotations'], key=lambda a: int(a['id']))
    assert len(entries) == 38759 and len(images) == 548
    ids = np.array([int(a['id']) for a in entries])
    boxes = np.array([a['bbox'] for a in entries], dtype=np.float64)
    sizes = np.array([[images[int(a['image_id'])]['width'],
                       images[int(a['image_id'])]['height']] for a in entries])
    centers = (boxes[:, :2] + boxes[:, 2:] / 2) / sizes * 640
    inside = ((centers >= 0) & (centers < 640)).all(1)
    cells = np.floor(centers / 8).astype(np.int64)
    border = inside & ((cells == 0) | (cells == 79)).any(1)
    direct = inside & ((centers[:, 0] < 8) | (centers[:, 0] >= 632) |
                       (centers[:, 1] < 8) | (centers[:, 1] >= 632))
    assert np.array_equal(border, direct)
    shortside = np.min(boxes[:, 2:] / sizes * 640, axis=1)
    area = np.array([a['area'] for a in entries])
    cohorts = dict(all=np.ones(len(ids), dtype=bool), COCO_small=area <= 1024,
                   input_under4=shortside < 4, input_4_to_under8=(shortside >= 4) & (shortside < 8),
                   input_ge8=shortside >= 8)
    output.mkdir()
    reports = {}
    for seed, paths in spec['seeds'].items():
        range_data = np.load(paths['range_per_GT'], allow_pickle=False)
        selection = np.load(paths['selection_per_GT'], allow_pickle=False)
        assert np.array_equal(range_data['annotation_id'], ids)
        assert np.array_equal(selection['annotation_id'], ids)
        assert np.allclose(range_data['input_shortside'], shortside, rtol=0, atol=1e-10)
        assert np.array_equal(range_data['original_area'], area)
        max_iou = range_data['max_iou']
        assert np.array_equal(max_iou[:, 2] >= .75, selection['geometry_75'])
        groups = {}
        for name, cohort in cohorts.items():
            rows = {}
            for spatial, mask in [('all', cohort), ('P3_outer_cell_centers', cohort & border),
                                  ('P3_interior_cell_centers', cohort & inside & ~border),
                                  ('outside_image_centers', cohort & ~inside)]:
                rows[spatial] = dict(GT=int(mask.sum()),
                    initial_IoU75_support=int((mask & (max_iou[:, 0] >= .75)).sum()),
                    pre_IoU75_support=int((mask & (max_iou[:, 1] >= .75)).sum()),
                    final_IoU75_support=int((mask & (max_iou[:, 2] >= .75)).sum()),
                    final_IoU75_failed=int((mask & (max_iou[:, 2] < .75)).sum()),
                    FDR_range_reachable=int((mask & range_data['any_closed_range']).sum()),
                    native_all_area_TP75=int((mask & selection['native_TP_75']).sum()))
            assert sum(rows[x]['GT'] for x in ('P3_outer_cell_centers',
                'P3_interior_cell_centers', 'outside_image_centers')) == rows['all']['GT']
            groups[name] = rows
        reports[seed] = groups
    result = dict(schema='anchor_border_association_v1', seeds=reports,
        protocol_sha256=sha(protocol), independent_cell_membership_check='PASS',
        new_model_forwards=0, new_optimizer_updates=0, new_epochs=0, new_evaluator_calls=0,
        scope=spec['scope'], interpretation=spec['interpretation'])
    write(output / 'report.json', result)
    np.savez_compressed(output / 'spatial_per_GT.npz', annotation_id=ids,
        centers_640=centers, P3_cell=cells, center_inside_image=inside,
        P3_outer_cell_center=border, input_shortside=shortside, original_area=area)
    write(output / 'manifest.json', dict(files={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    assert 'torch' not in sys.modules
    return {seed: groups['input_under4'] for seed, groups in reports.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.protocol, args.output), ensure_ascii=False), flush=True)
