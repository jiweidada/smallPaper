"""Read frozen source and label geometry; no detector, evaluator or training call."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def defaults(tree, cls):
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    return {arg.arg: ast.literal_eval(value) for arg, value in zip(init.args.args[-len(init.args.defaults):], init.args.defaults)}


def labels(annotation_path, split):
    dataset = read(annotation_path)
    images = {int(im['id']): im for im in dataset['images']}
    counts, tiny_counts, classes = Counter(), Counter(), defaultdict(set)
    observations = []
    for a in dataset['annotations']:
        if a.get('iscrowd', 0):
            continue
        im = images[int(a['image_id'])]
        x, y, w, h = map(float, a['bbox'])
        if w <= 0 or h <= 0:
            continue
        # Label geometry, float64, no augmentation or clipping. Matches the
        # existing diagnosis's source-annotation projected shortside definition.
        cx = (x + w / 2) * 640 / im['width']
        cy = (y + h / 2) * 640 / im['height']
        short = min(w * 640 / im['width'], h * 640 / im['height'])
        category = int(a['category_id'])
        assert 0 <= category < 10 and np.isfinite([cx, cy, short]).all()
        inside = 0 <= cx < 640 and 0 <= cy < 640
        cell = (int(a['image_id']), int(cx // 8), int(cy // 8)) if inside else None
        if inside:
            counts[cell] += 1
            tiny_counts[cell] += short < 4
            classes[cell].add(category)
        observations.append((int(a['id']), int(a['image_id']), cx, cy, short, category, cell))
    summary = dict(images=len(images), valid_positive_non_crowd_GT=len(observations),
        centres_inside_640=sum(row[-1] is not None for row in observations),
        centres_outside_640=sum(row[-1] is None for row in observations),
        occupied_P3_cells=len(counts), multi_GT_P3_cells=sum(n > 1 for n in counts.values()),
        GT_in_multi_GT_P3_cells=sum(n for n in counts.values() if n > 1),
        excess_GT_over_one_per_occupied_cell=sum(n - 1 for n in counts.values()),
        multi_category_P3_cells=sum(len(v) > 1 for v in classes.values()),
        input_under4_GT=sum(row[4] < 4 for row in observations),
        input_under4_in_multi_GT_P3_cell=sum(row[4] < 4 and row[-1] is not None and counts[row[-1]] > 1 for row in observations),
        input_under4_sharing_cell_with_another_under4=sum(n for n in tiny_counts.values() if n > 1))

    # Independent whole-array grouping via integer composite keys. The reader
    # above uses per-image tuple Counters; this reference uses sorted NumPy IDs.
    pos = {iid: i for i, iid in enumerate(sorted(images))}
    valid = [row for row in observations if row[-1] is not None]
    keys = np.asarray([pos[row[1]] * 6400 for row in valid], dtype=np.int64) + np.floor(np.asarray([row[3] for row in valid]) / 8).astype(np.int64) * 80 + np.floor(np.asarray([row[2] for row in valid]) / 8).astype(np.int64)
    tiny = np.asarray([row[4] < 4 for row in valid], dtype=np.bool_)
    _, inverse, number = np.unique(keys, return_inverse=True, return_counts=True)
    tiny_per_cell = np.bincount(inverse, weights=tiny, minlength=len(number)).astype(np.int64)
    assert len(number) == summary['occupied_P3_cells']
    assert int((number > 1).sum()) == summary['multi_GT_P3_cells']
    assert int(number[number > 1].sum()) == summary['GT_in_multi_GT_P3_cells']
    assert int((number - 1).sum()) == summary['excess_GT_over_one_per_occupied_cell']
    assert int((tiny & (number[inverse] > 1)).sum()) == summary['input_under4_in_multi_GT_P3_cell']
    assert int(tiny_per_cell[tiny_per_cell > 1].sum()) == summary['input_under4_sharing_cell_with_another_under4']
    if split == 'val':
        assert summary['valid_positive_non_crowd_GT'] == 38759 and summary['input_under4_GT'] == 5096
    by_image = {iid: dict(image_id=iid, GT=0, occupied_P3_cells=0, multi_GT_P3_cells=0) for iid in images}
    for key, number_in_cell in counts.items():
        row = by_image[key[0]]
        row['GT'] += number_in_cell
        row['occupied_P3_cells'] += 1
        row['multi_GT_P3_cells'] += number_in_cell > 1
    rows = [by_image[iid] for iid in sorted(images)]
    return summary, observations, rows


def inspect(packet, output):
    packet, output = Path(packet).resolve(), Path(output).resolve()
    assert not output.exists()
    binding = read(packet / 'code/source_binding.json')['native_files_sha256']
    source_names = ('dfine_decoder.py', 'deim_criterion.py', 'denoising.py', 'matcher.py', 'deim.py')
    sources = {name: packet / 'native/engine/deim' / name for name in source_names}
    for name, path in sources.items():
        assert sha(path) == binding['engine/deim/' + name]
    trees = {name: ast.parse(path.read_text(encoding='utf8')) for name, path in sources.items()}
    config = read(packet / 'prepared/resolved.yml')
    recipe = config['ResidualGeometryDEIM']['recipe']['DFINETransformer']
    actual = dict(defaults(trees['dfine_decoder.py'], 'DFINETransformer'), **recipe)
    criterion = dict(defaults(trees['deim_criterion.py'], 'DEIMCriterion'), **config['DEIMCriterion'])
    assert config['ResidualGeometryDEIM']['arm'] == 'baseline'
    assert actual['num_queries'] == 300 and actual['num_denoising'] == 100
    assert actual['learn_query_content'] is False and actual['mgsa_groups'] == 0 and actual['o2m_t'] is False
    assert defaults(trees['deim.py'], 'DEIM')['o2m_c'] is False
    assert criterion['use_uni_set'] is True and criterion['losses'] == ['mal', 'boxes', 'local']
    spans = {
        'topK_then_encoder_bbox': dict(source='dfine_decoder.py', lines=[825, 850]),
        'selected_logits_gather': dict(source='dfine_decoder.py', lines=[884, 908]),
        'DN_construct_and_split': dict(source='dfine_decoder.py', lines=[1026, 1088]),
        'encoder_aux_and_DN_losses': dict(source='deim_criterion.py', lines=[673, 731]),
        'regression_union': dict(source='deim_criterion.py', lines=[590, 609]),
        'native_one_to_one_matcher': dict(source='matcher.py', lines=[191, 211])}
    evidence = {key: dict(**value, source_sha256=sha(sources[value['source']]),
        text='\n'.join(sources[value['source']].read_text(encoding='utf8').splitlines()[value['lines'][0]-1:value['lines'][1]]))
        for key, value in spans.items()}
    output.mkdir()
    summaries = {}
    for split in ('train', 'val'):
        path = packet / f'reference/{split}_annotations.json'
        summary, observations, rows = labels(path, split)
        summaries[split] = dict(**summary, annotation_sha256=sha(path), whole_array_count_reference_passed=True)
        np.savez_compressed(output / f'{split}_label_geometry.npz',
            annotation_id=np.asarray([x[0] for x in observations], dtype=np.int64),
            image_id=np.asarray([x[1] for x in observations], dtype=np.int64),
            centre640=np.asarray([[x[2], x[3]] for x in observations], dtype=np.float64),
            shortside640=np.asarray([x[4] for x in observations], dtype=np.float64),
            category_id=np.asarray([x[5] for x in observations], dtype=np.int64))
        with (output / f'{split}_cell_counts.csv').open('x', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(status='FROZEN_SOURCE_PATH_AND_LABEL_CELL_DESCRIPTION', identity=read(packet / 'prepared/receipt.json')['identity'],
        code_sha256=sha(__file__), source_evidence=evidence,
        active_config=dict(num_queries=300, learn_query_content=False, MGSA_groups=0, O2M_transformer=False,
            O2M_CNN=False, num_denoising_parameter=100, DN_actual_count_can_exceed100=True,
            losses=criterion['losses'], regression_uses_matching_union=True),
        gradient_path_findings=[
            'Encoder bbox head is applied after scoreTopK, only to300 selected token vectors.',
            'Encoder auxiliary MAL sees selected logits; bbox/GIoU uses the shared matching union. Shared head parameters still learn from selected entries.',
            'Initial reference boxes and native selected query content are detached before decoder input. Decoder losses have no direct initial-reference/content gradient route to encoder heads.',
            'Decoder cross-attention receives original encoder memory, so shared encoder/backbone features can receive decoder/DN gradients. The proposal valid-mask rebind is not an in-place erasure of all decoder memory.',
            'DN inputs are constructed from GT boxes/labels and bypass encoder bbox head. DN supplies GT supervision to decoder but does not directly supervise the initial encoder bbox head.',
            'These are source-level connectivity findings, not measured gradient amplitudes, proof of an optimization bug, or evidence that GT are globally unsupervised.'],
        label_cell_counts=summaries,
        interpretation='Multiple GT mapped to the same containing8px P3 cell create conflicting exact targets for a naive one-box-per-cell auxiliary label rule. Neighbouring cells, multi-scale queries and decoder motion can resolve them. These counts are not observed TopK membership, a detector-capacity/AP bound, a unique failure cause or actual augmented training exposure.',
        literature_boundary='Dense supervision already exists in RT-DETRv3 (WACV2025). PaQ-DETR (CVPR2026) addresses shared pattern/query supervision; our native queries are dynamic score-ranked token content, so persistent learnable-query activation statistics cannot be copied as a diagnosis without a matching observation unit.',
        new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0,
        source_or_acceptance_gates_changed=False, current_random_controls_unchanged=True)
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    write(output / 'report.json', report)
    write(output / 'manifest.json', dict(files={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    return dict(status=report['status'], label_cell_counts=summaries,
        new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--packet', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); print(json.dumps(inspect(a.packet, a.output)), flush=True)
