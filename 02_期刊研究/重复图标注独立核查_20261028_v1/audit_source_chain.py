"""Six-file lineage audit, PIL + standard library only; no NumPy or models."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DISCOVERY = PROJECT / '新问题入口重寻_20261028_v1/image_duplicates_attempt02'
RAW_ROOT = Path('D:/数据集/visdrone2019')
OLD_SNAPSHOT = PROJECT / 'data/00_实验计划与证据索引/原始遮挡忽略与像素对照独立复核_20261006_v1/原始验证标注快照'
GROUPS = [[191, 590], [2768, 4366], [6795, 6834]]
COORD_ATOL = 0.01


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


INPUT_HASHES = {}


def bind(path):
    path = Path(path)
    assert path.is_file(), str(path)
    value = sha(path)
    INPUT_HASHES[str(path)] = value
    return value


def image_record(path):
    digest = bind(path)
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        result = {'path': str(path), 'bytes_sha256': digest,
                  'width': rgb.width, 'height': rgb.height,
                  'RGB_bytes_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
    return result


def raw_records(path):
    digest = bind(path)
    rows = []
    for i, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = [float(v) for v in line.strip().strip(',').split(',')]
        assert len(fields) == 8, (str(path), i, fields)
        assert all(v.is_integer() for v in fields), (str(path), i, fields)
        rows.append({'row_1based': i, 'values': [int(v) for v in fields]})
    return digest, rows


def positive_counter(rows, score1_only=False):
    return Counter(tuple(r['values'][:4] + [r['values'][5] - 1]) for r in rows
                   if 1 <= r['values'][5] <= 10 and
                   (not score1_only or r['values'][4] == 1))


def counter_rows(counter):
    return [{'xywh_and_class0': list(key), 'multiplicity': count}
            for key, count in sorted(counter.items())]


protocol = HERE / '00_源链只读核对_事前登记.txt'
bind(protocol)
bind(DISCOVERY / 'source_rows.json')
bind(DISCOVERY / 'exact_groups.json')
source_rows = read_json(DISCOVERY / 'source_rows.json')
exact = read_json(DISCOVERY / 'exact_groups.json')
assert exact['compressed_bytes'] == GROUPS
assert exact['decoded_RGB_pixels'] == GROUPS

cocos = {}
for split, index in [('train', 191), ('val', 6795)]:
    folder = Path(source_rows[index]['path']).parent.parent
    path = folder / 'annotations' / (split + '.json')
    bind(path)
    data = read_json(path)
    assert {int(x['id']) for x in data['categories']} == set(range(10))
    images = {x['id']: x for x in data['images']}
    annotations = defaultdict(list)
    for item in data['annotations']:
        annotations[item['image_id']].append(item)
    cocos[split] = {'path': path, 'images': images, 'annotations': annotations}

snapshots = HERE / 'raw_annotation_snapshots'
snapshots.mkdir(exist_ok=False)
items = []
raw_by_index = {}
coco_excerpt = []
for global_index in [i for group in GROUPS for i in group]:
    source = source_rows[global_index]
    split = source['split']
    current_image = Path(source['path'])
    original_folder = RAW_ROOT / ('VisDrone2019-DET-' + split)
    original_image = original_folder / 'images' / source['file_name']
    raw_path = original_folder / 'annotations' / (original_image.stem + '.txt')
    current_raw_path = current_image.parent.parent / 'annotations' / raw_path.name
    current = image_record(current_image)
    original = image_record(original_image)
    assert current['bytes_sha256'] == source['bytes_sha256'], global_index
    raw_digest, raw = raw_records(raw_path)
    raw_by_index[global_index] = raw
    with (snapshots / (split + '_' + raw_path.name)).open('xb') as f:
        f.write(raw_path.read_bytes())
    positive = positive_counter(raw)
    positive_score1 = positive_counter(raw, score1_only=True)
    image_meta = cocos[split]['images'][source['image_id']]
    assert image_meta['file_name'] == source['file_name']
    anns = cocos[split]['annotations'][source['image_id']]
    rounded = Counter(tuple([round(float(v)) for v in a['bbox']] + [int(a['category_id'])]) for a in anns)
    max_delta = max((abs(float(v) - round(float(v))) for a in anns for v in a['bbox']), default=0.)
    same_shape = ((current['width'], current['height']) ==
                  (original['width'], original['height']) ==
                  (source['width'], source['height']) ==
                  (image_meta['width'], image_meta['height']))
    record = {'global_source_row': global_index, 'split': split,
              'image_id': source['image_id'], 'file_name': source['file_name'],
              'current_image': current, 'original_image': original,
              'C_equals_D_compressed_bytes': current['bytes_sha256'] == original['bytes_sha256'],
              'C_equals_D_RGB': current['RGB_bytes_sha256'] == original['RGB_bytes_sha256'],
              'all_dimensions_equal': same_shape,
              'raw_TXT_path': str(raw_path), 'raw_TXT_sha256': raw_digest,
              'raw_rows_total': len(raw), 'raw_positive_rows_all_scores': sum(positive.values()),
              'raw_positive_score1': sum(positive_score1.values()),
              'raw_positive_score0': sum(r['values'][4] == 0 and 1 <= r['values'][5] <= 10 for r in raw),
              'raw_nonpositive_class_counts': dict(sorted(Counter(r['values'][5] for r in raw if r['values'][5] not in range(1, 11)).items())),
              'raw_positive_class0_counts': dict(sorted(Counter(r['values'][5] - 1 for r in raw if 1 <= r['values'][5] <= 10).items())),
              'COCO_annotations': len(anns), 'COCO_max_rounding_difference_original_px': max_delta,
              'COCO_rounding_within_0_01_px': max_delta <= COORD_ATOL,
              'COCO_equals_own_D_raw_all_positive_keys': rounded == positive,
              'COCO_equals_own_D_raw_score1_keys': rounded == positive_score1,
              'COCO_only_keys': counter_rows(rounded - positive),
              'raw_only_keys': counter_rows(positive - rounded)}
    if current_raw_path.exists():
        current_raw_digest, current_raw = raw_records(current_raw_path)
        record['C_raw_TXT'] = {'path': str(current_raw_path), 'sha256': current_raw_digest,
                               'bytes_equal_D_raw': current_raw_digest == raw_digest,
                               'eight_field_multiset_equal_D_raw': Counter(tuple(x['values']) for x in current_raw) == Counter(tuple(x['values']) for x in raw)}
    if split == 'val':
        snapshot = OLD_SNAPSHOT / raw_path.name
        record['old_20261006_val_snapshot'] = {'exists': snapshot.exists()}
        if snapshot.exists():
            old_digest = bind(snapshot)
            record['old_20261006_val_snapshot'].update(path=str(snapshot), sha256=old_digest, bytes_equal_current_D=old_digest == raw_digest)
    items.append(record)
    coco_excerpt.append({'global_source_row': global_index, 'split': split,
                         'COCO_source': str(cocos[split]['path']), 'image': image_meta,
                         'annotations': anns})

by_index = {x['global_source_row']: x for x in items}
groups = []
for group in GROUPS:
    left, right = [by_index[i] for i in group]
    raw_left, raw_right = [raw_by_index[i] for i in group]
    ca, cb = positive_counter(raw_left), positive_counter(raw_right)
    ga = Counter(tuple(row['values'][:4]) for row in raw_left if 1 <= row['values'][5] <= 10)
    gb = Counter(tuple(row['values'][:4]) for row in raw_right if 1 <= row['values'][5] <= 10)
    geom_common = sum((ga & gb).values())
    exact_common = sum((ca & cb).values())
    groups.append({'global_source_rows': group, 'split': left['split'],
                   'image_ids': [left['image_id'], right['image_id']],
                   'file_names': [left['file_name'], right['file_name']],
                   'C_byte_equal': left['current_image']['bytes_sha256'] == right['current_image']['bytes_sha256'],
                   'C_RGB_equal': left['current_image']['RGB_bytes_sha256'] == right['current_image']['RGB_bytes_sha256'],
                   'D_byte_equal': left['original_image']['bytes_sha256'] == right['original_image']['bytes_sha256'],
                   'D_RGB_equal': left['original_image']['RGB_bytes_sha256'] == right['original_image']['RGB_bytes_sha256'],
                   'C_equals_D_each': all(x['C_equals_D_compressed_bytes'] and x['C_equals_D_RGB'] for x in [left, right]),
                   'D_raw_TXT_byte_equal': left['raw_TXT_sha256'] == right['raw_TXT_sha256'],
                   'D_raw_eight_field_multisets_equal': Counter(tuple(x['values']) for x in raw_left) == Counter(tuple(x['values']) for x in raw_right),
                   'D_raw_positive_class_geometry_multisets_equal': ca == cb,
                   'raw_positive_counts': [sum(ca.values()), sum(cb.values())],
                   'COCO_counts': [left['COCO_annotations'], right['COCO_annotations']],
                   'exact_positive_class_geometry_common': exact_common,
                   'exact_positive_geometry_common_ignoring_class': geom_common,
                   'minimum_class_disagreements_on_exact_shared_geometry': geom_common - exact_common,
                   'left_only_exact_class_geometry_keys': counter_rows(ca - cb),
                   'right_only_exact_class_geometry_keys': counter_rows(cb - ca),
                   'interpretation_limit': 'Exact annotation-version set difference, not truth adjudication or object-level IoU matching.'})

changed = [path for path, digest in INPUT_HASHES.items() if sha(Path(path)) != digest]
assert not changed, changed
save('selected_COCO_records.json', coco_excerpt)
save('source_chain_results.json', {'status': 'PASS', 'date_utc': datetime.now(timezone.utc).isoformat(),
                                 'discovery_scope': 'Three peer-selected exact duplicate pairs only, current local source identities.',
                                 'COCO_rounding_tolerance_original_px': COORD_ATOL,
                                 'items': items, 'groups': groups,
                                 'input_hashes_before_and_after_equal': INPUT_HASHES,
                                 'changed_inputs': changed,
                                 'operations': {'new_model_loads': 0, 'forwards': 0, 'training': 0,
                                                'gradients': 0, 'new_COCO_evaluation': 0,
                                                'IoU_assignment': 0, 'GPU_remote_test_reads': 0,
                                                'NumPy_BLAS_calls': 0, 'original_D_images_decoded': 6,
                                                'current_C_images_decoded': 6}})
print(json.dumps({'status': 'PASS', 'groups': [{k: g[k] for k in ['split', 'image_ids', 'D_byte_equal', 'C_equals_D_each', 'D_raw_eight_field_multisets_equal', 'raw_positive_counts', 'COCO_counts', 'exact_positive_class_geometry_common', 'exact_positive_geometry_common_ignoring_class', 'minimum_class_disagreements_on_exact_shared_geometry']} for g in groups],
                  'all_COCOs_equal_own_original_raw': all(x['COCO_equals_own_D_raw_all_positive_keys'] and x['COCO_rounding_within_0_01_px'] for x in items),
                  'all_dimensions_equal': all(x['all_dimensions_equal'] for x in items),
                  'max_coordinate_delta': max(x['COCO_max_rounding_difference_original_px'] for x in items),
                  'val_old_snapshots_equal': [x['old_20261006_val_snapshot'].get('bytes_equal_current_D') for x in items if x['split'] == 'val'],
                  'changed_inputs': changed}, ensure_ascii=False))
