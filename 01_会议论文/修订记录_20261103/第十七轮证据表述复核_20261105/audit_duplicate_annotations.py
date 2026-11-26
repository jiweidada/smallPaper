"""Compare annotation records for exact JPEG duplicates, without evaluation."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'train_val_byte_overlap_audit.json'
OUTPUT = HERE / 'duplicate_annotation_audit.json'

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert not OUTPUT.exists()
    data = json.loads(INPUT.read_text(encoding='utf-8'))
    assert data['status'] == 'HASHED_CURRENT_FILES'
    groups = []
    for split in ['train', 'validation']:
        info = data[split]
        ann_path = Path(info['annotation'])
        assert digest(ann_path) == info['annotation_sha256']
        annotations = json.loads(ann_path.read_text(encoding='utf-8-sig'))
        name_map = {img['file_name']: img for img in annotations['images']}
        by_image = defaultdict(list)
        for a in annotations['annotations']:
            # ID and image_id necessarily differ; compare fields used by bbox evaluation.
            by_image[a['image_id']].append((a['category_id'], tuple(a['bbox']), a.get('area'), a.get('iscrowd', 0), a.get('ignore')))
        for image_sha, names in info['within_split_identical_byte_groups'].items():
            assert len(names) == 2
            a, b = (name_map[n] for n in names)
            ca, cb = (Counter(by_image[x['id']]) for x in (a, b))
            diff_a, diff_b = list((ca-cb).elements()), list((cb-ca).elements())
            record = {'split': split, 'image_sha256': image_sha, 'files': names,
                      'image_ids': [a['id'], b['id']],
                      'dimensions': [[a['width'], a['height']], [b['width'], b['height']]],
                      'annotation_counts': [sum(ca.values()), sum(cb.values())],
                      'common_bbox_signatures': sum((ca & cb).values()),
                      'unmatched_bbox_signatures': [len(diff_a), len(diff_b)],
                      'annotations_match_exactly_on_bbox_eval_fields': ca == cb,
                      'unmatched_examples': [diff_a[:5], diff_b[:5]]}
            groups.append(record)
    output = {'source_audit': str(INPUT), 'source_sha256': digest(INPUT),
              'groups': groups,
              'scope': 'Exact JPEG byte duplicates within each current split; compares category, bbox, area, iscrowd and ignore only. It does not measure AP impact or establish historical data identity.',
              'models_run': 0, 'evaluator_runs': 0, 'optimizer_updates': 0}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    for group in groups:
        print(json.dumps(group, ensure_ascii=False))

if __name__ == '__main__':
    main()
