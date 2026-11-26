"""Verify exact duplicates, the original text labels, and CPU receipt bindings."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import filecmp
import json

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
DATA = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone')

def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    out = HERE / 'duplicate_provenance_verification.json'
    assert not out.exists()
    pair_audit = json.loads((HERE/'duplicate_annotation_audit.json').read_text(encoding='utf-8'))
    split_audit = json.loads((HERE/'train_val_byte_overlap_audit.json').read_text(encoding='utf-8'))
    assert pair_audit['source_sha256'] == digest(HERE/'train_val_byte_overlap_audit.json')
    details = []
    for group in pair_audit['groups']:
        part = 'VisDrone2019-DET-train' if group['split'] == 'train' else 'VisDrone2019-DET-val'
        folder = DATA/part
        image_paths = [folder/'images'/n for n in group['files']]
        assert filecmp.cmp(image_paths[0], image_paths[1], shallow=False)
        assert all(digest(p) == group['image_sha256'] for p in image_paths)
        info = split_audit[group['split']]
        coco_path = Path(info['annotation'])
        coco = json.loads(coco_path.read_text(encoding='utf-8-sig'))
        image_ids = {x['file_name']: x['id'] for x in coco['images']}
        by_image = defaultdict(list)
        for annotation in coco['annotations']:
            by_image[annotation['image_id']].append(annotation)
        label_checks = []
        for name in group['files']:
            text_path = folder/'annotations'/(Path(name).stem + '.txt')
            original = []
            for line in text_path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                fields = [int(x) for x in line.split(',')]
                if fields[4] == 1 and 1 <= fields[5] <= 10:
                    original.append(fields)
            coco_objects = by_image[image_ids[name]]
            assert len(original) == len(coco_objects)
            assert all(a[5]-1 == b['category_id'] for a, b in zip(original, coco_objects))
            max_error = max((abs(v-w) for a, b in zip(original, coco_objects)
                             for v, w in zip(a[:4], b['bbox'])), default=0.0)
            label_checks.append({'file': name, 'original_text': str(text_path),
                                 'original_text_sha256': digest(text_path),
                                 'retained_original_box_count': len(original),
                                 'coco_box_count': len(coco_objects),
                                 'original_category_counts': dict(Counter(a[5] for a in original)),
                                 'original_to_coco_max_coordinate_difference_px': max_error})
        details.append({'split': group['split'], 'image_sha256': group['image_sha256'],
                        'byte_for_byte_comparison_equal': True, 'labels': label_checks})
    val_manifest = PAPER/'本地CPU核验_20261103/validation_image_manifest.json'
    val_hash = digest(val_manifest)
    val_annotation_hash = split_audit['validation']['annotation_sha256']
    receipts = []
    for run in ['N0', 'N4', 'S1']:
        path = PAPER/f'本地CPU核验_20261103/results/{run}/receipt.json'
        receipt = json.loads(path.read_text(encoding='utf-8-sig'))
        assert receipt['status'] == 'COMPLETED' and receipt['completed_images'] == 548
        assert receipt['image_manifest_sha256'] == val_hash
        assert receipt['annotation_sha256'] == val_annotation_hash
        receipts.append({'run_id': run, 'path': str(path), 'sha256': digest(path),
                         'image_manifest_sha256': val_hash, 'annotation_sha256': val_annotation_hash})
    evidence = {'status': 'BYTE_IDENTITY_AND_CPU_INPUT_RECEIPTS_VERIFIED',
                'source_audit_sha256': digest(HERE/'train_val_byte_overlap_audit.json'),
                'pair_annotation_audit_sha256': digest(HERE/'duplicate_annotation_audit.json'),
                'duplicate_groups': details, 'cpu_receipts': receipts,
                'limits': ['Current inputs and original annotation text are checked, not the official downloaded image archive.',
                           'No causal explanation for the duplicate files is established.',
                           'No model, AP recalculation, deduplication or data modification was performed.',
                           'Historical training data identity remains unverified.']}
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
