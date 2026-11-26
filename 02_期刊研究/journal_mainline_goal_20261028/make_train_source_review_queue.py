"""Build a small source-label review queue; never infer visibility or evaluate a model."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).parent / 'smot_source_metadata_20261029'
source = root / 'author_train_annotations.json'
data = json.loads(source.read_text(encoding='utf-8'))
videos = {v['id']: v['name'] for v in data['videos']}
images = {im['id']: im for im in data['images']}
by_image = defaultdict(list)
by_video = defaultdict(list)
for im in data['images']:
    by_video[im['video_id']].append(im)
for a in data['annotations']:
    by_image[a['image_id']].append(a)

def intersects(a, b):
    return min(a[0]+a[2], b[0]+b[2]) > max(a[0], b[0]) and min(a[1]+a[3], b[1]+b[3]) > max(a[1], b[1])

selected = {'0005', '0019', '0028', '0042', '0044'}
receipt = {
    'scope': 'TRAIN source annotation indexing for later manual inspection only',
    'train_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'model_run': False, 'prediction_or_coco_matching': False, 'source_pixels_read': False,
    'warning': 'No-GT is not proof of absence; the same image coordinates are not proof of the same world position. This queue contains zero verified pairs.',
    'videos': []
}
for vid, name in videos.items():
    if name not in selected:
        continue
    ims = sorted(by_video[vid], key=lambda im: im['frame_id'])
    counts = [len(by_image[im['id']]) for im in ims]
    queue = []
    # Deterministic source-only candidate indexing: earliest labelled box whose
    # exact image-coordinate rectangle has no annotated box on a later frame.
    for pos in ims:
        for a in sorted(by_image[pos['id']], key=lambda a: a['id']):
            later = [im for im in ims if im['frame_id'] > pos['frame_id'] and not any(intersects(a['bbox'], b['bbox']) for b in by_image[im['id']])]
            if later:
                neg = later[0]
                queue.append({'positive_file': pos['file_name'], 'positive_image_id': pos['id'],
                              'positive_annotation_id': a['id'], 'track_id': a['track_id'],
                              'bbox_xywh': a['bbox'], 'later_review_file': neg['file_name'],
                              'later_image_id': neg['id'], 'frame_gap': neg['frame_id']-pos['frame_id'],
                              'later_frame_other_gt_count': len(by_image[neg['id']]),
                              'status': 'unverified_label_index_only'})
                break
        if queue:
            break
    receipt['videos'].append({'name': name, 'video_id': vid, 'frame_count': len(ims),
                              'no_annotation_frame_count': sum(c == 0 for c in counts),
                              'annotated_track_ids': sorted({a['track_id'] for im in ims for a in by_image[im['id']]}),
                              'first_frame': ims[0]['file_name'], 'last_frame': ims[-1]['file_name'],
                              'source_review_queue': queue})
path = root / 'train_source_manual_review_queue.json'
path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
