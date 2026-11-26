"""Read original TinyPerson annotations and image availability, no mutation."""
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

root = Path('/24085403014/DEIM/tiny_set')
report = {'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'root': str(root),
          'annotations': [], 'root_directories': [], 'download_modules': {}}
for path in root.iterdir():
    if path.is_dir():
        report['root_directories'].append({'path': str(path), 'children': [p.name for p in path.iterdir()][:20]})
names = ['annotations/tiny_set_train.json', 'annotations/tiny_set_test.json',
         'annotations/tiny_set_train_with_dense.json', 'annotations/tiny_set_test_with_dense.json',
         'annotations/task/tiny_set_test_all.json',
         'erase_with_uncertain_dataset/annotations/task/tiny_set_train_all.json',
         'erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json']
for name in names:
    path = root / name
    if not path.is_file():
        report['annotations'].append({'path': str(path), 'exists': False})
        continue
    raw = path.read_bytes(); data = json.loads(raw)
    images = {im['id']: im for im in data['images']}
    split = 'test' if 'test' in path.name else 'train'
    image_root = root / ('erase_with_uncertain_dataset' if name.startswith('erase_') else '') / split
    missing = [im['file_name'] for im in images.values() if not (image_root / im['file_name']).is_file()]
    anns = data['annotations']
    normal = [ann for ann in anns if not any(ann.get(key, False) for key in ['ignore', 'uncertain', 'in_dense_image', 'logo'])]
    sizes = [(ann['bbox'][2] * ann['bbox'][3]) ** .5 for ann in normal]
    input_sizes = [(ann['bbox'][2] * ann['bbox'][3] * 640 * 640 / (images[ann['image_id']]['width'] * images[ann['image_id']]['height'])) ** .5 for ann in normal]
    stems = [Path(im['file_name']).stem for im in images.values()]
    groups = Counter(stem.rsplit('_I', 1)[0] if '_I' in stem else stem for stem in stems)
    report['annotations'].append({'path': str(path), 'exists': True, 'bytes': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest(), 'image_root': str(image_root),
        'images': len(images), 'annotations': len(anns), 'normal': len(normal),
        'categories': data['categories'], 'missing_images': missing,
        'flag_counts': {key: dict(Counter(str(ann.get(key, '<missing>')) for ann in anns))
                        for key in ['ignore', 'uncertain', 'in_dense_image', 'logo', 'iscrowd']},
        'normal_size_median_original': statistics.median(sizes) if sizes else None,
        'normal_size_median_stretch640': statistics.median(input_sizes) if input_sizes else None,
        'normal_size_bins_stretch640': {str(high): sum(size < high for size in input_sizes) for high in [1, 4, 8, 16, 32]},
        'video_groups': dict(groups), 'sample_images': list(images.values())[:2], 'sample_annotations': anns[:2]})
for name in ['requests', 'gdown', 'pycocotools', 'PIL', 'huggingface_hub']:
    report['download_modules'][name] = importlib.util.find_spec(name) is not None
opendata = Path('/24085403014/DEIM/datasets/tiny_person/OpenDataLab___TinyPerson')
report['opendata_contents'] = []
if opendata.is_dir():
    for current in [opendata, *[p for p in opendata.iterdir() if p.is_dir()]]:
        report['opendata_contents'].append({'path': str(current), 'files': [{'name': p.name, 'bytes': p.stat().st_size} for p in current.iterdir() if p.is_file()][:20]})
print(json.dumps(report, ensure_ascii=False))
