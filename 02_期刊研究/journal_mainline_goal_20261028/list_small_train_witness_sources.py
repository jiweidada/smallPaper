import json
from collections import Counter
from pathlib import Path

root = Path(__file__).parent / 'smot_source_metadata_20261029'
rows = json.loads((root / 'train_rows.json').read_text(encoding='utf-8'))
data = json.loads((root / 'author_train_annotations.json').read_text(encoding='utf-8'))
file_ids = {r['data-tooltip'].split(' ')[0]: r['data-id'] for r in rows if r.get('data-tooltip', '').endswith('Compressed archive') and 'data-id' in r}
imgs = Counter(i['video_id'] for i in data['images'])
annimgs = {i['id']: i for i in data['images']}
ann = Counter(annimgs[a['image_id']]['video_id'] for a in data['annotations'])
names = {'0005','0019','0028','0042','0044'}
sources = [{'video': v['name'], 'video_id': v['id'], 'public_file_id': file_ids.get(v['name']+'.zip'), 'frames': imgs[v['id']], 'gt_appearances': ann[v['id']]} for v in data['videos'] if v['name'] in names]
(root / 'small_train_source_candidates.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(sources, ensure_ascii=False, indent=2))
