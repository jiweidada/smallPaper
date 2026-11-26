"""Retrieve public static source files only; never import downloaded code."""
import json
from retrieve_static import ROOT, retrieve

files = ['ultralytics/data/augment.py', 'ultralytics/models/yolo/detect/val.py']
records = []
for path in files:
    records.append(retrieve('last_tag_' + path.replace('/', '_'),
                            'https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.191/' + path,
                            ROOT / 'source_code' / 'v8.4.191_2004cba021c6' / path))
(ROOT / 'last_source_receipts.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps([{k: row.get(k) for k in ['key', 'status', 'bytes', 'error']} for row in records], ensure_ascii=True))
