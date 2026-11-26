"""Download fixed-version source text for inspection; never execute it."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys, json
root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'深入核查_20261028_v1'))
from retrieve_static import retrieve
files=['ultralytics/data/augment.py','ultralytics/data/base.py','ultralytics/data/dataset.py']
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    pending=[pool.submit(retrieve,'label_filter_'+p.replace('/','_'),
                         'https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.191/'+p,
                         root/'source_code'/'v8.4.191_2004cba021c6'/p) for p in files]
    for future in as_completed(pending):
        row=future.result();rows.append(row)
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','error']},ensure_ascii=True),flush=True)
(root/'filter_source_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
