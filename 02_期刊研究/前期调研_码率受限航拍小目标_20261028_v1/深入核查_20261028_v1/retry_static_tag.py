from retrieve_static import ROOT, retrieve
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
files=['ultralytics/utils/ops.py','ultralytics/engine/trainer.py','ultralytics/engine/predictor.py','ultralytics/engine/validator.py','ultralytics/nn/autobackend.py','ultralytics/data/augment.py','ultralytics/models/yolo/detect/val.py']
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
 jobs=[pool.submit(retrieve,'tag_retry_'+p.replace('/','_'),'https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.191/'+p,ROOT/'source_code'/'v8.4.191_2004cba021c6'/p) for p in files]
 for job in as_completed(jobs):
  r=job.result();rows.append(r)
  (ROOT/'tag_retry_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps({'key':r['key'],'status':r['status'],'bytes':r.get('bytes'),'error':r.get('error')},ensure_ascii=True),flush=True)
