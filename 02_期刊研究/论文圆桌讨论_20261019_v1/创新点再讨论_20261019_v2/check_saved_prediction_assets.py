"""Check existing output availability only, without scoring, inference, or training."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'data/07_第二数据集_20261017/独立中期评估_20261018_v1'
INFER=ROOT/'inference_v3'
receipt=json.loads((INFER/'complete.json').read_text(encoding='utf-8'))
records=[]
assert len(receipt['models'])==6
for m in receipt['models']:
    p=Path(m['predictions'])
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==m['predictions_sha256']
    with np.load(p,allow_pickle=False) as z:
        a=z['detections']; columns=z['columns'].tolist()
        ids,counts=np.unique(a[:,0],return_counts=True)
        assert len(a)==5153*300 and len(ids)==5153 and (counts==300).all()
        assert np.isfinite(a).all()
        assert columns==['image_id','x','y','w','h','score']
    records.append({'seed':m['seed'],'arm':m['arm'],'epoch_zero_based':m['epoch_zero_based'],
                    'path':str(p),'sha256':digest,'images':len(ids),'predictions':len(a),
                    'minimum_predictions_per_image':int(counts.min()),'maximum_predictions_per_image':int(counts.max()),
                    'columns':columns,'query_ids_saved':False,'in_memory_fields_finite':True})
script=ROOT/'local_inference.py'
text=script.read_text(encoding='utf-8')
assert 'score_threshold=0.' in text and "len(result['scores']) == 300" in text
result={'scope':'availability/integrity only; no new inference, model loading, AP evaluation, or geometry/ranking diagnosis',
        'source_receipt':str(INFER/'complete.json'),'source_receipt_sha256':hashlib.sha256((INFER/'complete.json').read_bytes()).hexdigest(),
        'exporter':str(script),'exporter_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
        'observed_export_settings':'single vehicle class; score_threshold=0; 300 exported boxes/frame',
        'limitations':['Saved output pool is available; no internal feature/query trace or persistent query IDs.',
                      'No comparison between score ranking and geometric coverage was computed in this check.',
                      'Ignored-region, extent-validity and one-to-one matching rules must be declared before any diagnostic.',
                      'This check concerns fixed epoch36 validation outputs, not final best_stg2 or official test.'],
        'models':records}
(HERE/'六份留存预测_可用性核查.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'files_verified':len(records),'images_per_model':5153,'predictions_per_image':300,'scoring_or_inference_run':False},ensure_ascii=False))
