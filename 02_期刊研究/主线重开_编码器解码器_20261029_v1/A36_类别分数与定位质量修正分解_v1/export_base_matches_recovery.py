"""Serialization repair: two base matching passes, no metric re-accumulation or network calls."""
from pathlib import Path
import contextlib,hashlib,io,json
import numpy as np
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(D/'score_protocol_FROZEN.json');manifest=read(D/'reference_inputs_manifest.json')
for n,h in manifest.items():assert sha(D/n)==h,n
assert sha(cfg['cocoeval_source'])==cfg['cocoeval_sha256']
assert (D/'coco_reference_result.json').exists() and not (D/'base_matches_recovery_result.json').exists()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt=read(D/'filtered_GT.json');base_records={};calls=0
for arm in cfg['arms']:
    pred=read(D/(arm+'_filtered_predictions.json'))
    with contextlib.redirect_stdout(io.StringIO()):
        g=COCO();g.dataset=gt;g.createIndex();d=g.loadRes(pred)
        ev=COCOeval(g,d,'bbox');ev.params.imgIds=list(range(1,13));ev.params.catIds=list(range(1,11));ev.params.maxDets=[1,10,500]
        ev.evaluate()
    calls+=1;records=[]
    for e in ev.evalImgs:
        if e is None:continue
        ai=next((i for i in [0,1] if e['aRng']==ev.params.areaRng[i]),None)
        if ai is None:continue
        records.append(dict(image_id=int(e['image_id']),category_id=int(e['category_id']),area_index=int(ai),
            scores=np.asarray(e['dtScores'],dtype=np.float64).tolist(),matches=np.asarray(e['dtMatches']).tolist(),
            ignore=np.asarray(e['dtIgnore']).tolist(),gt_ignore=np.asarray(e['gtIgnore']).tolist()))
    base_records[arm]=records
assert calls==2
out=dict(status='BASE_MATCH_SERIALIZATION_RECOVERY_COMPLETE',base_eval_records=base_records,
         matching_only_metric_calls=calls,metric_accumulate_calls=0,learned_component_calls=0)
with (D/'base_matches_recovery_result.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(dict(status=out['status'],matching_only_metric_calls=2,learned_component_calls=0)))
