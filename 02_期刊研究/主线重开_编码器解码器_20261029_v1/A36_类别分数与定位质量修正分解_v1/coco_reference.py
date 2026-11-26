"""Independent pycocotools reference; no learned network/component invocation."""
from pathlib import Path
import collections,contextlib,copy,hashlib,io,json,sys,time
import numpy as np
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(D/'score_protocol_FROZEN.json');manifest=read(D/'reference_inputs_manifest.json')
for n,h in manifest.items():assert sha(D/n)==h,n
assert not (D/'coco_reference_result.json').exists()
assert sha(cfg['cocoeval_source'])==cfg['cocoeval_sha256']
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt=read(D/'filtered_GT.json');draws=np.load(D/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['draws']
images={x['id']:x for x in gt['images']};annotations=collections.defaultdict(list)
for a in gt['annotations']:annotations[a['image_id']].append(a)
references={};base_records={};calls=0;start=time.monotonic()
for arm in cfg['arms']:
    pred=read(D/(arm+'_filtered_predictions.json'));byimage=collections.defaultdict(list)
    for p in pred:byimage[p['image_id']].append(p)
    references[arm]={}
    for column in [-1,0,499,999]:
        chosen=list(range(1,13)) if column==-1 else (draws[column]+1).tolist()
        replica=dict(info={},images=[],categories=gt['categories'],annotations=[]);dets=[]
        for new_id,old_id in enumerate(chosen,1):
            replica['images'].append(dict(images[old_id],id=new_id))
            for a in annotations[old_id]:replica['annotations'].append(dict(a,id=len(replica['annotations'])+1,image_id=new_id))
            for p in byimage[old_id]:dets.append(dict(p,image_id=new_id))
        with contextlib.redirect_stdout(io.StringIO()):
            g=COCO();g.dataset=replica;g.createIndex()
            if dets:d=g.loadRes(dets)
            else:d=COCO();d.dataset=dict(info={},images=replica['images'],categories=gt['categories'],annotations=[]);d.createIndex()
            ev=COCOeval(g,d,'bbox');ev.params.imgIds=list(range(1,13));ev.params.catIds=list(range(1,11));ev.params.maxDets=[1,10,500]
            ev.evaluate();ev.accumulate()
        calls+=1
        def average(v):
            v=np.asarray(v);v=v[v>-1];return float(v.mean()) if len(v) else 0.
        precision=ev.eval['precision'];idx75=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.75))[0])
        references[arm][str(column)]=dict(AP=average(precision[:,:,:,0,2]),APsmall=average(precision[:,:,:,1,2]),
            AP50=average(precision[0,:,:,0,2]),AP75=average(precision[idx75,:,:,0,2]),APsmall75=average(precision[idx75,:,:,1,2]))
        if column==-1:
            records=[]
            for entry in ev.evalImgs:
                if entry is None:continue
                area_index=next((i for i in [0,1] if entry['aRng']==ev.params.areaRng[i]),None)
                if area_index is None:continue
                records.append(dict(image_id=entry['image_id'],category_id=entry['category_id'],area_index=area_index,
                    scores=list(entry['dtScores']),matches=np.asarray(entry['dtMatches']).tolist(),
                    ignore=np.asarray(entry['dtIgnore']).tolist(),gt_ignore=np.asarray(entry['gtIgnore']).tolist()))
            base_records[arm]=records
assert calls==8
out=dict(status='PYCOCOTOOLS_REFERENCE_COMPLETE_NO_MODEL_CALLS',references=references,base_eval_records=base_records,
         metric_engine_calls=calls,learned_component_calls=0,elapsed_seconds=time.monotonic()-start)
with (D/'coco_reference_result.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(dict(status=out['status'],metric_engine_calls=calls,learned_component_calls=0,elapsed_seconds=out['elapsed_seconds'])))
