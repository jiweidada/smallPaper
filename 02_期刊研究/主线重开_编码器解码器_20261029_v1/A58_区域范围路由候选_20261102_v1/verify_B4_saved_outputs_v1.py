"""Check saved array aggregation and GT/prediction structure, no model."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import zipfile
import numpy as np

P=Path(__file__).resolve().parent
R54=P.parent/'A54_场景尺度约束候选_20261101_v1'
def load(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rpc=load(R54/'A58_B4_interim_export_v1.json');assert rpc['exit']==0
export=json.loads(rpc['stdout'])
download=load(R54/'A58_B4_interim_download_v1.json');assert download['status']=='TRANSFERRED'
z=R54/'A57_downloads/A58_B4_interim_evidence_v1.zip'
assert z.stat().st_size==export['bytes'] and sha(z)==export['sha256']
assert not (P/'B4_interim_evidence_v1.zip').exists();shutil.copyfile(z,P/'B4_interim_evidence_v1.zip')
d=P/'B4_interim_remote_v1';d.mkdir(exist_ok=False)
with zipfile.ZipFile(z) as f:
    assert all(Path(n).name==n for n in f.namelist())
    for n in f.namelist():(d/n).write_bytes(f.read(n))
manifest=load(d/'B4_artifact_manifest.json')
assert manifest['epoch']==4 and manifest['arm']=='B' and manifest['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
for name,row in manifest['files'].items():
    if name=='epoch04.pth':
        assert not (d/name).exists()
        continue
    assert (d/name).stat().st_size==row['bytes'] and sha(d/name)==row['sha256'],name
m=load(d/'epoch04_EMA_metrics.json');assert m==export['metrics']
arrays=np.load(d/'epoch04_EMA_precision_recall.npz',allow_pickle=False)
p,rec,ious=arrays['precision'],arrays['recall'],arrays['iou_thresholds']
assert p.shape==(10,101,10,4,3) and rec.shape==(10,10,4,3)
assert np.isfinite(p).all() and np.isfinite(rec).all()
assert p.min()>=-1 and p.max()<=1 and rec.min()>=-1 and rec.max()<=1
def ap(area=0,iou=None):
    x=p[:,:,:,area,-1]
    if iou is not None:x=x[np.isclose(ious,iou)]
    x=x[x>=0]
    return float(x.mean()*100)
def ar(area=0):
    x=rec[:,:,area,-1];x=x[x>=0]
    return float(x.mean()*100)
actual=dict(AP=ap(),AP50=ap(iou=.5),AP75=ap(iou=.75),APs=ap(1),APs75=ap(1,.75),APm=ap(2),APl=ap(3),ARs=ar(1))
delta={k:actual[k]-m[k] for k in actual}
assert all(abs(v)<1e-10 for v in delta.values())
gt=load(d/'epoch04_EMA_GT.json');pred=load(d/'epoch04_EMA_predictions.json')
assert len(gt['images'])==548 and len(gt['annotations'])==38602 and gt['ignored_valid_gt']==157
assert len(pred)==m['predictions_after_ignore']==154939
small=sum(a['area']<=1024 for a in gt['annotations'])
geometry=load(P/'annotation_geometry_remote_v1/result.json')
assert small==geometry['splits']['val']['small_original_area']['objects']==26444
image_ids={x['id'] for x in gt['images']};categories={c['id'] for c in gt['categories']}
assert categories==set(range(1,11))
for x in pred:
    assert x['image_id'] in image_ids and x['category_id'] in categories
    assert all(math.isfinite(float(v)) for v in x['bbox']+[x['score']])
    assert x['bbox'][2]>0 and x['bbox'][3]>0 and 0<=x['score']<=1
flow=load(d/'epoch04_input_flow.json');assert flow['epoch']==4 and flow['optimizer_updates']==3236
assert len(flow['first_eight_actual_input_sha256'])==len(flow['first_eight_pre_forward_rng_sha256'])==8
init=load(d/'initialization.json');assert init['all_initial_tensors_identical'] and init['common_init_sha256']=='80df462029e32551a018b1f056794347d1739b908df28f52111ca4dfe54d3302'
out=dict(status='PASS_SAVED_BYTES_METRIC_ARRAYS_GT_PRED_STRUCTURE',metrics=m,recomputed_metrics=actual,
    maximum_metric_difference_pp=max(abs(v) for v in delta.values()),GT=38602,small_GT=small,predictions=len(pred),
    checkpoint_preserved_remote=manifest['files']['epoch04.pth'],new_audit_detector_forwards=0,
    new_optimizer_updates=0,extra_COCO_matching_runs=0,
    limits='Saved COCO precision/recall aggregation and structural checks only; no second detector run or independent repeat of COCO matching. B4 is not fixed12 verdict.')
with (P/'B4_saved_outputs_local_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out))
