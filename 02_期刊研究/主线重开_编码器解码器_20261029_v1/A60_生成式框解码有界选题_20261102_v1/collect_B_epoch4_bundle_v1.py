"""Verify the first existing AP artifact; no new detector or COCO evaluation."""
from pathlib import Path
import hashlib
import json
import math
import zipfile
import numpy as np

P=Path(__file__).resolve().parent
D=P.parent/'A54_场景尺度约束候选_20261101_v1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    out=P/'epoch4_B_remote_v1';assert not out.exists()
    archive=D/'A60_downloads/epoch4_B_bundle_v1.zip'
    ready=json.loads((D/'A60_downloads/epoch4_B_ready_v1.json').read_text())
    assert sha(archive)==ready['zip_sha256']=='184fac6efe3ec15aa85f9d19e632735cee2278bfc1117b0ec1b0f38592fa936f'
    assert archive.stat().st_size==ready['zip_bytes'] and ready['files']==18
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('manifest.json'));assert len(manifest)==18
        for name,m in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data=z.read(name);assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256'],name
        out.mkdir();z.extractall(out)
    assert sha(out/'protocol_FROZEN_v1.json')==sha(P/'protocol_FROZEN_v1.json')
    scope=json.loads((P/'protocol_FROZEN_v1.json').read_text())
    row=json.loads((out/'B/epoch04.json').read_text())
    metrics=json.loads((out/'B/epoch04_raw_metrics.json').read_text())
    gt=json.loads((out/'B/epoch04_raw_GT.json').read_text())
    predictions=json.loads((out/'B/epoch04_raw_predictions.json').read_text())
    checkpoint=json.loads((out/'B/epoch04_checkpoint_metadata.json').read_text())
    init=json.loads((out/'B/initialization.json').read_text())
    run=json.loads((out/'B/run_once.json').read_text())
    assert metrics==row['metrics']==ready['metrics']
    assert checkpoint['sha256']==row['checkpoint_sha256']==ready['checkpoint_sha256']
    assert checkpoint['protocol_sha256']==run['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
    assert checkpoint['initialization_sha256']==init['initialization_sha256']==scope['initialization_sha256']
    assert not checkpoint['checkpoint_downloaded'] and not checkpoint['checkpoint_loaded_for_audit']
    assert len(gt['images'])==548 and len(gt['annotations'])==38602 and gt['ignored_valid_gt']==157
    assert sum(a['area']<=1024 for a in gt['annotations'])==26444
    assert len(predictions)==metrics['predictions_after_ignore']==78414
    sizes={r['id']:(r['width'],r['height']) for r in gt['images']};per_image={i:0 for i in sizes}
    for prediction in predictions:
        assert 1<=prediction['category_id']<=10 and prediction['image_id'] in sizes
        assert math.isfinite(prediction['score']) and 0<=prediction['score']<=1
        x,y,w,h=prediction['bbox'];iw,ih=sizes[prediction['image_id']]
        assert all(math.isfinite(v) for v in [x,y,w,h]) and w>0 and h>0
        assert x>=-1e-5 and y>=-1e-5 and x+w<=iw+1e-3 and y+h<=ih+1e-3
        per_image[prediction['image_id']]+=1
    assert max(per_image.values())<=300
    with np.load(out/'B/epoch04_raw_precision_recall.npz') as arrays:
        precision=arrays['precision'];recall=arrays['recall'];thresholds=arrays['iou_thresholds']
        assert precision.shape==(10,101,10,4,3) and recall.shape==(10,10,4,3)
        def ap(area=0,iou=None):
            a=precision[:,:,:,area,-1]
            if iou is not None:a=a[np.isclose(thresholds,iou)]
            a=a[a>=0];return float(a.mean()*100)
        a=recall[:,:,1,-1];a=a[a>=0]
        derived=dict(AP=ap(),AP50=ap(iou=.5),AP75=ap(iou=.75),APs=ap(1),APs75=ap(1,.75),APm=ap(2),APl=ap(3),ARs=float(a.mean()*100))
    differences={k:derived[k]-metrics[k] for k in derived}
    assert all(abs(v)<1e-10 for v in differences.values()),differences
    events=[json.loads(line) for line in (out/'B/epoch4_actual_events_prefix.jsonl').read_text().splitlines()]
    assert len(events)==55552
    starts=[x for x in events if x['event']=='forward_start']
    posts=[x for x in events if x['event']=='forward_complete']
    updates=[x for x in events if x['event']=='optimizer_returned']
    assert len(starts)==len(posts)==26158 and len(updates)==3236
    assert [x['count'] for x in starts]==[x['count'] for x in posts]==list(range(1,26159))
    assert sum(x['phase']=='train' for x in starts)==25884 and sum(x['phase']=='eval' for x in starts)==274
    for e in range(1,5):
        s=[x for x in starts if x['epoch']==e and x['phase']=='train']
        u=[x for x in updates if x['epoch']==e]
        assert [x['position'] for x in s]==list(range(6471))
        assert [x['group_images'] for x in u]==[8]*808+[7]
    assert all(math.isfinite(x[k]) for x in updates for k in ['mean_loss','gradient_norm','time_MLP_gradient_L1'])
    result=dict(status='PASS_EXISTING_B4_ARTIFACTS_AND_SAVED_PRECISION_RECALL_METRICS',
        verified_payloads=18,zip_sha256=sha(archive),original_train_forwards=25884,original_val_forwards=274,
        original_optimizer_updates=3236,completed_training_epochs=4,GT_images=548,scored_GT=38602,small_GT=26444,
        predictions=78414,maximum_predictions_per_image=max(per_image.values()),metrics=metrics,
        saved_precision_recall_metric_differences=differences,checkpoint_sha256=checkpoint['sha256'],
        original_journal_events=55552,formal_M_Q_started=False,candidate_efficacy_tested=False,B12_gate_completed=False,
        new_detector_forwards=0,new_optimizer_updates=0,new_COCO_evaluations=0,checkpoint_loads=0,
        scope='File/hash/count/box-validity and saved precision-recall aggregation verification. No CNN replay or independent COCO rematching; B4 only, not fixed12 or candidate improvement.')
    with (P/'B_epoch4_bundle_local_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],verified_payloads=18,AP=metrics['AP'],APs=metrics['APs'],
        maximum_metric_difference=max(abs(v) for v in differences.values()),original_forwards=26158,original_updates=3236,
        new_forwards=0,new_updates=0,new_COCO_evaluations=0,B12_gate_completed=False,candidate_efficacy_tested=False)))

if __name__=='__main__':main()
