"""Verify the new B8 artifact and its fixed evaluation cohort; no model replay."""
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
    out=P/'epoch8_B_remote_v1';assert not out.exists()
    archive=D/'A60_downloads/epoch8_B_bundle_v1.zip';ready=json.loads((D/'A60_downloads/epoch8_B_ready_v1.json').read_text())
    assert sha(archive)==ready['zip_sha256']=='b0d48e798f3e1e528d6906d75ca36ee2516234943a1d91811ba1386586a8a6c7'
    assert archive.stat().st_size==ready['zip_bytes'] and ready['files']==9
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('manifest.json'));assert len(manifest)==9
        for name,m in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data=z.read(name);assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256'],name
        out.mkdir();z.extractall(out)
    assert sha(out/'protocol_FROZEN_v1.json')==sha(P/'protocol_FROZEN_v1.json')
    assert sha(out/'B/epoch08_raw_GT.json')==sha(P/'epoch4_B_remote_v1/B/epoch04_raw_GT.json')
    row=json.loads((out/'B/epoch08.json').read_text());m=json.loads((out/'B/epoch08_raw_metrics.json').read_text())
    old=json.loads((P/'epoch4_B_remote_v1/B/epoch04_raw_metrics.json').read_text())
    checkpoint=json.loads((out/'B/epoch08_checkpoint_metadata.json').read_text())
    scope=json.loads((P/'protocol_FROZEN_v1.json').read_text())
    assert m==row['metrics']==ready['metrics']
    assert checkpoint['sha256']==row['checkpoint_sha256']==ready['checkpoint_sha256']
    assert checkpoint['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
    assert checkpoint['initialization_sha256']==scope['initialization_sha256']
    assert m['eval_prior_stream_sha256']==old['eval_prior_stream_sha256']
    assert m['evaluated_GT']==38602 and m['ignored_valid_GT']==157
    gt=json.loads((out/'B/epoch08_raw_GT.json').read_text());pred=json.loads((out/'B/epoch08_raw_predictions.json').read_text())
    assert len(gt['images'])==548 and sum(a['area']<=1024 for a in gt['annotations'])==26444
    assert len(pred)==m['predictions_after_ignore']==79938
    sizes={r['id']:(r['width'],r['height']) for r in gt['images']};counts={k:0 for k in sizes}
    for v in pred:
        assert v['image_id'] in sizes and 1<=v['category_id']<=10 and 0<=v['score']<=1
        x,y,w,h=v['bbox'];iw,ih=sizes[v['image_id']]
        assert all(math.isfinite(a) for a in [x,y,w,h,v['score']]) and w>0 and h>0
        assert x>=-1e-5 and y>=-1e-5 and x+w<=iw+1e-3 and y+h<=ih+1e-3
        counts[v['image_id']]+=1
    assert max(counts.values())<=300
    with np.load(out/'B/epoch08_raw_precision_recall.npz') as arrays:
        precision=arrays['precision'];recall=arrays['recall'];thresholds=arrays['iou_thresholds']
        assert precision.shape==(10,101,10,4,3) and recall.shape==(10,10,4,3)
        def ap(area=0,iou=None):
            a=precision[:,:,:,area,-1]
            if iou is not None:a=a[np.isclose(thresholds,iou)]
            a=a[a>=0];return float(a.mean()*100)
        a=recall[:,:,1,-1];a=a[a>=0]
        derived=dict(AP=ap(),AP50=ap(iou=.5),AP75=ap(iou=.75),APs=ap(1),APs75=ap(1,.75),APm=ap(2),APl=ap(3),ARs=float(a.mean()*100))
    differences={k:derived[k]-m[k] for k in derived};assert all(abs(v)<1e-10 for v in differences.values())
    events=[json.loads(x) for x in (out/'B/epoch8_actual_events_prefix.jsonl').read_text().splitlines()]
    assert len(events)==111104
    starts=[x for x in events if x['event']=='forward_start'];posts=[x for x in events if x['event']=='forward_complete']
    updates=[x for x in events if x['event']=='optimizer_returned']
    assert len(starts)==len(posts)==52316 and len(updates)==6472
    assert [x['count'] for x in starts]==[x['count'] for x in posts]==list(range(1,52317))
    assert sum(x['phase']=='train' for x in starts)==51768 and sum(x['phase']=='eval' for x in starts)==548
    for e in range(1,9):
        assert [x['position'] for x in starts if x['epoch']==e and x['phase']=='train']==list(range(6471))
        assert [x['group_images'] for x in updates if x['epoch']==e]==[8]*808+[7]
    assert all(math.isfinite(x[k]) for x in updates for k in ['mean_loss','gradient_norm','time_MLP_gradient_L1'])
    result=dict(status='PASS_EXISTING_B8_ARTIFACTS_FIXED_COHORT_AND_SAVED_METRICS',verified_payloads=9,
        zip_sha256=sha(archive),original_train_forwards=51768,original_validation_forwards=548,original_optimizer_updates=6472,
        completed_training_epochs=8,metrics=m,saved_precision_recall_metric_differences=differences,
        B8_minus_B4_pp={k:m[k]-old[k] for k in derived},B4_B8_GT_identical=True,B4_B8_eval_prior_stream_identical=True,
        checkpoint_sha256=checkpoint['sha256'],original_journal_events=111104,
        B12_learning_gate_completed=False,candidate_efficacy_tested=False,
        new_detector_forwards=0,new_optimizer_updates=0,new_COCO_evaluations=0,checkpoint_loads=0,
        scope='New B8 saved-artifact integrity, cohort/priors, counts and metric aggregation; not independent COCO rematching, convergence, fixed12 gate or Q improvement.')
    with (P/'B_epoch8_bundle_local_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],AP=m['AP'],APs=m['APs'],B8_minus_B4_AP=result['B8_minus_B4_pp']['AP'],
        B8_minus_B4_APs=result['B8_minus_B4_pp']['APs'],max_metric_difference=max(abs(v) for v in differences.values()),
        fixed_GT=True,fixed_eval_priors=True,original_forwards=52316,original_updates=6472,new_forwards=0,new_updates=0,new_COCO_evaluations=0)))

if __name__=='__main__':main()
