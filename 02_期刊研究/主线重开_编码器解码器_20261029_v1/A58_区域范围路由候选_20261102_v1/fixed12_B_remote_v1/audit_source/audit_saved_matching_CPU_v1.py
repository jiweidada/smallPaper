"""One CPU-only COCO matching from saved fixed12 predictions per arm."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import signal
import sys
import time
import traceback

ROOT=Path('/root/uav_a58_20261102_v1/pilot_v1')
INPUTS=Path('/root/uav_a54_20261101_v1/A58_uploads')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def save_new(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def main(arm):
    scope=json.loads((INPUTS/'fixed12_audit_scope_FROZEN_v1.json').read_text())
    assert sha(Path(__file__))==scope['CPU_matching_script_sha256']
    assert sha(ROOT/'protocol_FROZEN_v1.json')==scope['formal_protocol_sha256']
    supervisor=json.loads((ROOT/'supervisor_progress.json').read_text())
    assert arm in supervisor['completed'], 'Original supervisor has not confirmed this arm exited successfully'
    d=ROOT/('arm_'+arm)
    result=json.loads((d/'result.json').read_text())
    assert result['status']=='COMPLETED' and result['epochs']==12 and result['optimizer_updates']==9708
    assert result['full_model_forwards']==10159 and result['protocol_sha256']==scope['formal_protocol_sha256']
    assert not Path('/proc/'+str(result['pid'])).exists(), 'Original arm process still exists; inspect same handle'
    out=ROOT/('saved_CPU_matching_'+arm+'_v1');out.mkdir(exist_ok=False)
    save_new(out/'run_once.json',dict(arm=arm,model_constructors=0,model_forwards=0,optimizer_updates=0,
                                    maximum_CPU_matching_calls=1,original_worker_pid=result['pid']))
    start=time.monotonic();calls=0
    def timeout(signum,frame):raise TimeoutError('CPU saved matching exceeded frozen 300s; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    try:
        sys.path.insert(0,'/root/uav_a47_20261101_v1/pydeps')
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        assert 'torch' not in sys.modules
        gt=json.loads((d/'epoch12_EMA_GT.json').read_text())
        pred=json.loads((d/'epoch12_EMA_predictions.json').read_text())
        assert len(gt['images'])==548 and len(gt['annotations'])==38602 and gt['ignored_valid_gt']==157
        assert len(pred)==result['endpoint_metrics']['predictions_after_ignore']
        with contextlib.redirect_stdout(io.StringIO()):
            coco=COCO();coco.dataset=gt;coco.createIndex()
            if pred:dt=coco.loadRes(pred)
            else:
                dt=COCO();dt.dataset=dict(images=gt['images'],annotations=[],categories=gt['categories']);dt.createIndex()
            ev=COCOeval(coco,dt,'bbox');ev.params.maxDets=[1,10,300]
            calls+=1;ev.evaluate();ev.accumulate()
        p,rec=ev.eval['precision'],ev.eval['recall']
        def ap(area=0,iou=None):
            v=p[:,:,:,area,-1]
            if iou is not None:v=v[np.isclose(ev.params.iouThrs,iou)]
            v=v[v>=0];return float(v.mean()*100) if len(v) else None
        v=rec[:,:,1,-1];v=v[v>=0]
        metrics=dict(AP=ap(),AP50=ap(iou=.5),AP75=ap(iou=.75),APs=ap(1),APs75=ap(1,.75),APm=ap(2),APl=ap(3),ARs=float(v.mean()*100))
        differences={k:metrics[k]-result['endpoint_metrics'][k] for k in metrics}
        assert all(abs(v)<1e-10 for v in differences.values()),differences
        native=np.load(d/'epoch12_EMA_precision_recall.npz',allow_pickle=False)
        pdelta=float(np.max(np.abs(p-native['precision'])))
        rdelta=float(np.max(np.abs(rec-native['recall'])))
        assert pdelta<1e-10 and rdelta<1e-10,(pdelta,rdelta)
        np.savez_compressed(out/'CPU_matching_precision_recall.npz',precision=p,recall=rec,iou_thresholds=ev.params.iouThrs)
        assert 'torch' not in sys.modules
        result_out=dict(status='PASS_SAVED_PREDICTION_CPU_MATCHING',arm=arm,epoch=12,metrics=metrics,
            metric_differences_pp=differences,maximum_precision_difference=pdelta,maximum_recall_difference=rdelta,
            saved_CPU_matching_calls=calls,model_constructors=0,model_forwards=0,optimizer_updates=0,
            monotonic_seconds=time.monotonic()-start,torch_imported=False,
            input_hashes={n:sha(d/n) for n in ['epoch12_EMA_GT.json','epoch12_EMA_predictions.json','epoch12_EMA_precision_recall.npz','epoch12_EMA_metrics.json']},
            scope_sha256=sha(INPUTS/'fixed12_audit_scope_FROZEN_v1.json'),script_sha256=sha(Path(__file__)),
            limits='Independent CPU invocation of same pinned pycocotools on saved predictions; not an independent detector run or another seed')
        save_new(out/'result.json',result_out);save_new(out/'exit.json',dict(exit=0))
        signal.alarm(0);print(json.dumps(result_out),flush=True)
    except BaseException:
        signal.alarm(0)
        save_new(out/'failure.json',dict(status='FAILED_CPU_AUDIT_NO_RETRY',arm=arm,saved_CPU_matching_calls=calls,
            model_forwards=0,optimizer_updates=0,monotonic_seconds=time.monotonic()-start,traceback=traceback.format_exc()))
        save_new(out/'exit.json',dict(exit=1))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B','M','Q'],required=True)
    main(parser.parse_args().arm)
