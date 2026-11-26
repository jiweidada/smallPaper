"""CPU-only metric recalculation from immutable fixed12 prediction files."""
import argparse,contextlib,hashlib,io,json,os,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'postprocess_ignore_strata'
sys.path.insert(0,str(OUT/'deps'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['B','M','Q'],required=True);arm=ap.parse_args().arm
    protocol_path=OUT/'protocol_FIXED_v1.json';protocol=read(protocol_path)
    assert sha(protocol_path)=='93935746481b113fcef6e3fa106ffcb3c6364072b8b308a5c14c2d202a609454'
    assert sha(ROOT/'protocol_FROZEN.json')==protocol['primary_training_protocol_sha256']
    for p,expected in protocol['source_files_sha256'].items():assert sha(Path(p))==expected
    folder=ROOT/('completed_'+arm+'_v1')/('arm_'+arm)
    verified=read(ROOT/(arm+'_completed_local_verification.json'))
    assert verified['status']=='PASS'
    manifest=read(folder.parent/'bundle_manifest.json')
    for n in ['result.json','epoch12_EMA_GT.json','epoch12_EMA_predictions.json','epoch12_EMA_metrics.json','epoch12_EMA_precision_recall.npz']:
        assert sha(folder/n)==manifest['files']['arm_'+arm+'/'+n]['sha256']
    original=read(folder/'result.json')
    assert original['epochs']==12 and original['optimizer_updates']==9708
    assert original['protocol_sha256']==protocol['primary_training_protocol_sha256']
    gt=read(folder/'epoch12_EMA_GT.json');pred=read(folder/'epoch12_EMA_predictions.json')
    cached=read(folder/'epoch12_EMA_metrics.json')
    target=OUT/('arm_'+arm);target.mkdir(exist_ok=False)
    save(target/'started.json',{'pid':os.getpid(),'scope':'CPU saved-prediction metric recalculation, not detector inference','script_sha256':sha(Path(__file__))})
    calls=0;t0=time.monotonic();rows={}
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO();coco.dataset=gt;coco.createIndex();dt=coco.loadRes(pred)
    def evaluate_group(name,ids):
        global calls
        assert calls<3
        prior=0
        for other in ['B','M','Q']:
            status=OUT/('arm_'+other)/'progress.json'
            if other!=arm and status.exists():prior+=read(status)['COCO_evaluate_attempts']
        assert prior+calls<9
        calls+=1
        save(target/'progress.json',{'status':'EVALUATING','group':name,'COCO_evaluate_attempts':calls,'model_forwards':0,'optimizer_updates':0,'pid':os.getpid()})
        print(json.dumps({'status':'CPU_COCO_EVALUATING','arm':arm,'group':name,'images':len(ids),'COCO_evaluate_attempts':calls}),flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            ev=COCOeval(coco,dt,'bbox');ev.params.maxDets=[1,10,300];ev.params.imgIds=sorted(ids)
            ev.evaluate();ev.accumulate()
        p=ev.eval['precision'];rec=ev.eval['recall']
        def ap(area=0,iou=None):
            z=p[:,:,:,area,-1]
            if iou is not None:z=z[np.isclose(ev.params.iouThrs,iou)]
            z=z[z>=0];return float(z.mean()*100) if len(z) else None
        z=rec[:,:,1,-1];z=z[z>=0]
        metrics={'AP':ap(),'AP50':ap(iou=.5),'AP75':ap(iou=.75),'APs':ap(1),'APs75':ap(1,.75),'APm':ap(2),'APl':ap(3),'ARs':float(z.mean()*100) if len(z) else None}
        allowed=set(ids);anns=[a for a in gt['annotations'] if a['image_id'] in allowed]
        result={'images':len(ids),'evaluated_GT':len(anns),'evaluated_small_GT_area_le1024':sum(a['area']<=1024 for a in anns),
                'prediction_class_pair_entries':sum(a['image_id'] in allowed for a in pred),'metrics_percent':metrics}
        np.savez_compressed(target/(name+'_precision_recall.npz'),precision=p,recall=rec,iou_thresholds=ev.params.iouThrs)
        save(target/(name+'_metrics.json'),result)
        if name=='full548':
            diffs={k:metrics[k]-cached[k] for k in protocol['metrics']}
            assert all(abs(x)<=1e-8 for x in diffs.values()),diffs
            with np.load(folder/'epoch12_EMA_precision_recall.npz') as reference:
                assert p.shape==reference['precision'].shape and rec.shape==reference['recall'].shape
                pdiff=float(np.max(np.abs(p-reference['precision'])))
                rdiff=float(np.max(np.abs(rec-reference['recall'])))
                assert pdiff<=1e-12 and rdiff<=1e-12,(pdiff,rdiff)
            result['cached_metric_deltas_pp']=diffs
            result['cached_precision_max_abs_difference']=pdiff
            result['cached_recall_max_abs_difference']=rdiff
            save(target/(name+'_metrics.json'),result)
        return result
    try:
        rows['full548']=evaluate_group('full548',[im['id'] for im in gt['images']])
        for name,ids in protocol['groups_image_ids'].items():
            rows[name]=evaluate_group(name,ids)
            for k,v in protocol['group_counts'][name].items():assert rows[name][k]==v
        result={'status':'COMPLETED','arm':arm,'endpoint':'fixedepoch12EMA saved predictions','groups':rows,
                'COCO_evaluate_attempts':calls,'model_forwards':0,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-t0,
                'protocol_sha256':sha(protocol_path),'script_sha256':sha(Path(__file__)),
                'numpy_version':np.__version__,'scope':'Secondary descriptive analysis only; source prediction metrics recalculated, detector not rerun. No primary gate change or failed-candidate subgroup rescue.'}
        save(target/'result.json',result)
        save(target/'progress.json',{'status':'COMPLETED','COCO_evaluate_attempts':calls,'model_forwards':0,'optimizer_updates':0,'pid':os.getpid()})
        print(json.dumps({'status':'COMPLETED','arm':arm,'groups':{k:v['metrics_percent'] for k,v in rows.items()},
                          'full_cached_precision_max_diff':rows['full548']['cached_precision_max_abs_difference'],
                          'full_cached_recall_max_diff':rows['full548']['cached_recall_max_abs_difference'],
                          'COCO_evaluate_attempts':calls,'model_forwards':0,'optimizer_updates':0}),flush=True)
    except BaseException as exc:
        save(target/'failure.json',{'status':'FAILED_NO_AUTO_RETRY','error':repr(exc),'traceback':traceback.format_exc(),'COCO_evaluate_attempts':calls,'model_forwards':0,'optimizer_updates':0})
        raise
