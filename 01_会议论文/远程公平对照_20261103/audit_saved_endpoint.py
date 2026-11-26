"""Read-only evidence audit: copy a saved endpoint; no model execution or evaluation rerun."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,os,shutil,signal,traceback
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
parser=argparse.ArgumentParser()
parser.add_argument('--arm',choices=['BASE','FULL'],required=True)
parser.add_argument('--endpoint',choices=['best','terminal'],default='best')
args=parser.parse_args()
run=ROOT/'runs'/args.arm
metric_source=run/f'{args.endpoint}.metrics.json'
row=read(metric_source)
folder=ROOT/'evidence_audits'/f'{args.arm}_{args.endpoint}_epoch{row["epoch"]:04}'
folder.mkdir(parents=True,exist_ok=False)
receipt={'status':'VERIFYING_SAVED_EVIDENCE','started_remote_utc':now(),'pid':os.getpid(),
         'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
         'arm':args.arm,'endpoint':args.endpoint,'actual_epoch':row['epoch'],
         'model_inference_passes':0,'optimizer_updates':0,'evaluator_reruns':0}
def save():
    tmp=folder/'receipt.json.tmp';tmp.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,folder/'receipt.json')
def expire(signum,frame):raise TimeoutError('Evidence-only audit exceeded 900 seconds')
signal.signal(signal.SIGALRM,expire);signal.alarm(900)
save()
try:
    source_checkpoint=run/('best.pt' if args.endpoint=='best' else 'terminal_epoch0319.pt')
    mapping={'checkpoint.pt':source_checkpoint,'predictions.json':run/f'{args.endpoint}.predictions.json',
             'evaluation_arrays.npz':run/f'{args.endpoint}.evaluation_arrays.npz','metrics.json':metric_source}
    for dest,source in mapping.items():shutil.copyfile(source,folder/dest)
    assert read(metric_source)==row==read(folder/'metrics.json'),'Endpoint changed while copied; no binding accepted'
    assert sha(folder/'checkpoint.pt')==row['checkpoint_sha256']
    assert sha(folder/'predictions.json')==row['prediction_sha256']
    assert sha(folder/'evaluation_arrays.npz')==row['evaluation_arrays_sha256']
    protocol=read(ROOT/'execution_protocol.json')
    assert row['protocol_sha256']==sha(ROOT/'execution_protocol.json')
    assert row['config_sha256']==sha(ROOT/'execution_configs'/f'{args.arm}.yml')
    assert row['weight_source']=='ema.module'
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    import torch
    import numpy as np
    torch.set_num_threads(1)
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert checkpoint['arm']==args.arm and checkpoint['weight_source']=='ema.module'
    assert checkpoint['actual_loop_epoch']==row['epoch']
    assert checkpoint['last_epoch']==row['last_epoch']
    assert checkpoint['optimizer_step_attempts']==row['optimizer_step_attempts']==(row['epoch']+1)*1617
    assert checkpoint['protocol_sha256']==row['protocol_sha256']
    assert checkpoint['configuration_sha256']==row['config_sha256']
    assert checkpoint['environment_sha256']==sha(ROOT/'environment.current.json')
    assert checkpoint['ema']['updates']==row['ema_updates']
    state=checkpoint['ema']['module'];h=hashlib.sha256()
    for key,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        h.update(json.dumps([key,str(tensor.dtype),list(tensor.shape)]).encode())
        h.update(tensor.numpy().tobytes())
    assert h.hexdigest()==row['ema_tensor_sha256'],'Checkpoint EMA does not match evaluated tensor hash'
    data=read('/tmp/mpsma_fair_20261103_data/val/annotations/val.json')
    ids={x['id'] for x in data['images']};categories={x['id'] for x in data['categories']}
    assert len(ids)==548
    predictions=read(folder/'predictions.json');counts={i:0 for i in ids}
    assert len(predictions)==164400
    for pred in predictions:
        assert pred['image_id'] in ids and pred['category_id'] in categories
        assert len(pred['bbox'])==4 and all(math.isfinite(v) for v in pred['bbox']) and math.isfinite(pred['score'])
        counts[pred['image_id']]+=1
    assert set(counts.values())=={300}
    arrays=np.load(folder/'evaluation_arrays.npz')
    p=arrays['precision'];r=arrays['recall']
    assert p.shape==(10,101,10,4,3) and r.shape==(10,10,4,3)
    def mean_valid(v):
        good=v[v>-1]
        return float(np.mean(good)) if good.size else -1.0
    selected=[p[:,:,:,0,2],p[0,:,:,0,2],p[5,:,:,0,2],p[:,:,:,1,2],p[:,:,:,2,2],p[:,:,:,3,2],
              r[:,:,0,0],r[:,:,0,1],r[:,:,0,2],r[:,:,1,2],r[:,:,2,2],r[:,:,3,2]]
    names=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
    recomputed=dict(zip(names,map(mean_valid,selected)))
    differences={k:recomputed[k]-row['metrics'][k] for k in names}
    assert all(abs(v)<=1e-12 for v in differences.values()),differences
    receipt.update(status='SAVED_ENDPOINT_BINDING_VERIFIED',completed_remote_utc=now(),
                   checkpoint_sha256=row['checkpoint_sha256'],ema_tensor_sha256=h.hexdigest(),
                   prediction_sha256=row['prediction_sha256'],evaluation_arrays_sha256=row['evaluation_arrays_sha256'],
                   protocol_sha256=row['protocol_sha256'],image_count=548,prediction_count=len(predictions),
                   metrics=row['metrics'],metrics_recomputed_from_saved_arrays=recomputed,
                   maximum_absolute_array_summary_difference=max(abs(v) for v in differences.values()),
                   array_summary_absolute_tolerance=1e-12,
                   conclusion_scope='Integrity of this saved epoch only; no endpoint switching or method-effect conclusion.')
    save();print(json.dumps(receipt),flush=True)
except BaseException:
    receipt.update(status='SAVED_EVIDENCE_AUDIT_FAILED',failed_remote_utc=now(),traceback=traceback.format_exc())
    save();raise
finally:signal.alarm(0)
