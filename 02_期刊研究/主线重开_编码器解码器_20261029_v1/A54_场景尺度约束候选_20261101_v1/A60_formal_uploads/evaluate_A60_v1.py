"""Shared single-step decoder evaluation; COCO-style, not VisDrone official."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.ops import batched_nms
sys.path.insert(0,'/root/uav_a47_20261101_v1/pydeps')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from runtime_A60_v1 import load_item,save_json
from box_progress_conditioning_v1 import sample_prior,single_step_inputs

def union_overlap_fraction(box,regions):
    x0,y0,x1,y1=[float(v) for v in box]
    area=max(0,x1-x0)*max(0,y1-y0)
    if area<=0:return 0.0
    rs=[]
    for r in regions:
        a,b,c,d=max(x0,r[0]),max(y0,r[1]),min(x1,r[2]),min(y1,r[3])
        if c>a and d>b:rs.append((a,b,c,d))
    xs=sorted({v for r in rs for v in (r[0],r[2])});covered=0.
    for a,c in zip(xs[:-1],xs[1:]):
        spans=sorted((r[1],r[3]) for r in rs if r[0]<c and r[2]>a)
        end=None;height=0.
        for lo,hi in spans:
            if end is None:height+=hi-lo;end=hi
            elif hi>end:height+=hi-max(lo,end);end=hi
        covered+=(c-a)*height
    return covered/area

def ground_truth(rows):
    images=[];annotations=[];ignored=0
    for r in rows:
        images.append(dict(id=r['id'],width=r['width'],height=r['height'],file_name=r['file']))
        for v in r['valid']:
            if union_overlap_fraction(v[:4],r['ignore'])>=.5:ignored+=1;continue
            x0,y0,x1,y1,cls=v;w=x1-x0;h=y1-y0
            annotations.append(dict(id=len(annotations)+1,image_id=r['id'],category_id=int(cls)+1,
                bbox=[x0,y0,w,h],area=w*h,iscrowd=0))
    return dict(images=images,annotations=annotations,categories=[dict(id=i,name=str(i)) for i in range(1,11)],
        info=dict(scope='COCO-style, class0 union IoA >= .5 filtered; not official VisDrone score'),ignored_valid_gt=ignored)

class ValImages(Dataset):
    def __init__(self,split):self.split=split;self.rows=split['rows']
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];p=Path(self.split['image_dir'])/row['file'];st=p.stat()
        assert st.st_size==row['image_bytes'] and st.st_mtime_ns==row['image_mtime_ns']
        image,target=load_item(row['file'],self.split['image_dir'],self.split['annotation_dir'])
        assert target['annotation_sha256']==row['annotation_sha256'] and len(target['labels'])==row['valid_GT']
        return image,index

def collate_val(items):return torch.stack([x[0] for x in items]),[x[1] for x in items]

def evaluation_prior(names):
    # Deterministic per filename, independent of arm, epoch, GT and global RNG.
    values=[]
    for name in names:
        seed=int.from_bytes(hashlib.sha256(('A60_eval_20261102|'+name).encode()).digest()[:8],'little')%(2**63-1)
        gen=torch.Generator(device='cpu');gen.manual_seed(seed)
        p=sample_prior(1,300,'cpu',generator=gen);p[...,2:].clamp_min_(1e-4);values.append(p)
    return torch.cat(values)

def postprocess(logits,boxes,row):
    assert logits.shape==(300,10) and boxes.shape==(300,4)
    scores,index=logits.sigmoid().flatten().topk(300,sorted=True)
    labels=index%10;b=boxes[index//10]
    keep=batched_nms(b,scores,labels,.6);b=b[keep].clone();scores=scores[keep];labels=labels[keep]
    b[:,0::2]=(b[:,0::2]/640*row['width']).clamp(0,row['width'])
    b[:,1::2]=(b[:,1::2]/640*row['height']).clamp(0,row['height'])
    predictions=[];ignored=0
    for box,score,label in zip(b.cpu().tolist(),scores.cpu().tolist(),labels.cpu().tolist()):
        x0,y0,x1,y1=box
        if x1<=x0 or y1<=y0:continue
        if union_overlap_fraction(box,row['ignore'])>=.5:ignored+=1;continue
        predictions.append(dict(image_id=row['id'],category_id=int(label)+1,
            bbox=[x0,y0,x1-x0,y1-y0],score=score))
    return predictions,ignored

def metric_from_predictions(gt,predictions,out,prefix):
    save_json(out/(prefix+'_GT.json'),gt);save_json(out/(prefix+'_predictions.json'),predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO();coco.dataset=gt;coco.createIndex()
        if predictions:dt=coco.loadRes(predictions)
        else:
            dt=COCO();dt.dataset=dict(images=gt['images'],annotations=[],categories=gt['categories']);dt.createIndex()
        ev=COCOeval(coco,dt,'bbox');ev.params.maxDets=[1,10,300];ev.evaluate();ev.accumulate()
    p=ev.eval['precision'];rec=ev.eval['recall']
    def ap(area=0,iou=None):
        z=p[:,:,:,area,-1]
        if iou is not None:z=z[np.isclose(ev.params.iouThrs,iou)]
        z=z[z>=0];return float(z.mean()*100) if len(z) else None
    z=rec[:,:,1,-1];z=z[z>=0]
    result=dict(AP=ap(),AP50=ap(iou=.5),AP75=ap(iou=.75),APs=ap(1),APs75=ap(1,.75),APm=ap(2),APl=ap(3),
        ARs=float(z.mean()*100) if len(z) else None,max_detections=300,evaluated_GT=len(gt['annotations']),
        ignored_valid_GT=gt['ignored_valid_gt'],predictions_after_ignore=len(predictions),
        area_convention='clipped original-image GT pixel area; small<=1024',
        scope='COCO-style; class0 union IoA filtering; not official VisDrone score')
    np.savez_compressed(out/(prefix+'_precision_recall.npz'),precision=p,recall=rec,iou_thresholds=ev.params.iouThrs)
    return result

@torch.inference_mode()
def evaluate(model,split,out,prefix):
    model.eval();start=time.monotonic();predictions=[];ignored=0
    gt=ground_truth(split['rows'])
    assert len(gt['annotations'])==38602 and gt['ignored_valid_gt']==157
    assert sum(a['area']<=1024 for a in gt['annotations'])==26444
    gen=torch.Generator();gen.manual_seed(20261102)
    loader=DataLoader(ValImages(split),batch_size=2,shuffle=False,num_workers=2,
        collate_fn=collate_val,pin_memory=True,generator=gen)
    prior_stream=hashlib.sha256()
    for images,indices in loader:
        rows=[split['rows'][i] for i in indices]
        prior=evaluation_prior([r['file'] for r in rows]);prior_stream.update(prior.numpy().tobytes())
        b,t=single_step_inputs(prior)
        outputs=model(images.cuda(non_blocking=True),b.cuda(),t.cuda())
        assert torch.isfinite(outputs['pred_logits']).all() and torch.isfinite(outputs['pred_boxes']).all()
        for logits,boxes,row in zip(outputs['pred_logits'],outputs['pred_boxes'],rows):
            p,n=postprocess(logits,boxes,row);predictions.extend(p);ignored+=n
    metrics=metric_from_predictions(gt,predictions,out,prefix)
    metrics.update(ignore_filtered_predictions=ignored,validation_seconds=time.monotonic()-start,
        eval_prior_stream_sha256=prior_stream.hexdigest(),inference_head_calls_per_batch=1,
        inference_queries=300,model_state='raw model, no EMA, fixed epoch')
    save_json(out/(prefix+'_metrics.json'),metrics)
    return metrics

@torch.inference_mode()
def latency(model,split):
    model.eval();row=split['rows'][0]
    image,_=load_item(row['file'],split['image_dir'],split['annotation_dir'])
    image=image[None].cuda();b,t=single_step_inputs(evaluation_prior([row['file']]));b=b.cuda();t=t.cuda()
    for _ in range(10):model(image,b,t)
    torch.cuda.synchronize();samples=[]
    for _ in range(30):
        a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
        a.record();model(image,b,t);z.record();z.synchronize();samples.append(a.elapsed_time(z))
    return dict(median_ms=float(np.median(samples)),mean_ms=float(np.mean(samples)),samples_ms=samples,
        warmup=10,repeats=30,batch=1,queries=300,head_calls=1,
        scope='full eager CUDA model forward; BF16 encoder/FP32 head, TF32 off; excludes image decoding and postprocess; not onboard latency')
