"""Common COCO-style ignore convention and standard flat top300 DETR decode."""
from pathlib import Path
import contextlib
import copy
import io
import time
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,'/root/uav_a47_20261101_v1/pydeps')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from common_A57 import ROOT,collate,save_json,assert_output,digest,load


def union_overlap_fraction(box,regions):
    x0,y0,x1,y1=[float(v) for v in box]
    area=max(0,x1-x0)*max(0,y1-y0)
    if area<=0:return 0.0
    rs=[]
    for r in regions:
        a,b,c,d=max(x0,r[0]),max(y0,r[1]),min(x1,r[2]),min(y1,r[3])
        if c>a and d>b:rs.append((a,b,c,d))
    xs=sorted({v for r in rs for v in (r[0],r[2])})
    covered=0.0
    for a,c in zip(xs[:-1],xs[1:]):
        spans=sorted((r[1],r[3]) for r in rs if r[0]<c and r[2]>a)
        end=None;height=0.0
        for lo,hi in spans:
            if end is None:height+=hi-lo;end=hi
            elif hi>end:height+=hi-max(lo,end);end=hi
        covered+=(c-a)*height
    return covered/area


def coco_ground_truth(dataset):
    images=[];annotations=[];nignored=0
    for r in dataset.rows:
        images.append({'id':r['id'],'width':r['width'],'height':r['height'],'file_name':r['file']})
        for v in r['valid']:
            x0,y0,x1,y1,cls=v
            if union_overlap_fraction(v[:4],r['ignore'])>=.5:nignored+=1;continue
            w,h=x1-x0,y1-y0
            annotations.append({'id':len(annotations)+1,'image_id':r['id'],'category_id':int(cls)+1,
                                'bbox':[x0,y0,w,h],'area':w*h,'iscrowd':0})
    return {'images':images,'annotations':annotations,'categories':[{'id':i,'name':str(i)} for i in range(1,11)],
            'info':{'scope':'COCO-style metric; class0 rectangle union IoA>=.5 filtered on GT/predictions; others11 is unscored background; not VisDrone official evaluator'},
            'ignored_valid_gt':nignored}


def bind_ground_truth(dataset):
    gt=coco_ground_truth(dataset)
    assert len(gt['images'])==548 and len(gt['annotations'])==38602 and gt['ignored_valid_gt']==157
    expected=ROOT/'validation_GT_FROZEN_v1.json'
    assert gt==load(expected)
    return gt


def postprocess(outputs,sizes):
    logits,boxes=outputs['pred_logits'].float(),outputs['pred_boxes'].float()
    b,q,c=logits.shape
    assert q==400 and c==10 and sizes.shape==(b,2)
    scores,indices=logits.sigmoid().flatten(1).topk(300,dim=1)
    query_indices=indices//10
    labels=indices%10
    cx,cy,w,h=boxes.unbind(-1)
    xyxy=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)
    xyxy=xyxy.gather(1,query_indices[:,:,None].expand(-1,-1,4))
    # The unchanged source dataset emits orig_size=[width,height].
    width,height=sizes.unbind(-1)
    xyxy=xyxy*torch.stack([width,height,width,height],1)[:,None]
    assert torch.isfinite(scores).all() and torch.isfinite(xyxy).all()
    return [{'boxes':xyxy[i],'scores':scores[i],'labels':labels[i]} for i in range(b)]


def metrics_from_predictions(dataset,predictions,outdir,prefix):
    gt=bind_ground_truth(dataset)
    save_json(outdir/(prefix+'_GT.json'),gt)
    save_json(outdir/(prefix+'_predictions.json'),predictions)
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):
        coco=COCO();coco.dataset=copy.deepcopy(gt);coco.createIndex()
        if predictions:dt=coco.loadRes(predictions)
        else:
            dt=COCO();dt.dataset={'images':gt['images'],'annotations':[],'categories':gt['categories']};dt.createIndex()
        ev=COCOeval(coco,dt,'bbox');ev.params.maxDets=[1,10,300]
        ev.evaluate();ev.accumulate()
    (outdir/(prefix+'_COCO_stdout.txt')).write_text(capture.getvalue(),encoding='utf-8')
    p=ev.eval['precision'];rec=ev.eval['recall']
    def ap(area=0,iou=None):
        z=p[:,:,:,area,-1]
        if iou is not None:z=z[np.isclose(ev.params.iouThrs,iou)]
        z=z[z>=0];return float(z.mean()*100) if len(z) else None
    def ar(area=0):
        z=rec[:,:,area,-1];z=z[z>=0];return float(z.mean()*100) if len(z) else None
    metrics={'AP':ap(),'AP50':ap(iou=.5),'AP75':ap(iou=.75),'APs':ap(1),'APs75':ap(1,.75),
             'APm':ap(2),'APl':ap(3),'ARs':ar(1),'max_detections':300,'evaluated_GT':38602,
             'ignored_valid_GT':157,'predictions_after_ignore':len(predictions),
             'area_convention':'clipped original-image GT pixel area; small<=1024',
             'scope':'COCO-style; class0 union IoA filtering; not official VisDrone score'}
    np.savez_compressed(outdir/(prefix+'_precision_recall.npz'),precision=p,recall=rec,iou_thresholds=ev.params.iouThrs)
    return metrics


@torch.inference_mode()
def evaluate_model(model,dataset,outdir,prefix):
    model.eval();predictions=[];filtered=0;start=time.monotonic()
    loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=2,collate_fn=collate,
                      pin_memory=True,generator=torch.Generator().manual_seed(20261101+700000))
    assert len(loader)==137
    for images,targets in loader:
        images=images.cuda(non_blocking=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(images)
        assert_output(outputs,len(targets),400)
        sizes=torch.stack([t['orig_size'] for t in targets]).cuda()
        results=postprocess(outputs,sizes)
        for target,result in zip(targets,results):
            iid=int(target['image_id']);row=dataset.rows[iid-1]
            boxes=result['boxes'].cpu().tolist();scores=result['scores'].cpu().tolist();labels=result['labels'].cpu().tolist()
            for box,score,label in zip(boxes,scores,labels):
                if union_overlap_fraction(box,row['ignore'])>=.5:filtered+=1;continue
                x0,y0,x1,y1=box
                if x1<=x0 or y1<=y0:continue
                predictions.append({'image_id':iid,'category_id':int(label)+1,
                                    'bbox':[x0,y0,x1-x0,y1-y0],'score':float(score)})
    metrics=metrics_from_predictions(dataset,predictions,Path(outdir),prefix)
    metrics.update(ignore_filtered_predictions=filtered,validation_seconds=time.monotonic()-start)
    save_json(Path(outdir)/(prefix+'_metrics.json'),metrics)
    return metrics


@torch.inference_mode()
def latency_ms(model,image):
    model.eval()
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        for _ in range(10):model(image)
        torch.cuda.synchronize();samples=[]
        for _ in range(30):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();model(image);b.record();b.synchronize();samples.append(a.elapsed_time(b))
    return {'median_ms':float(np.median(samples)),'mean_ms':float(np.mean(samples)),'samples_ms':samples,
            'warmup':10,'repeats':30,'scope':'batch1 native eager CUDA BF16 model forward; excludes postprocess; not onboard/TensorRT latency'}
