from __future__ import annotations
import contextlib,io,json,pathlib,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from runtime_common import R,VisDroneDataset,collate,save_json

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
            if union_overlap_fraction(v[:4],r['ignore'])>=0.5:nignored+=1;continue
            w,h=x1-x0,y1-y0
            annotations.append({'id':len(annotations)+1,'image_id':r['id'],'category_id':int(cls)+1,'bbox':[x0,y0,w,h],'area':w*h,'iscrowd':0})
    return {'images':images,'annotations':annotations,'categories':[{'id':i,'name':str(i)} for i in range(1,11)],'info':{'scope':'COCO-style metric; class0 rectangle union IoA>=.5 filtered on GT/predictions; others11 is unscored background; not VisDrone official evaluator'},'ignored_valid_gt':nignored}

def metrics_from_predictions(dataset,predictions,outdir,prefix):
    outdir=pathlib.Path(outdir);gt=coco_ground_truth(dataset)
    save_json(outdir/(prefix+'_GT.json'),gt)
    save_json(outdir/(prefix+'_predictions.json'),predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO();coco.dataset=gt;coco.createIndex()
        if predictions:dt=coco.loadRes(predictions)
        else:
            dt=COCO();dt.dataset={'images':gt['images'],'annotations':[],'categories':gt['categories']};dt.createIndex()
        ev=COCOeval(coco,dt,'bbox');ev.params.maxDets=[1,10,300]
        ev.evaluate();ev.accumulate()
    p=ev.eval['precision'];rec=ev.eval['recall']
    def ap(area=0,iou=None):
        z=p[:,:,:,area,-1]
        if iou is not None:z=z[np.isclose(ev.params.iouThrs,iou)]
        z=z[z>=0];return float(z.mean()*100) if len(z) else None
    def ar(area=0):
        z=rec[:,:,area,-1];z=z[z>=0];return float(z.mean()*100) if len(z) else None
    metrics={'AP':ap(),'AP50':ap(iou=.5),'AP75':ap(iou=.75),'APs':ap(1),'APs75':ap(1,.75),'APm':ap(2),'APl':ap(3),'ARs':ar(1),'max_detections':300,'evaluated_GT':len(gt['annotations']),'ignored_valid_GT':gt['ignored_valid_gt'],'predictions_after_ignore':len(predictions),'area_convention':'clipped original-image GT pixel area; small<=1024','scope':'COCO-style; class0 union IoA filtering; not official VisDrone score'}
    np.savez_compressed(outdir/(prefix+'_precision_recall.npz'),precision=p,recall=rec,iou_thresholds=ev.params.iouThrs)
    save_json(outdir/(prefix+'_metrics.json'),metrics)
    return metrics

@torch.inference_mode()
def evaluate_model(model,postprocessor,dataset,outdir,prefix,device='cuda'):
    model.eval();predictions=[];filtered=0;t0=time.monotonic()
    loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=2,collate_fn=collate,pin_memory=True)
    for images,targets in loader:
        images=images.to(device,non_blocking=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):outputs=model(images)
        assert torch.isfinite(outputs['pred_logits']).all() and torch.isfinite(outputs['pred_boxes']).all()
        sizes=torch.stack([t['orig_size'] for t in targets]).to(device)
        results=postprocessor({'pred_logits':outputs['pred_logits'].float(),'pred_boxes':outputs['pred_boxes'].float()},sizes)
        for target,result in zip(targets,results):
            iid=int(target['image_id']);row=dataset.rows[iid-1]
            boxes=result['boxes'].cpu().tolist();scores=result['scores'].cpu().tolist();labels=result['labels'].cpu().tolist()
            for box,score,label in zip(boxes,scores,labels):
                if union_overlap_fraction(box,row['ignore'])>=.5:filtered+=1;continue
                x0,y0,x1,y1=box
                if x1<=x0 or y1<=y0:continue
                predictions.append({'image_id':iid,'category_id':int(label)+1,'bbox':[x0,y0,x1-x0,y1-y0],'score':float(score)})
    metrics=metrics_from_predictions(dataset,predictions,outdir,prefix)
    metrics.update(ignore_filtered_predictions=filtered,validation_seconds=time.monotonic()-t0)
    save_json(pathlib.Path(outdir)/(prefix+'_metrics.json'),metrics)
    return metrics

@torch.inference_mode()
def latency_ms(model,image,warmup=10,repeats=30):
    model.eval()
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        for _ in range(warmup):model(image)
        torch.cuda.synchronize();samples=[]
        for _ in range(repeats):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();model(image);b.record();b.synchronize();samples.append(a.elapsed_time(b))
    return {'median_ms':float(np.median(samples)),'mean_ms':float(np.mean(samples)),'samples_ms':samples,'warmup':warmup,'repeats':repeats,'scope':'batch1 native eager CUDA model forward, BF16 autocast, excludes decode/postprocess; not onboard/TensorRT latency'}
