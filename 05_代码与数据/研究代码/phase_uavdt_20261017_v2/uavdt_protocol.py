"""Native UAVDT overall DET matching and process-local DEIM extension points."""
import json
from pathlib import Path

import numpy as np

METRIC_NAMES=['AP70','AR70']
PROTOCOL={'iou_threshold':.7,'class_agnostic':True,'ignore_rule':'strict full rectangle containment; author test sequence pass list',
          'matching':'descending score, one GT once, last GT on equal best IoU','coordinates':'continuous original-image xywh',
          'AP':'VOC continuous precision envelope','native_per_image_detection_cap':None}

def voc_ap(recall,precision):
    r=np.r_[0.,recall,1.];p=np.r_[0.,precision,0.]
    p=np.maximum.accumulate(p[::-1])[::-1];ids=np.flatnonzero(r[1:]!=r[:-1])+1
    return float(np.sum((r[ids]-r[ids-1])*p[ids]))

def contained_mask(detections,regions):
    d=np.asarray(detections,dtype=np.float64).reshape(-1,5);r=np.asarray(regions,dtype=np.float64).reshape(-1,4)
    if not len(d) or not len(r):return np.zeros(len(d),dtype=bool)
    return ((d[:,None,0]>r[None,:,0])&(d[:,None,1]>r[None,:,1])&
            (d[:,None,0]+d[:,None,2]<r[None,:,0]+r[None,:,2])&
            (d[:,None,1]+d[:,None,3]<r[None,:,1]+r[None,:,3])).any(axis=1)

def match_frame(gt,dt,threshold=.7):
    gt=np.asarray(gt,dtype=np.float64).reshape(-1,4);dt=np.asarray(dt,dtype=np.float64).reshape(-1,5)
    assert np.isfinite(gt).all() and np.isfinite(dt).all()
    order=np.argsort(-dt[:,4],kind='stable');d=dt[order];chosen=np.full(len(d),-1,dtype=np.int32)
    if not len(gt) or not len(d):return d[:,4],chosen,order
    wh=np.maximum(0.,np.minimum((d[:,:2]+d[:,2:4])[:,None,:],(gt[:,:2]+gt[:,2:4])[None,:,:])-np.maximum(d[:,None,:2],gt[None,:,:2]))
    inter=wh[:,:,0]*wh[:,:,1];union=d[:,2,None]*d[:,3,None]+gt[None,:,2]*gt[None,:,3]-inter
    overlap=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);matched=np.zeros(len(gt),dtype=bool)
    for i in np.flatnonzero((overlap>=threshold).any(axis=1)):
        row=overlap[i];available=np.flatnonzero((~matched)&(row>=threshold))
        if len(available):
            maximum=row[available].max();winner=available[row[available]==maximum][-1]
            chosen[i]=winner;matched[winner]=True
    return d[:,4],chosen,order

def curve(scores,flags,count):
    scores=np.asarray(scores,dtype=np.float64);flags=np.asarray(flags,dtype=np.int8)
    flags=flags[np.argsort(-scores,kind='stable')]
    tp=np.cumsum(flags==1);fp=np.cumsum(flags==0)
    recall=tp/max(1,count);precision=tp/np.maximum(1,tp+fp)
    return {'AP70':voc_ap(recall,precision),'AR70':float(recall[-1]) if len(recall) else 0.},recall,precision

class NativeScorer:
    def __init__(self,annotation):
        self.annotation=annotation
        self.images={int(image['id']):image for image in annotation['images']}
        self.order=sorted(self.images,key=lambda i:(self.images[i]['uav_sequence'],self.images[i]['uav_frame']))
        grouped={i:[] for i in self.images}
        for ann in annotation['annotations']:grouped[int(ann['image_id'])].append(ann['bbox'])
        self.gt={i:np.asarray(v,dtype=np.float64).reshape(-1,4) for i,v in grouped.items()}
        self.count=sum(map(len,self.gt.values()));self.cleanup()

    def cleanup(self):self.parts={};self.ignored=0

    def update(self,image_id,dt):
        image_id=int(image_id)
        if image_id not in self.images or image_id in self.parts:raise RuntimeError('Unexpected or duplicate evaluation frame '+str(image_id))
        image=self.images[image_id];dt=np.asarray(dt,dtype=np.float64).reshape(-1,5)
        if image['uav_apply_native_ignore']:
            ignored=contained_mask(dt,image['uav_ignore_boxes']);self.ignored+=int(ignored.sum());dt=dt[~ignored]
        scores,assignment,_=match_frame(self.gt[image_id],dt)
        self.parts[image_id]=(scores,(assignment>=0).astype(np.int8))

    def finish(self):
        if set(self.parts)!=set(self.images):raise RuntimeError('Incomplete evaluation image coverage')
        scores=np.concatenate([self.parts[i][0] for i in self.order])
        flags=np.concatenate([self.parts[i][1] for i in self.order])
        named,recall,precision=curve(scores,flags,self.count)
        return named,recall,precision

def erase_ignored(image,regions,positives):
    """Common training adaptation: remove unlabeled ignored pixels, protect all positive GT pixels."""
    from PIL import Image
    if not regions:return image
    original=np.asarray(image);out=original.copy();height,width=original.shape[:2]
    protected=np.zeros((height,width),dtype=bool)
    def bounds(box):
        x,y,w,h=box
        return max(0,int(np.floor(x))),max(0,int(np.floor(y))),min(width,int(np.ceil(x+w))),min(height,int(np.ceil(y+h)))
    for box in positives:
        x1,y1,x2,y2=bounds(box)
        if x2>x1 and y2>y1:protected[y1:y2,x1:x2]=True
    for box in regions:
        x1,y1,x2,y2=bounds(box)
        if x2<=x1 or y2<=y1:continue
        fill=np.rint(original[y1:y2,x1:x2].mean(axis=(0,1))).astype(out.dtype)
        patch=out[y1:y2,x1:x2];patch[~protected[y1:y2,x1:x2]]=fill
    return Image.fromarray(out)

class UAVDTEvaluator:
    def __init__(self,ann_file):
        self.ann_file=str(ann_file);self.annotation=json.loads(Path(ann_file).read_text())
        self.scorer=NativeScorer(self.annotation);self.iou_types=['bbox'];self.stats={};self.named_metrics=None

    def cleanup(self):self.scorer.cleanup();self.stats={};self.named_metrics=None

    def update(self,predictions,targets=None):
        for image_id,prediction in predictions.items():
            boxes=prediction['boxes'].detach().cpu().numpy().astype(np.float64)
            scores=prediction['scores'].detach().cpu().numpy().astype(np.float64)
            labels=prediction['labels'].detach().cpu().numpy()
            if not (labels==0).all():raise RuntimeError('UAVDT model must use contiguous vehicle label zero')
            dt=np.column_stack([boxes[:,:2],boxes[:,2:]-boxes[:,:2],scores])
            self.scorer.update(image_id,dt)

    def synchronize_between_processes(self):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size()!=1:raise RuntimeError('Single GPU process per training only')

    def accumulate(self):
        self.named_metrics,_,_=self.scorer.finish()
        self.stats={'uavdt_eval_bbox':[self.named_metrics[name] for name in METRIC_NAMES]}

    def summarize(self):print(json.dumps({'protocol':'native UAVDT vehicle AP70','metrics':self.named_metrics,'ignored_predictions':self.scorer.ignored}),flush=True)

def install_native_adapter():
    from engine.core import YAMLConfig
    from engine.solver.det_solver import DetSolver
    from engine.data.dataset.coco_dataset import CocoDetection
    if getattr(CocoDetection,'_uavdt_extension_installed',False):return
    original_eval=YAMLConfig.evaluator.fget;original_names=DetSolver._coco_metric_names;original_image=CocoDetection._load_image
    def evaluator(self):
        settings=self.yaml_cfg.get('evaluator',{})
        if settings.get('type')!='UAVDTEvaluator':return original_eval(self)
        if self._evaluator is None:self._evaluator=UAVDTEvaluator(settings['ann_file'])
        return self._evaluator
    def names(self):return METRIC_NAMES if isinstance(getattr(self,'evaluator',None),UAVDTEvaluator) else original_names(self)
    def load_image(self,image_id):
        image=original_image(self,image_id)
        if self.coco.dataset.get('uavdt_split')=='train':
            assert not self.cache_flag and not self.ram_cache
            info=self.coco.imgs[image_id]
            positives=[ann['bbox'] for ann in self.coco.imgToAnns[image_id]]
            image=erase_ignored(image,info['uav_ignore_boxes'],positives)
        return image
    YAMLConfig.evaluator=property(evaluator);DetSolver._coco_metric_names=names
    CocoDetection._load_image=load_image;CocoDetection._uavdt_extension_installed=True
