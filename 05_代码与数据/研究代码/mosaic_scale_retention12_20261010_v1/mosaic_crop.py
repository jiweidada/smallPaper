"""Training-only source-region selection before native Mosaic.Resize.

Native four-source sampling, Compose policy, affine and downstream transforms
remain in use. Native modules/source files are not changed. This is bbox-only.
"""
import hashlib
import math
import random
import torch
import torchvision.transforms.v2.functional as F
from engine.data._misc import convert_to_tv_tensor

ARMS=('identity','target_crop','random_crop')
CONTRACT=dict(schema='mosaic_scale_retention12_v1',crop_fraction=.5,
    risk_reference_size=640,risk_shortside_threshold=8.,minimum_risky_fraction=.25,
    preferred_anchor_shortside_range=[4.,16.],
    bbox_retention='positive intersection with crop; chosen anchor fully contained',
    rng='local SHA256(seed,epoch,image_id,worker_id,resize_occurrence), three uniform draws',
    native_global_rng_advanced=False,model_changes=False,inference_changes=False,
    no_validation_label_usage=True)
INSTANCE_FIELDS={'boxes','labels','area','iscrowd','mixup'}
METADATA_FIELDS={'image_id','idx','orig_size','size','mosaic_source_image_ids'}


def crop_at(image,target,left,top,width,height):
    """Deterministic crop and aligned bbox-only instance fields; no mutation."""
    assert image.mode=='RGB' and 0<=left and 0<=top
    assert left+width<=image.width and top+height<=image.height
    assert not any(k in target for k in ('masks','keypoints'))
    boxes=target['boxes'].as_subclass(torch.Tensor)
    assert boxes.ndim==2 and boxes.shape[1]==4 and bool(torch.isfinite(boxes).all())
    n=len(boxes)
    for key,value in target.items():
        if torch.is_tensor(value) and value.ndim and len(value)==n:
            if key not in INSTANCE_FIELDS|METADATA_FIELDS:
                raise ValueError('Unclassified target field: '+key)
    adjusted=boxes.clone()-boxes.new_tensor([left,top,left,top])
    adjusted[:,0::2].clamp_(0,width);adjusted[:,1::2].clamp_(0,height)
    keep=(adjusted[:,2]>adjusted[:,0])&(adjusted[:,3]>adjusted[:,1])
    out={}
    for key,value in target.items():
        if key in INSTANCE_FIELDS:
            assert torch.is_tensor(value) and len(value)==n
            out[key]=value[keep].clone()
        else:out[key]=value.clone() if torch.is_tensor(value) else value
    out['boxes']=convert_to_tv_tensor(adjusted[keep],key='boxes',box_format='xyxy',spatial_size=(height,width))
    if 'area' in out:
        wh=adjusted[keep,2:]-adjusted[keep,:2]
        out['area']=(wh[:,0]*wh[:,1]).to(out['area'].dtype)
    out['orig_size']=torch.as_tensor([width,height])
    if 'size' in out:out['size']=torch.as_tensor([height,width])
    return F.crop(image,top,left,height,width),out,keep


class MosaicSourceCrop:
    def __init__(self,dataset,seed,arm,observer=None):
        if arm not in ARMS:raise ValueError(arm)
        self.dataset=dataset;self.seed=int(seed);self.arm=arm;self.observer=observer;self.occurrence=0

    def __call__(self,module,inputs):
        packed=len(inputs)==1
        sample=inputs[0] if packed else inputs
        assert len(sample)==2
        image,target=sample
        occurrence=self.occurrence;self.occurrence+=1
        if self.arm=='identity':return None
        assert image.mode=='RGB' and image.width>=2 and image.height>=2
        boxes=target['boxes'].as_subclass(torch.Tensor)
        assert boxes.ndim==2 and boxes.shape[1]==4
        n=len(boxes);w,h=image.size
        wh=boxes[:,2:]-boxes[:,:2]
        shorts=(wh*wh.new_tensor([640/w,640/h])).min(-1).values
        risky=shorts<CONTRACT['risk_shortside_threshold']
        fraction=float(risky.float().mean()) if n else 0.
        if fraction<CONTRACT['minimum_risky_fraction']:
            if self.observer:self.observer(dict(arm=self.arm,image_id=int(target['image_id']),occurrence=occurrence,
                applied=False,risky_fraction=fraction,source_GT=n,kept_GT=n))
            return None
        cw=max(1,w//2);ch=max(1,h//2)
        info=torch.utils.data.get_worker_info();worker=-1 if info is None else info.id
        key=f'{self.seed}:{int(self.dataset.epoch)}:{int(target["image_id"])}:{worker}:{occurrence}'
        local=random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(),'big'))
        select,xdraw,ydraw=[local.random() for _ in range(3)]
        intervals=[]
        for i,(x1,y1,x2,y2) in enumerate(boxes.tolist()):
            xmin=max(0,math.ceil(x2-cw));xmax=min(w-cw,math.floor(x1))
            ymin=max(0,math.ceil(y2-ch));ymax=min(h-ch,math.floor(y1))
            if xmin<=xmax and ymin<=ymax:intervals.append((i,xmin,xmax,ymin,ymax))
        preferred=[r for r in intervals if 4<=float(shorts[r[0]])<16]
        small=[r for r in intervals if float(shorts[r[0]])<16]
        candidates=preferred or small or intervals
        anchor=None
        if self.arm=='target_crop' and candidates:
            anchor,xmin,xmax,ymin,ymax=candidates[min(int(select*len(candidates)),len(candidates)-1)]
        else:xmin,xmax,ymin,ymax=0,w-cw,0,h-ch
        left=xmin+min(int(xdraw*(xmax-xmin+1)),xmax-xmin)
        top=ymin+min(int(ydraw*(ymax-ymin+1)),ymax-ymin)
        cropped,out,keep=crop_at(image,target,left,top,cw,ch)
        if anchor is not None:
            assert bool(keep[anchor]),'Chosen anchor was removed'
            assert bool((boxes[anchor,:2]>=boxes.new_tensor([left,top])).all())
            assert bool((boxes[anchor,2:]<=boxes.new_tensor([left+cw,top+ch])).all())
        if self.observer:self.observer(dict(arm=self.arm,image_id=int(target['image_id']),occurrence=occurrence,
            applied=True,risky_fraction=fraction,source_GT=n,kept_GT=int(keep.sum()),
            crop=[left,top,cw,ch],source_size=[w,h],anchor_index=anchor,local_rng_key_sha256=hashlib.sha256(key.encode()).hexdigest()))
        return ((cropped,out),) if packed else (cropped,out)


def install(dataset,seed,arm,observer=None):
    operators=[op for op in dataset._transforms.transforms if type(op).__name__=='Mosaic']
    assert len(operators)==1 and operators[0].use_cache is False and operators[0].probability==1.
    assert dataset._transforms.policy['epoch']==[4,64,120] and dataset._transforms.mosaic_prob==.5
    hook=MosaicSourceCrop(dataset,seed,arm,observer)
    handle=operators[0].resize.register_forward_pre_hook(hook)
    return hook,handle
