"""Shared data, target and optimizer utilities for the controlled A60 detector."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image
import torch
from torch import nn
from model_A60_v1 import EndpointDetector,configuration
from criterion_A60_v1 import SetCriterionDynamicK,HungarianMatcherDynamicK
from box_progress_conditioning_v1 import sample_prior,training_inputs,xywh_to_xyxy

TRAIN=Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone/VisDrone2019-DET-train')
VAL=Path('/root/uav_a48_20261101_v1/val_inputs')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def save_json(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def backend(seed=20261102):
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def load_item(name,image_dir=None,annotation_dir=None,flip=False):
    image_dir=TRAIN/'images' if image_dir is None else Path(image_dir)
    annotation_dir=TRAIN/'annotations' if annotation_dir is None else Path(annotation_dir)
    with Image.open(image_dir/name) as im:
        im=im.convert('RGB');w,h=im.size
        image=np.asarray(im.resize((640,640),Image.Resampling.BILINEAR)).copy()
    raw=(annotation_dir/(Path(name).stem+'.txt')).read_bytes()
    boxes=[];labels=[];original=[];ignore=[]
    for line in raw.decode().splitlines():
        z=[float(v.strip()) for v in line.split(',') if v.strip()]
        assert len(z)>=8
        x,y,bw,bh,score,cat=z[:6]
        a=max(0.,min(w,x));b=max(0.,min(h,y));c=max(0.,min(w,x+bw));d=max(0.,min(h,y+bh))
        if c<=a or d<=b:continue
        if int(cat)==0:ignore.append([a,b,c,d])
        if score>0 and 1<=int(cat)<=10:
            boxes.append([(a+c)/(2*w),(b+d)/(2*h),(c-a)/w,(d-b)/h])
            labels.append(int(cat)-1);original.append([a,b,c,d,int(cat)-1])
    b=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
    if flip:
        image=image[:,::-1].copy();b[:,0]=1.-b[:,0]
    t=dict(boxes=b,labels=torch.tensor(labels,dtype=torch.long),name=name,width=w,height=h,
        original=original,ignore=ignore,annotation_sha256=hashlib.sha256(raw).hexdigest())
    return torch.from_numpy(image).permute(2,0,1).float(),t

def training_query_count(targets):
    n=max(len(t['labels']) for t in targets)
    q=max(300,32*math.ceil(n/32))
    assert q<=1024 and q>=n
    return q

@torch.no_grad()
def prepare_training(targets,seed,progress=None):
    gen=torch.Generator(device='cpu');gen.manual_seed(int(seed))
    q=training_query_count(targets);batch=len(targets)
    prior=sample_prior(batch,q,'cpu',generator=gen)
    # Shared numerical floor follows the source detector's valid-width convention.
    prior[...,2:].clamp_min_(1e-4)
    endpoints=[]
    for target in targets:
        gt=target['boxes'];assert len(gt)<=q
        pad=sample_prior(1,q-len(gt),'cpu',generator=gen)[0]
        pad[...,2:].clamp_min_(1e-4)
        all_boxes=torch.cat((gt,pad),dim=0)
        endpoints.append(all_boxes[torch.randperm(q,generator=gen)])
    endpoint=torch.stack(endpoints)
    u=torch.rand(batch,generator=gen) if progress is None else torch.full((batch,),float(progress))
    return dict(prior=prior,endpoint=endpoint,progress=u,queries=q)

def mode_inputs(mode,prepared,device):
    box,t,alpha=training_inputs(mode,prepared['prior'],prepared['endpoint'],prepared['progress'])
    return box.to(device),t.to(device),box,alpha

def loss_targets(targets,device):
    out=[]
    for t in targets:
        wh=torch.full((4,),640.,device=device)
        b=t['boxes'].to(device);absolute=xywh_to_xyxy(b)*wh
        out.append(dict(labels=t['labels'].to(device),boxes=b,boxes_xyxy=absolute,
            image_size_xyxy=wh,image_size_xyxy_tgt=wh[None].repeat(len(b),1),
            area=(absolute[:,2]-absolute[:,0])*(absolute[:,3]-absolute[:,1])))
    return out

def criterion():
    cfg=configuration()
    matcher=HungarianMatcherDynamicK(cfg,cost_class=2.,cost_bbox=5.,cost_giou=2.,use_focal=True)
    weights={'loss_ce':2.,'loss_bbox':5.,'loss_giou':2.}
    weights.update({k+'_'+str(i):v for i in range(5) for k,v in list(weights.items())})
    return SetCriterionDynamicK(cfg,10,matcher,weights,.1,['labels','boxes'],True)

def optimizer(model):
    groups={}
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        lr=5e-4 if '.class_logits.' in name else (1e-5 if name.startswith('backbone.bottom_up.') else 1e-4)
        wd=0. if name.endswith('.bias') or '.norm' in name else 1e-4
        groups.setdefault((lr,wd),[]).append(param)
    return torch.optim.AdamW([dict(params=p,lr=lr,weight_decay=wd,initial_lr=lr) for (lr,wd),p in groups.items()],betas=(.9,.999))

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
