"""Exact old geometry policy; unique int64 lexicographic keys for float32 IoU."""
import torch
from torchvision.ops import box_convert,box_iou

@torch.no_grad()
def prediction_pool(boxes,valid,maxlogit,top,other_k=8,threshold=.5):
    b,n,_=boxes.shape;q=top.shape[1]
    if other_k!=8 or threshold!=.5 or n!=8400 or q!=300:
        raise ValueError('Protocol54 fixes8400 proposals,300 queries,8 others,IoU.5.')
    valid=valid.bool()&torch.isfinite(boxes).all(-1)&(boxes[...,2:]>0).all(-1)
    if not valid.gather(1,top).all() or not torch.isfinite(maxlogit).all():
        raise ValueError('Invalid native self proposal or score.')
    relation=torch.stack([box_iou(box_convert(boxes[i,top[i]],'cxcywh','xyxy'),box_convert(boxes[i],'cxcywh','xyxy')) for i in range(b)])
    eligible=valid[:,None,:]&(relation>=threshold)&(torch.arange(n,device=top.device)[None,None,:]!=top[:,:,None])
    ranked=torch.where(eligible,relation,torch.full_like(relation,-torch.inf))
    secondary=torch.argsort(maxlogit,dim=-1,descending=True,stable=True)[:,None,:].expand(-1,q,-1)
    work=ranked.gather(-1,secondary)
    if work.dtype==torch.float32:
        # Positive float32 bits preserve numerical order exactly. The low 14 bits
        # encode earlier secondary position; 8400 < 2**14, so no affinity is changed.
        bits=work.contiguous().view(torch.int32).to(torch.int64)
        keys=bits*16384+(n-1-torch.arange(n,device=work.device,dtype=torch.int64))
        position=keys.topk(other_k,dim=-1,sorted=True).indices
        values=work.gather(-1,position);candidate=secondary.gather(-1,position)
        mask=torch.isfinite(values)
        indices=[torch.where(mask,candidate,torch.full_like(candidate,-1))]
        affinities=[torch.where(mask,values,torch.zeros_like(values))];masks=[mask]
    else:
        # Exact legacy-v3 fallback; only float32 key path is benchmarked here.
        indices=[];affinities=[];masks=[]
        for _ in range(other_k):
            values,position=work.max(-1,keepdim=True)
            candidate=secondary.gather(-1,position);mask=torch.isfinite(values)
            indices.append(torch.where(mask,candidate,torch.full_like(candidate,-1)))
            affinities.append(torch.where(mask,values,torch.zeros_like(values)));masks.append(mask)
            work.scatter_(-1,position,-torch.inf)
    own=relation.gather(-1,top[:,:,None])
    if not (own==1).all():raise ValueError('Native self IoU must equal1.')
    return (torch.cat([top[:,:,None],*indices],-1),torch.cat([own,*affinities],-1),
            torch.cat([torch.ones_like(own,dtype=torch.bool),*masks],-1))
