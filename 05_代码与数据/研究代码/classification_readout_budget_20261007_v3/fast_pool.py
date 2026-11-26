"""Exact old geometry policy; eight ordered maxima replace a full affinity sort."""
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
    indices=[];affinities=[];masks=[]
    for _ in range(other_k):
        # max returns the first position of a tie, preserving secondary/index order.
        values,position=work.max(-1,keepdim=True)
        candidate=secondary.gather(-1,position);mask=torch.isfinite(values)
        indices.append(torch.where(mask,candidate,torch.full_like(candidate,-1)))
        affinities.append(torch.where(mask,values,torch.zeros_like(values)));masks.append(mask)
        work.scatter_(-1,position,-torch.inf)
    own=relation.gather(-1,top[:,:,None])
    if not (own==1).all():raise ValueError('Native self IoU must equal1.')
    return (torch.cat([top[:,:,None],*indices],-1),torch.cat([own,*affinities],-1),
            torch.cat([torch.ones_like(own,dtype=torch.bool),*masks],-1))
