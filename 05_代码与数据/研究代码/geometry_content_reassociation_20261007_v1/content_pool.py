"""Borrowed attention/residual primitives, proposal-affinity content adapter.

Research candidate only. No GT, labels, matched identities or diagnostic cohorts
are accepted by either function. The original geometry is owned by the caller.
"""
import math
import torch
from torch import nn
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
    # Initial index order -> stable max-logit order -> stable affinity order.
    secondary=torch.argsort(maxlogit,dim=-1,descending=True,stable=True)[:,None,:].expand(-1,q,-1)
    ranked_secondary=ranked.gather(-1,secondary)
    order=torch.argsort(ranked_secondary,dim=-1,descending=True,stable=True)[...,:other_k]
    others=secondary.gather(-1,order);values=ranked_secondary.gather(-1,order);mask=torch.isfinite(values)
    others=torch.where(mask,others,torch.full_like(others,-1));values=torch.where(mask,values,torch.zeros_like(values))
    own_value=relation.gather(-1,top[:,:,None])
    if not (own_value==1).all():raise ValueError('Native self IoU must equal1.')
    return torch.cat([top[:,:,None],others],-1),torch.cat([own_value,values],-1),torch.cat([torch.ones_like(own_value,dtype=torch.bool),mask],-1)

class ContentReassociation(nn.Module):
    def __init__(self,dim=256,attention_dim=32,use_conditions=True):
        super().__init__();self.use_conditions=use_conditions
        self.norm=nn.LayerNorm(dim);self.query=nn.Linear(dim,attention_dim);self.key=nn.Linear(dim,attention_dim)
        self.conditions=nn.Sequential(nn.Linear(15,32),nn.GELU(),nn.Linear(32,1))
        self.gamma=nn.Parameter(torch.zeros(()));self.scale=1/math.sqrt(attention_dim)

    def forward(self,content,dense,logits,boxes,indices,affinity,mask):
        b,q,k=indices.shape;safe=indices.clamp_min(0)
        batch=torch.arange(b,device=indices.device)[:,None,None]
        pooled=dense[batch,safe];proposal=boxes[batch,safe];prob=torch.sigmoid(logits[batch,safe])
        own=proposal[:,:,0];own_prob=prob[:,:,0]
        relcentre=(proposal[...,:2]-own[:,:,None,:2])/own[:,:,None,2:].clamp_min(1e-8)
        relsize=torch.log(proposal[...,2:].clamp_min(1e-8)/own[:,:,None,2:].clamp_min(1e-8))
        pair=torch.cat([relcentre,relsize,affinity[...,None],prob-own_prob[:,:,None]],-1)
        query=self.query(self.norm(content));key=self.key(self.norm(pooled))
        score=(query[:,:,None]*key).sum(-1)*self.scale
        cond=self.conditions(pair).squeeze(-1)
        score=score+cond if self.use_conditions else score+cond*0
        attention=score.masked_fill(~mask,-torch.inf).softmax(-1)
        residual=(attention[...,None]*(pooled-content[:,:,None])).sum(-2)
        adjusted=content+torch.tanh(self.gamma)*residual
        return adjusted,{'attention':attention,'scores':score,'residual':residual,'gamma':self.gamma}

def normal_content_only(adapter,content,dense,logits,boxes,indices,affinity,mask):
    """Keep any DN prefix intact and adapt the300 normal queries at the suffix."""
    n=indices.shape[1]
    if content.shape[1]<n:raise ValueError('Missing normal query suffix.')
    normal,info=adapter(content[:,-n:],dense,logits,boxes,indices,affinity,mask)
    return torch.cat([content[:,:-n],normal],dim=1),info
