"""Borrowed LN/QK attention, linear classifier, CE. No GT accepted at inference."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_convert,box_iou

def preserve_score_multiset(native,semantic):
    if native.shape!=semantic.shape or native.shape[-1]!=10:raise ValueError('Expected ten classes.')
    if native.dtype!=semantic.dtype or not torch.isfinite(native).all() or not torch.isfinite(semantic).all():raise ValueError('Finite equal-dtype scores required.')
    order=torch.argsort(semantic.detach(),dim=-1,descending=True,stable=True)
    values=torch.sort(native,dim=-1,descending=True,stable=True).values
    return torch.empty_like(native).scatter(-1,order,values)

class ClassificationReadout(nn.Module):
    def __init__(self,mode='attention',zero_output=True):
        super().__init__()
        if mode not in ['attention','self','mean']:raise ValueError(mode)
        self.mode=mode;self.norm=nn.LayerNorm(256);self.query=nn.Linear(256,32);self.key=nn.Linear(256,32);self.output=nn.Linear(512,10)
        if zero_output:nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,dense,native,indices,mask):
        b,q,k=indices.shape
        if hidden.shape!=(b,q,256) or native.shape!=(b,q,10) or dense.shape!=(b,8400,256):raise ValueError('Native shapes changed.')
        if mask.shape!=indices.shape or not mask[...,0].all() or (indices[mask]<0).any() or (indices[mask]>=8400).any():raise ValueError('Invalid pool.')
        hidden,dense,native=hidden.detach(),dense.detach(),native.detach()
        batch=torch.arange(b,device=indices.device)[:,None,None]
        pooled=self.norm(dense[batch,indices.clamp_min(0)]);h=self.norm(hidden)
        if self.mode=='attention':
            score=(self.query(h)[:,:,None]*self.key(pooled)).sum(-1)/math.sqrt(32)
            attention=score.masked_fill(~mask,-torch.inf).softmax(-1)
        elif self.mode=='mean':attention=mask.to(pooled.dtype)/mask.sum(-1,keepdim=True)
        else:
            attention=torch.zeros_like(mask,dtype=pooled.dtype);attention[...,0]=1
        residual=(attention[...,None]*(pooled-pooled[:,:,:1])).sum(-2)
        delta=self.output(torch.cat([h,residual],-1)).to(native.dtype)
        return native+delta,{'attention':attention,'pool_residual':residual,'delta':delta}

@torch.no_grad()
def training_geometry_labels(boxes,target,visible,assigned):
    q=len(boxes);ng=len(target['boxes']);device=boxes.device
    labels=torch.full((q,),-1,dtype=torch.long,device=device);owner=labels.clone();reason=torch.zeros(q,dtype=torch.long,device=device)
    if visible.shape!=(ng,) or assigned.shape!=(q,):raise ValueError('Metadata mismatch.')
    if not torch.isfinite(visible).all() or ((visible<0)|(visible>1.00001)).any():raise ValueError('Invalid visibility.')
    if ((assigned<-1)|(assigned>=ng)).any():raise ValueError('Invalid final assignment.')
    if not ng:return owner,labels,reason
    I=box_iou(box_convert(boxes.detach(),'cxcywh','xyxy'),box_convert(target['boxes'],'cxcywh','xyxy'))
    support=(I>=.5).sum(1);j=I.argmax(1);unique=support==1;full=visible[j]>=.999
    conflict=(assigned>=0)&(target['labels'][assigned.clamp_min(0)]!=target['labels'][j]);valid=unique&full&~conflict
    owner[valid]=j[valid];labels[valid]=target['labels'][j[valid]]
    reason[unique&~full]=1;reason[unique&full&conflict]=2;reason[valid]=3
    return owner,labels,reason

def semantic_ce(scores,labels,owner):
    valid=labels>=0
    if not torch.equal(valid,owner>=0):raise ValueError('Eligibility mismatch.')
    if not valid.any():return scores.sum()*0
    values=F.cross_entropy(scores[valid],labels[valid],reduction='none')
    _,inverse,count=owner[valid].unique(return_inverse=True,return_counts=True)
    return (values/count[inverse]).sum()/len(count)

@torch.no_grad()
def final_assignment(matcher,output,targets):
    matches=matcher({'pred_logits':output['pred_logits'].detach(),'pred_boxes':output['pred_boxes'].detach()},targets)['indices'];result=[]
    for (queries,gt),L in zip(matches,output['pred_logits']):
        assigned=torch.full((len(L),),-1,dtype=torch.long,device=L.device);assigned[queries]=gt;result.append(assigned)
    return result

def paired_semantic_loss(scores,outputs,targets,visible,matcher):
    loss=0;details={}
    for view in ['global','local']:
        assignment=final_assignment(matcher,outputs[view],targets[view]);per=[];meta=[]
        for i,tar in enumerate(targets[view]):
            owner,labels,reason=training_geometry_labels(outputs[view]['pred_boxes'][i],tar,visible[view][i],assignment[i])
            per.append(semantic_ce(scores[view][i],labels,owner));meta.append({'owner':owner,'labels':labels,'reason':reason,'final_assignment':assignment[i]})
        loss=loss+.5*torch.stack(per).mean();details[view]=meta
    return loss,details
