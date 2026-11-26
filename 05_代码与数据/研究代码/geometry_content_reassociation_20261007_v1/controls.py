"""Explicit controls and future optimizer parameter groups; no optimizer construction."""
import torch
from content_pool import ContentReassociation

class ContentOnlyAttentionControl(ContentReassociation):
    """Same active parameter count, content descriptors replace geometry/prob descriptors."""
    def forward(self,content,dense,logits,boxes,indices,affinity,mask):
        batch=torch.arange(indices.shape[0],device=indices.device)[:,None,None]
        pooled=dense[batch,indices.clamp_min(0)]
        query=self.query(self.norm(content));key=self.key(self.norm(pooled))
        descriptors=(key-query[:,:,None])[...,:15]
        score=(query[:,:,None]*key).sum(-1)*self.scale+self.conditions(descriptors).squeeze(-1)
        attention=score.masked_fill(~mask,-torch.inf).softmax(-1)
        residual=(attention[...,None]*(pooled-content[:,:,None])).sum(-2)
        adjusted=content+torch.tanh(self.gamma)*residual
        return adjusted,{'attention':attention,'scores':score,'residual':residual,'gamma':self.gamma}

def future_parameter_groups(wrapped):
    """Return group metadata and params; caller may construct optimizer only when authorized.

    Prefix-aware LR avoids doubling backbone LR when a registered wrapper is used.
    This function never creates an optimizer and does not change requires_grad.
    """
    groups={};seen=set()
    for full_name,p in wrapped.named_parameters():
        if not p.requires_grad:continue
        if id(p) in seen:raise ValueError('Duplicate active parameter.')
        seen.add(id(p));name=full_name.removeprefix('base.')
        lr=.0001 if name.startswith('backbone.') else .0002
        wd=0. if any(s in name for s in ('bn','norm','bias')) or name.endswith('gamma') else .0001
        key=(lr,wd);group=groups.setdefault(key,{'lr':lr,'initial_lr':lr,'weight_decay':wd,'params':[],'names':[]});group['params'].append(p);group['names'].append(full_name)
    return list(groups.values())
