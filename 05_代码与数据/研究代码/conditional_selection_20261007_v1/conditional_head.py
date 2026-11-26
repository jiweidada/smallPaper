"""Borrowed class-dependent scoring and smooth rank; no novelty or trained-gain claim."""
import torch
from torch import nn
from torch.nn import functional as F

class ConditionalSelection(nn.Module):
    def __init__(self,shared=False):
        super().__init__()
        self.proj=nn.Linear(256,10,bias=False)
        nn.init.zeros_(self.proj.weight)
        self.shared=bool(shared)

    def forward(self,content,logits):
        delta=self.proj(content.detach())
        if self.shared:delta=delta.mean(-1,keepdim=True)
        return logits.detach()+delta

def budget_loss(scores,pairs,normalizer):
    if scores.ndim!=2 or scores.shape!=(8400,10):raise ValueError('Fixed native8400x10 required.')
    if not pairs:return scores.sum()*0.
    all_priority=scores.max(-1).values
    terms=[]
    for p in pairs:
        t=int(p['positive_token']);truth=int(p['truth']);mask=torch.arange(8400,device=scores.device)!=t
        rank=1.+(all_priority[mask]-scores[t,truth]).sigmoid().sum()
        terms.append(float(p['IoU'])*F.softplus((rank-300.)/300.))
    return .25*torch.stack(terms).sum()/max(float(normalizer),1.)
