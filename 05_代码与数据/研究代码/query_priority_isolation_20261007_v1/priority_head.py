"""Isolated query-priority residual; a borrowed ranking mechanism, not a novelty claim."""
import torch
from torch import nn
from torch.nn import functional as F

class QueryPriorityResidual(nn.Module):
    def __init__(self,hidden_dim=256):
        super().__init__()
        self.proj=nn.Linear(hidden_dim,1,bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self,content,class_logits):
        if content.shape[:-1]!=class_logits.shape[:-1] or content.shape[-1]!=self.proj.in_features:
            raise ValueError('Dense content and class logits must have matching token dimensions.')
        # Pair loss trains only the residual. Native class and feature objectives stay separate.
        return class_logits.detach().max(-1).values+self.proj(content.detach()).squeeze(-1)

def fixed_pair_priority_loss(priority,pairs,normalizer):
    """Pairs supplied by a documented training-only GT assignment, never by inference."""
    if priority.ndim!=1:raise ValueError('Expected a single dense token score vector.')
    if not pairs:return priority.sum()*0.
    positives=torch.tensor([p['positive_token'] for p in pairs],device=priority.device)
    negatives=torch.tensor([p['negative_token'] for p in pairs],device=priority.device)
    weights=priority.new_tensor([p['IoU'] for p in pairs])
    return .25*(weights*F.softplus(priority[negatives]-priority[positives])).sum()/max(float(normalizer),1.)
