"""Fixed-pair representatives, full native pool competition; borrowed smooth-rank idea."""
import torch
from torch.nn import functional as F

def global_priority_loss(priority,pairs,normalizer,budget=300,control=False):
    if priority.ndim!=1:raise ValueError('Expected one full native token score vector.')
    if budget!=300:raise ValueError('This isolated protocol fixes the native budget to300.')
    if not pairs:return priority.sum()*0.
    terms=[]
    for pair in pairs:
        positive=int(pair['positive_token'])
        mask=torch.arange(len(priority),device=priority.device)!=positive
        differences=priority[mask]-priority[positive]
        if control:value=F.softplus(differences).mean()
        else:
            rank=1.+differences.sigmoid().sum()
            value=F.softplus((rank-float(budget))/float(budget))
        terms.append(value*float(pair['IoU']))
    return .25*torch.stack(terms).sum()/max(float(normalizer),1.)
