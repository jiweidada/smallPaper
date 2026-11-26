"""Same learned semantic scores, alternative inference rules; no GT input.
Product rule is a DCR-style operation adaptation, not official DCR reproduction.
"""
import torch
from semantic_branch import preserve_score_multiset
def readout(native,semantic,rule):
    if native.shape!=semantic.shape or native.dtype!=semantic.dtype:raise ValueError('Shape/dtype mismatch.')
    if rule=='spectrum':return preserve_score_multiset(native,semantic)
    if rule=='direct':return semantic
    if rule=='product':
        p=native.sigmoid()*semantic.softmax(-1)
        if not ((p>0)&(p<1)).all():raise ValueError('Product not representable as finite logits; no silent flooring.')
        return p.log()-torch.log1p(-p)
    if rule=='native':return native
    raise ValueError(rule)
