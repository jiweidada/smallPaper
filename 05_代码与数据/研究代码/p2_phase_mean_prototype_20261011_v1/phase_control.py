"""Counterfactual for four within-cell positions; no added parameters or buffers."""
from torch.nn import functional as F

def phase_mean(folded):
    if folded.ndim!=4 or folded.shape[1]%4:
        raise ValueError('Expected B,4C,H,W pixel-unshuffled features')
    b,c,h,w=folded.shape
    phases=folded.reshape(b,c//4,4,h,w)
    return phases.mean(2,keepdim=True).expand(-1,-1,4,-1,-1).reshape_as(folded)

class PhaseMeanForward:
    def forward(self,p2,p3):
        detail=self.reduce(p2)
        h,w=detail.shape[-2:]
        if h%2 or w%2:
            detail=F.pad(detail,(0,w%2,0,h%2))
        detail=F.pixel_unshuffle(detail,2)
        detail=phase_mean(detail)
        if detail.shape[-2:]!=p3.shape[-2:]:
            raise ValueError('P2 spatial reorder and P3 sizes differ')
        return p3+self.restore(detail)

def controlled_class(original_branch_class):
    return type('PhaseMeanP2ToP3Residual',(PhaseMeanForward,original_branch_class),{})
