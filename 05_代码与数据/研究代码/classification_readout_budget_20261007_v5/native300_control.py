"""Required same-head native300-only supply control; no added bbox predictions."""
import torch
from torchvision.ops import box_convert,box_iou
from compact_candidates import native_identity
from compact_wrapper import CompactClassificationDetector

@torch.no_grad()
def native300_pool(boxes,logits,top):
    if boxes.shape[-2:]!=(300,4) or logits.shape[-2:]!=(300,10) or top.shape[-1]!=300:raise ValueError('Native300 inputs required.')
    relation=torch.stack([box_iou(box_convert(row,'cxcywh','xyxy'),box_convert(row,'cxcywh','xyxy')) for row in boxes])
    valid=torch.isfinite(boxes).all(-1)&(boxes[...,2:]>0).all(-1)
    if not valid.all():raise ValueError('Invalid native self box.')
    eligible=(relation>=.5)&(top[:,None,:]!=top[:,:,None])
    ids=torch.argsort(top,dim=-1,stable=True)
    byscore=torch.argsort(logits.max(-1).values.gather(-1,ids),dim=-1,descending=True,stable=True)
    secondary=ids.gather(-1,byscore)[:,None,:].expand(-1,300,-1)
    work=relation.masked_fill(~eligible,-torch.inf).gather(-1,secondary)
    # Full stable sort is intentional in this low-cost control, no extra cap/threshold.
    rank=torch.argsort(work,dim=-1,descending=True,stable=True)[...,:8]
    val=work.gather(-1,rank);mask=torch.isfinite(val)
    ix=top[:,None,:].expand(-1,300,-1).gather(-1,secondary.gather(-1,rank))
    ix=torch.where(mask,ix,torch.full_like(ix,-1));aff=torch.where(mask,val,torch.zeros_like(val))
    self_aff=relation.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)
    if not (self_aff==1).all():raise ValueError('Self IoU must equal1.')
    return torch.cat([top[:,:,None],ix],-1),torch.cat([self_aff,aff],-1),torch.cat([torch.ones_like(self_aff,dtype=torch.bool),mask],-1)

class Native300ClassificationControl(CompactClassificationDetector):
    def _capture(self,memory,logits,anchors,k):
        result=self._select(memory,logits,anchors,k)
        if self.requires_pool:
            top=native_identity(anchors,result[2])
            self._pending=(memory.detach(),logits.detach(),top)
        return result

    def _capture_input(self,*args,**kwargs):
        bundle=self._input(*args,**kwargs)
        if self.requires_pool:
            if self._pending is None:raise RuntimeError('Missing native selection.')
            dense,L,top=self._pending;self._pending=None
            boxes=bundle[1][:,-300:].sigmoid();table_logits=L.gather(1,top[...,None].expand(-1,-1,10))
            pi,aff,mask=native300_pool(boxes,table_logits,top)
            self._pool=(dense,L,top,top,boxes,pi,aff,mask)
        return bundle
