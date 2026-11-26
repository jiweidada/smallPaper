"""Fixed native300 + valid confidence-ranked extra300, classification only."""
import torch
from torchvision.ops import box_convert,box_iou

@torch.no_grad()
def native_identity(anchors,selected):
    """Recover identities from the ORIGINAL selection result, then check exactly."""
    if anchors.shape[-2:]!=(8400,4) or selected.shape[-2:]!=(300,4):raise ValueError('Fixed native shapes required.')
    if not torch.isfinite(selected).all():raise ValueError('Native selected anchors must be valid.')
    level_wh=anchors[:,[81,6400,8000],2]
    level=(selected[...,2,None]-level_wh[:,None,:]).abs().argmin(-1)
    side=torch.tensor([80,40,20],device=anchors.device)[level]
    offset=torch.tensor([0,6400,8000],device=anchors.device)[level]
    xy=(selected[...,:2].sigmoid()*side[...,None]-.5).round().long()
    index=offset+xy[...,1]*side+xy[...,0]
    if ((index<0)|(index>=8400)).any() or not torch.equal(anchors.gather(1,index[...,None].expand(-1,-1,4)),selected):raise ValueError('Actual selected anchor identity not exactly recovered.')
    if any(len(row.unique())!=300 for row in index):raise ValueError('Native query identities must be unique.')
    return index

@torch.no_grad()
def candidate_table(logits,top,valid):
    b,n,c=logits.shape
    if n!=8400 or c!=10 or top.shape!=(b,300) or valid.shape!=(b,8400):raise ValueError('Fixed native shapes required.')
    if not torch.isfinite(logits).all() or not valid.gather(1,top).all():raise ValueError('Invalid native selection or logits.')
    native=torch.zeros((b,n),device=top.device,dtype=torch.bool).scatter_(1,top,True)
    eligible=valid&~native
    score=logits.max(-1).values.masked_fill(~eligible,-torch.inf)
    extra=torch.argsort(score,dim=-1,descending=True,stable=True)[:,:300]
    if not eligible.gather(1,extra).all():raise ValueError('Insufficient valid non-native candidates.')
    return torch.cat([top,extra],1)

@torch.no_grad()
def compact_pool(boxes,logits,table):
    b,n,_=boxes.shape
    if n!=600 or table.shape!=(b,600) or logits.shape!=(b,600,10):raise ValueError('Fixed600 candidate table required.')
    valid=torch.isfinite(boxes).all(-1)&(boxes[...,2:]>0).all(-1)
    if not valid[:,:300].all():raise ValueError('Native self boxes invalid.')
    relation=torch.stack([box_iou(box_convert(x[:300],'cxcywh','xyxy'),box_convert(x,'cxcywh','xyxy')) for x in boxes])
    eligible=valid[:,None,:]&(relation>=.5)&(table[:,None,:]!=table[:,:300,None])
    work=relation.masked_fill(~eligible,-torch.inf)
    ids=torch.argsort(table,dim=-1,stable=True)
    byscore=torch.argsort(logits.max(-1).values.gather(-1,ids),dim=-1,descending=True,stable=True)
    secondary=ids.gather(-1,byscore)[:,None,:].expand(-1,300,-1)
    ordered=work.gather(-1,secondary)
    if ordered.dtype==torch.float32:
        bits=ordered.contiguous().view(torch.int32).to(torch.int64)
        keys=bits*1024+(599-torch.arange(600,device=ordered.device,dtype=torch.int64))
        rank=keys.topk(8,dim=-1,sorted=True).indices
    else:rank=torch.argsort(ordered,dim=-1,descending=True,stable=True)[...,:8]
    val=ordered.gather(-1,rank);position=secondary.gather(-1,rank);mask=torch.isfinite(val)
    ix=table[:,None,:].expand(-1,300,-1).gather(-1,position)
    ix=torch.where(mask,ix,torch.full_like(ix,-1));aff=torch.where(mask,val,torch.zeros_like(val))
    self_aff=relation.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)
    if not (self_aff==1).all():raise ValueError('Self IoU must be1.')
    return torch.cat([table[:,:300,None],ix],-1),torch.cat([self_aff,aff],-1),torch.cat([torch.ones_like(self_aff,dtype=torch.bool),mask],-1)
