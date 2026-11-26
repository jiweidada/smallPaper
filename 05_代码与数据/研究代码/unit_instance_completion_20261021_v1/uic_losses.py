import torch, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from engine.deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

def _focal_cost(prob_pos, alpha=0.25, gamma=2.0):
    p = prob_pos
    neg=(1-alpha)*(p**gamma)*(-(1-p+1e-8).log())
    pos=alpha*((1-p)**gamma)*(-(p+1e-8).log())
    return pos-neg

def matcher_cost(new_boxes, new_logits, u_boxes, u_labels, num_classes,
                 cost_class=2.0,cost_bbox=5.0,cost_giou=2.0,alpha=0.25,gamma=2.0):
    """4 new queries x |U_W| GT cost (cxcywh normalized)."""
    from engine.deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
    prob=torch.sigmoid(new_logits)
    if u_boxes.shape[0]==0:
        return torch.zeros((new_boxes.shape[0],0))
    out_prob=prob[:,u_labels]
    cc=_focal_cost(out_prob,alpha,gamma)
    cb=torch.cdist(new_boxes,u_boxes,p=1)
    cg=-generalized_box_iou(box_cxcywh_to_xyxy(new_boxes),box_cxcywh_to_xyxy(u_boxes))
    C=cost_bbox*cb+cost_class*cc+cost_giou*cg
    return torch.nan_to_num(C,nan=1.0)

def hungarian_match(C, k):
    """return list of (row, col) matching min(k, min(N,M)) pairs by cost."""
    N,M=C.shape
    if N==0 or M==0: return []
    Cn=C.detach().cpu().numpy()
    r,c=linear_sum_assignment(Cn)
    pairs=list(zip(r.tolist(),c.tolist()))
    pairs.sort(key=lambda p:(Cn[p[0],p[1]], p[0], p[1]))
    pairs=pairs[:k]
    return pairs

def loss_labels_mal(new_logits, new_boxes, indices, num_boxes, num_classes, gamma=1.5, mal_alpha=None):
    """Faithful copy of DEIMCriterion.loss_labels_mal body but indices are (src_idx, tgt_idx) over the 4-query subset.
    new_logits:[Nq,C]; new_boxes:[Nq,4]; indices: list of (src,tgt) python;
    targets boxes/labels are the matched U_W GT, passed via _mal_target pack (see compute_losses)."""
    raise RuntimeError("use _mal_core")

def _mal_core(src_logits, src_boxes, tgt_logits_labels, indices, num_boxes, num_classes, gamma, mal_alpha):
    # indices: (src_idx tensor, tgt_idx tensor) aligned to tgt boxes/labels
    raise RuntimeError("internal")

def compute_losses(new_queries, u_boxes, u_labels, num_classes=10, denominator=None,
                   matcher_wd=None, alpha=0.75, gamma=1.5, mal_alpha=None, k=4,
                   denom_mode="external"):
    """new_queries: dict pred_logits [4,C], pred_boxes [4,4] (cxcywh normalized local coords).
    u_boxes:[M,4] cxcywh, u_labels:[M]. denominator: scalar (max(1,|U_W|) passed externally).
    Returns dict of losses + match info. Does NOT call DEIMCriterion.forward."""
    mwd = matcher_wd or dict(cost_class=2.0,cost_bbox=5.0,cost_giou=2.0,alpha=0.25,gamma=2.0)
    src_logits=new_queries["pred_logits"]; src_boxes=new_queries["pred_boxes"]
    Nq=src_logits.shape[0]; M=u_boxes.shape[0]
    C=matcher_cost(src_boxes,src_logits,u_boxes,u_labels,num_classes,**mwd)
    pairs=hungarian_match(C,min(k,M))
    src_idx=torch.tensor([p[0] for p in pairs],dtype=torch.long)
    tgt_idx=torch.tensor([p[1] for p in pairs],dtype=torch.long)
    den = float(denominator) if denominator is not None else float(max(1,M))
    losses={}
    if len(pairs)==0:
        # only classification negative term
        target=torch.zeros_like(src_logits)
        pred_score=torch.sigmoid(src_logits).detach()
        weight=pred_score.pow(gamma)*(1-target)+target   # mal_alpha None branch
        l=F.binary_cross_entropy_with_logits(src_logits,target,weight=weight,reduction='none')
        losses["loss_mal"]=l.mean(1).sum()*src_logits.shape[1]/den
        losses["loss_bbox"]=torch.tensor(0.0)
        losses["loss_giou"]=torch.tensor(0.0)
        losses["num_matched"]=0; losses["overflow"]=int(max(0,M-k)); losses["denominator"]=den
        losses["pairs"]=[]
        return losses
    # ---- MAL exactly as loss_labels_mal ----
    ious,_=box_iou(box_cxcywh_to_xyxy(src_boxes[src_idx]),box_cxcywh_to_xyxy(u_boxes[tgt_idx]))
    ious=torch.diag(ious).detach()
    # build target over full Nq with class assignment
    target_classes=torch.full((Nq,),num_classes,dtype=torch.long)
    target_classes[src_idx]=u_labels[tgt_idx]
    target=F.one_hot(target_classes,num_classes=num_classes+1)[...,:-1]
    target_score_o=torch.zeros(Nq,dtype=src_logits.dtype)
    target_score_o[src_idx]=ious.to(target_score_o.dtype)
    target_score=target_score_o.unsqueeze(-1)*target
    target_score=target_score.pow(gamma)
    pred_score=torch.sigmoid(src_logits).detach()
    if mal_alpha is not None:
        weight=mal_alpha*pred_score.pow(gamma)*(1-target)+target
    else:
        weight=pred_score.pow(gamma)*(1-target)+target
    lmal=F.binary_cross_entropy_with_logits(src_logits,target_score,weight=weight,reduction='none')
    losses["loss_mal"]=lmal.mean(1).sum()*src_logits.shape[1]/den
    # ---- boxes (L1 sum + GIoU) exactly as loss_boxes ----
    sb=src_boxes[src_idx]; tb=u_boxes[tgt_idx]
    l1=F.l1_loss(sb,tb,reduction='none').sum()/den
    giou=1-torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(sb),box_cxcywh_to_xyxy(tb)))
    losses["loss_bbox"]=l1
    losses["loss_giou"]=giou.sum()/den
    losses["num_matched"]=len(pairs)
    losses["overflow"]=int(max(0,M-k))
    losses["denominator"]=den
    losses["pairs"]=[list(map(int,p)) for p in pairs]
    return losses
