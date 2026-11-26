import torch, torch.nn.functional as F

def native_postprocess(logits, boxes, orig_size, num_classes=10, num_top_queries=300):
    """Exact reimplementation of PostProcessor.forward (use_focal_loss=True).
    logits:[B,Q,C] boxes:[B,Q,4] cxcywh (normalized). orig_size:[B,2] (h,w) or (w,h) per DEIM: repeat(1,2).
    Returns labels[B,k], boxes[B,k,4] xyxy scaled, scores[B,k]."""
    bbox=torchvision_box_convert(boxes)  # cxcywh->xyxy
    bbox=bbox*orig_size.repeat(1,2).unsqueeze(1)
    scores=torch.sigmoid(logits)
    scores,index=torch.topk(scores.flatten(1),num_top_queries,dim=-1)
    labels=index%num_classes
    boxidx=index//num_classes
    sel=bbox.gather(1, boxidx.unsqueeze(-1).repeat(1,1,bbox.shape[-1]))
    return labels, sel, scores

def torchvision_box_convert(boxes):
    # cxcywh -> xyxy
    cx,cy,w,h=boxes.unbind(-1)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],dim=-1)

def fuse_and_postprocess(global_logits, global_boxes, new_logits, new_boxes, orig_size,
                         num_classes=10, topq=300, score_thr=0.001, per_class_max=100,
                         return_pretop=False):
    """global:*[B,300,C]/[B,300,4]; new:*[B,4,C]/[B,4,4]; concat query dim -> sigmoid flatten top300
    -> strictly >score_thr -> per-class top per_class_max. No NMS/temperature/quota.
    Returns dict labels/boxes(orig-scale xyxy)/scores  per image list, plus optional pre-top."""
    B=global_logits.shape[0]
    logits=torch.cat([global_logits,new_logits],dim=1)     # [B,304,C]
    boxes=torch.cat([global_boxes,new_boxes],dim=1)
    bxyxy=torchvision_box_convert(boxes)*orig_size.repeat(1,2).unsqueeze(1)  # orig scale
    scores=torch.sigmoid(logits)
    flat=scores.flatten(1)                                  # [B,304*C]
    topv,topi=torch.topk(flat,topq,dim=-1)
    labels=topi%num_classes; boxidx=topi//num_classes
    sel=bxyxy.gather(1,boxidx.unsqueeze(-1).repeat(1,1,4))
    result_pre={"labels":labels,"boxes":sel,"scores":topv}
    out=[]
    for b in range(B):
        m=topv[b]>score_thr
        l=labels[b][m]; s=topv[b][m]; bx=sel[b][m]
        # per-class keep top per_class_max (already score-sorted desc from topk)
        keep=[]
        cnt={}
        for k in range(l.shape[0]):
            c=int(l[k]); 
            if cnt.get(c,0)<per_class_max:
                cnt[c]=cnt.get(c,0)+1; keep.append(k)
        keep=torch.tensor(keep,dtype=torch.long) if keep else torch.zeros(0,dtype=torch.long)
        out.append({"labels":l[keep],"scores":s[keep],"boxes":bx[keep]})
    return out, result_pre
