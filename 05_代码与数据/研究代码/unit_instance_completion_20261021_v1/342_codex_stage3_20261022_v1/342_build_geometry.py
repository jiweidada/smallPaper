"""Copy approved geom342 unchanged, then add stage-3-only geometry helpers."""
import importlib,difflib
C=importlib.import_module('342_stage3_context')
old=(C.PREV/'geom342.py').read_text(encoding='utf-8')
addition=r'''

# ---- Stage 3 additions: the stage 1/2 functions above are byte-preserved. ----
import numpy as np
import torch.nn.functional as F
import stage1_cache_v3 as frozen_stage1

def norm_cxcywh_to_640_xyxy(boxes_norm, size=640):
    b=torch.as_tensor(boxes_norm).float()*float(size)
    assert b.device.type=='cpu' and b.ndim==2 and b.shape[1]==4
    assert torch.isfinite(b).all() and (b[:,2:]>0).all()
    cx,cy,w,h=b.unbind(-1)
    out=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)
    return _boxes(out)

def orig_xywh_to_640_xyxy(gt_xywh,orig_w,orig_h,size=640):
    a=torch.as_tensor(gt_xywh).float()
    assert a.ndim==2 and a.shape[1]==4 and torch.isfinite(a).all()
    assert (a[:,2:]>0).all() and orig_w>0 and orig_h>0
    x,y,w,h=a.unbind(-1);sx=float(size)/float(orig_w);sy=float(size)/float(orig_h)
    return _boxes(torch.stack([x*sx,y*sy,(x+w)*sx,(y+h)*sy],-1))

def full_plane_cxcywh(boxes_xyxy_640):
    return xyxy_to_cxcywh(boxes_xyxy_640)/640

def norm_cxcywh_to_xyxy(boxes):
    b=torch.as_tensor(boxes)
    assert b.ndim==2 and b.shape[1]==4 and torch.isfinite(b).all() and (b[:,2:]>0).all()
    cx,cy,w,h=b.unbind(-1)
    return _boxes(torch.stack([cx-.5*w,cy-.5*h,cx+.5*w,cy+.5*h],-1))

def select_window(final300_boxes_640):
    # Required frozen selector, including row-major ties and x/y=640 handling.
    return frozen_stage1.select_window(_boxes(final300_boxes_640))

def window_xyxy(rc):
    return assert_window(frozen_stage1.window_xyxy(rc))

def complete_mask(gt_xyxy_640,W):
    b=_boxes(gt_xyxy_640);w=assert_window(W)
    return torch.tensor([frozen_stage1.gt_complete_in_window(row.tolist(),w) for row in b],dtype=torch.bool)

def short_side(boxes_xyxy_640):
    b=_boxes(boxes_xyxy_640)
    return torch.minimum(b[:,2]-b[:,0],b[:,3]-b[:,1])

def kg_window(gt_xyxy_640,GW,W):
    b=_boxes(gt_xyxy_640);win=assert_window(W)
    centers=xyxy_to_cxcywh(b.double())[:,:2]
    x0,y0,x1,y1=map(float,win);ww=x1-x0;hh=y1-y0
    K=torch.zeros(len(b),256)
    for j in GW.nonzero().flatten().tolist():
        cx,cy=map(float,centers[j]);gx=(cx-x0)/ww*16-.5;gy=(cy-y0)/hh*16-.5
        ix=int(np.floor(gx));iy=int(np.floor(gy));fx=gx-ix;fy=gy-iy
        row=np.zeros(256,dtype=np.float64)
        for dy in (0,1):
            for dx in (0,1):
                weight=(fx if dx else 1-fx)*(fy if dy else 1-fy)
                xi=min(max(ix+dx,0),15);yi=min(max(iy+dy,0),15)
                row[yi*16+xi]+=weight
        assert row.sum()>0
        K[j]=torch.from_numpy(row/row.sum()).float()
    return K

def crop128(pil,W,orig_wh,size=128):
    ow,oh=orig_wh;assert pil.size==(ow,oh)
    img=np.asarray(pil.convert('RGB'),dtype=np.float32)/255.
    h,w=img.shape[:2];t=torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    x0,y0,x1,y1=map(float,assert_window(W));ww=x1-x0;hh=y1-y0
    u=torch.arange(size,dtype=torch.float32)+.5
    gx=(x0+u*ww/size)*(ow/640)-.5;gy=(y0+u*hh/size)*(oh/640)-.5
    xn=((gx+.5)/w*2-1).view(1,1,size).expand(1,size,size)
    yn=((gy+.5)/h*2-1).view(1,size,1).expand(1,size,size)
    grid=torch.stack([xn,yn],-1)
    return F.grid_sample(t,grid,mode='bilinear',padding_mode='border',align_corners=False)[0].contiguous()

def m0_matrices(cand_cxcywh,cand_logits,gt_cxcywh,gt_labels,cand_classes):
    # FP32 native formula; only native representation operations occur here.
    from engine.deim.box_ops import generalized_box_iou
    cb=norm_cxcywh_to_xyxy(cand_cxcywh.float());gb=norm_cxcywh_to_xyxy(gt_cxcywh.float())
    lo=torch.maximum(cb[:,None,:2],gb[None,:,:2]);hi=torch.minimum(cb[:,None,2:],gb[None,:,2:])
    wh=(hi-lo).clamp_min(0);inter=wh[...,0]*wh[...,1]
    ca=(cb[:,2]-cb[:,0])*(cb[:,3]-cb[:,1]);ga=(gb[:,2]-gb[:,0])*(gb[:,3]-gb[:,1])
    iou=inter/(ca[:,None]+ga[None,:]-inter).clamp_min(1e-12)
    prob=cand_logits.float().sigmoid()[:,gt_labels]
    neg=.75*prob.pow(2)*(-(1-prob+1e-8).log())
    pos=.25*(1-prob).pow(2)*(-(prob+1e-8).log())
    cc=pos-neg
    C=5*torch.cdist(cand_cxcywh.float(),gt_cxcywh.float(),p=1)+2*cc-2*generalized_box_iou(cb,gb)
    assert torch.isfinite(C).all() and torch.isfinite(iou).all()
    valid=(cand_classes[:,None]==gt_labels[None,:])&(iou>=.5)
    return C,iou,valid
'''
new=old+addition
with open(C.HERE/'geom342.py','x',encoding='utf-8',newline='\n') as f:f.write(new)
C.text_out('342_geom342_阶段3扩展.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(C.PREV/'geom342.py'),tofile=str(C.HERE/'geom342.py'))))
print('stage3 geom342 created; original stage1/2 source retained verbatim')
