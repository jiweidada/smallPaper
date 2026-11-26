"""342 shared CPU geometry. All outputs retain unclipped xyxy edges.

The enclosing 342 directory is a new version; the pre-existing geom342.py is
preserved. geometry_cost returns [B,256,J], including B=1 for [J,4] input.
"""
import importlib
_ctx = importlib.import_module('342_context')
import torch
core311 = importlib.import_module('core')

def _boxes(boxes):
    b = torch.as_tensor(boxes)
    if not b.is_floating_point():
        b = b.float()
    assert b.device.type == 'cpu' and b.ndim in (2, 3) and b.shape[-1] == 4
    assert bool(torch.isfinite(b).all()), 'non-finite box'
    assert bool((b[..., 2:] > b[..., :2]).all()), 'expected positive xyxy edges'
    return b

def assert_window(window):
    w = torch.as_tensor(window, dtype=torch.float32)
    assert w.device.type == 'cpu' and w.shape == (4,) and bool(torch.isfinite(w).all())
    assert bool((w[2:] > w[:2]).all()), 'invalid xyxy window'
    assert bool(((w >= 0) & (w <= 640)).all()), 'window outside 640 plane'
    return w

def win_norm_xyxy(boxes_640_xyxy, window_xyxy_640):
    b = _boxes(boxes_640_xyxy)
    w = assert_window(window_xyxy_640).to(b.dtype)
    origin = w[:2].repeat(2)
    extent = (w[2:] - w[:2]).repeat(2)
    out = (b - origin) / extent
    _boxes(out)
    return out

def xyxy_to_cxcywh(boxes):
    b = _boxes(boxes)
    return torch.cat(((b[..., :2] + b[..., 2:]) * .5, b[..., 2:] - b[..., :2]), -1)

def win_norm_cxcywh(boxes_640_xyxy, window_xyxy_640):
    return xyxy_to_cxcywh(win_norm_xyxy(boxes_640_xyxy, window_xyxy_640))

def win_norm_to_window_pixels(boxes_win_norm_xyxy, window_xyxy_640, output_size=None):
    b = _boxes(boxes_win_norm_xyxy)
    w = assert_window(window_xyxy_640).to(b.dtype)
    extent = w[2:] - w[:2] if output_size is None else b.new_tensor(output_size)
    assert extent.shape == (2,) and bool((extent > 0).all())
    return b * extent.repeat(2)

def grid_xy(dtype=torch.float32):
    return core311.grid_positions(dtype=dtype)

def geometry_cost(boxes_win_norm_xyxy, grid_xy=None):
    b = _boxes(boxes_win_norm_xyxy)
    # A project-specific sanity guard, not clipping or candidate filtering.
    # Native sigmoid boxes in a 64px window are within +/-15 normalized units.
    # Out-of-window boxes remain legal. Extremely small custom windows may need
    # a separately reviewed bound; they are not silently accepted here.
    assert bool((b.abs() < 32).all()), 'pixel-space input or unreviewed coordinate range'
    if b.ndim == 2:
        b = b.unsqueeze(0)
    xy = core311.grid_positions(dtype=b.dtype) if grid_xy is None else grid_xy.to(b.dtype)
    assert xy.device.type == 'cpu' and xy.shape == (256, 2)
    assert bool(torch.isfinite(xy).all()) and bool(((xy > 0) & (xy < 1)).all())
    assert torch.equal(xy, core311.grid_positions(dtype=b.dtype)), 'grid must be frozen 311_v2 grid'
    lower, upper = b[..., :2], b[..., 2:]
    outside = (lower[:, None] - xy[None, :, None]).clamp_min(0)
    outside = outside + (xy[None, :, None] - upper[:, None]).clamp_min(0)
    result = (outside / (upper - lower)[:, None]).sum(-1)
    assert bool(torch.isfinite(result).all())
    return result

def intersecting_candidates(fullp_boxes_640, window_xyxy_640):
    b = _boxes(fullp_boxes_640)
    assert b.ndim == 2
    w = assert_window(window_xyxy_640).to(b.dtype)
    lower = torch.maximum(b[:, :2], w[:2])
    upper = torch.minimum(b[:, 2:], w[2:])
    return (upper > lower).all(-1)


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


# Raw native query arrays include queries not retained by fullP. Their coordinates
# are preserved even if decoded w/h are nonpositive. This does not validate them
# as candidate boxes, nor change the frozen window selector or candidate filter.
def raw_query_cxcywh_to_640_xyxy(boxes_norm):
    b=torch.as_tensor(boxes_norm).float()*640.
    assert b.device.type=='cpu' and b.shape==(300,4) and torch.isfinite(b).all()
    cx,cy,w,h=b.unbind(-1)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)

def select_raw_query_window(raw_edges_640):
    b=torch.as_tensor(raw_edges_640)
    assert b.device.type=='cpu' and b.shape==(300,4) and torch.isfinite(b).all()
    return frozen_stage1.select_window(b)


# ---- 343 inverse transforms. Existing approved geometry text is unchanged. ----
def _orig_wh(orig_wh,dtype=torch.float32):
    wh=torch.as_tensor(orig_wh,dtype=dtype)
    assert wh.shape==(2,) and torch.isfinite(wh).all() and (wh>0).all()
    return wh

def win_norm_xyxy_to_640(boxes,W):
    b=_boxes(boxes);w=assert_window(W).to(b.dtype)
    return _boxes(b*(w[2:]-w[:2]).repeat(2)+w[:2].repeat(2))

def plane640_to_orig_xyxy(boxes,orig_wh):
    b=_boxes(boxes);wh=_orig_wh(orig_wh,b.dtype)
    return _boxes(b*(wh/640).repeat(2))

def orig_xyxy_to_plane640(boxes,orig_wh):
    b=_boxes(boxes);wh=_orig_wh(orig_wh,b.dtype)
    return _boxes(b*(640/wh).repeat(2))

def local_cxcywh_to_orig_xyxy(local_boxes,W,orig_wh):
    b=torch.as_tensor(local_boxes)
    assert b.ndim==2 and b.shape[-1]==4 and torch.isfinite(b).all()
    assert ((b>=0)&(b<=1)).all() and (b[:,2:]>0).all(), 'expected sigmoid window-normalized cxcywh'
    local_edges=norm_cxcywh_to_xyxy(b)
    return plane640_to_orig_xyxy(win_norm_xyxy_to_640(local_edges,W),orig_wh)

def orig_xyxy_to_local_cxcywh(boxes,W,orig_wh):
    return win_norm_cxcywh(orig_xyxy_to_plane640(boxes,orig_wh),W)

def global_queries_to_orig_xyxy(raw_cxcywh,orig_wh):
    # Native PostProcessor operation order, including unselected raw queries.
    b=torch.as_tensor(raw_cxcywh)
    assert b.shape==(300,4) and torch.isfinite(b).all()
    cx,cy,w,h=b.unbind(-1)
    edges=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)
    return edges*_orig_wh(orig_wh,b.dtype).repeat(2)

def orig_xyxy_to_coco_xywh(boxes):
    b=_boxes(boxes)
    return torch.cat((b[:,:2],b[:,2:]-b[:,:2]),-1)

def pairwise_iou(a,b,allow_raw_a=False):
    a=torch.as_tensor(a);b=_boxes(b)
    assert a.ndim==2 and a.shape[1]==4 and torch.isfinite(a).all()
    if not allow_raw_a:_boxes(a)
    lower=torch.maximum(a[:,None,:2],b[None,:,:2]);upper=torch.minimum(a[:,None,2:],b[None,:,2:])
    wh=(upper-lower).clamp_min(0);inter=wh[...,0]*wh[...,1]
    aa=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1);bb=(b[:,2:]-b[:,:2]).prod(-1)
    return inter/(aa[:,None]+bb[None,:]-inter).clamp_min(1e-12)
