"""Extend the approved shared module in a new directory; old source read-only."""
import importlib,difflib
C=importlib.import_module('343_context')
base=C.S3/'342_geometry_stage3_v2'/'geom342.py'
old=base.read_text(encoding='utf-8')
extra=r'''

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
'''
folder=C.HERE/'343_geometry';folder.mkdir()
with open(folder/'geom342.py','x',encoding='utf-8',newline='\n') as f:f.write(old+extra)
C.text('343_geom342_反变换扩展.diff',''.join(difflib.unified_diff(old.splitlines(True),(old+extra).splitlines(True),fromfile=str(base),tofile=str(folder/'geom342.py'))))
print('created shared inverse-transform version')
