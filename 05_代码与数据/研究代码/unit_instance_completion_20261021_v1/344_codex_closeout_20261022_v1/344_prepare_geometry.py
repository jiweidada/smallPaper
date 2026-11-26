"""Publish a separate geom342 version; preserve every approved 343 function."""
import importlib,difflib
C=importlib.import_module('344_context')
old=(C.S343/'343_geometry_v2/geom342.py').read_text(encoding='utf-8')
addition='''

# ---- 344 explicit raw nonpositive-width/height exception, authorized in 344. ----
def fullp_nonpositive_mask(raw_cxcywh,query_indices):
    b=torch.as_tensor(raw_cxcywh)
    q=torch.as_tensor(query_indices)
    assert b.device.type=='cpu' and b.shape==(300,4) and torch.isfinite(b).all()
    assert q.device.type=='cpu' and q.ndim==1 and q.dtype==torch.long
    assert ((q>=0)&(q<300)).all()
    return (b[q,2:]<=0).any(-1)


def intersecting_fullp_candidates(boxes,W,nonpositive_mask):
    b=torch.as_tensor(boxes);bad=torch.as_tensor(nonpositive_mask)
    assert b.device.type=='cpu' and b.ndim==2 and b.shape[1]==4 and torch.isfinite(b).all()
    assert bad.device.type=='cpu' and bad.dtype==torch.bool and bad.shape==(len(b),)
    # The mask is computed from raw w/h before conversion, never from score/GT.
    assert ((b[bad,2:]-b[bad,:2])<=0).any(-1).all()
    result=torch.zeros(len(b),dtype=torch.bool)
    result[~bad]=intersecting_candidates(b[~bad],W)
    return result


def fullp_m0_matrices(raw_cxcywh,query_indices,boxes,logits,gt_cxcywh,gt_labels,cand_classes):
    bad=fullp_nonpositive_mask(raw_cxcywh,query_indices)
    b=torch.as_tensor(boxes)
    assert b.shape==(len(query_indices),4) and torch.isfinite(b).all()
    expected=raw_query_cxcywh_to_640_xyxy(raw_cxcywh)[query_indices]
    assert torch.equal(b,expected), 'fullP must preserve native signed raw coordinates'
    _boxes(b[~bad])  # All abnormalities other than the authorized raw sign fail.
    cb=torch.cat(((b[:,:2]+b[:,2:])*.5,b[:,2:]-b[:,:2]),-1)/640.
    assert torch.isfinite(cb).all()
    n,m=len(b),len(gt_cxcywh)
    costs=torch.zeros(n,m);ious=torch.zeros(n,m);legal=torch.zeros(n,m,dtype=torch.bool)
    c,i,v=m0_matrices(cb[~bad],logits[~bad],gt_cxcywh,gt_labels,cand_classes[~bad])
    costs[~bad]=c;ious[~bad]=i;legal[~bad]=v
    assert not legal[bad].any() and (ious[bad]==0).all()
    return cb,costs,ious,legal,bad
'''
folder=C.HERE/'344_geometry';folder.mkdir()
with open(folder/'geom342.py','x',encoding='utf-8',newline='\n') as f:f.write(old+addition)
C.text('344_geom342_vs343.diff',''.join(difflib.unified_diff(old.splitlines(True),(old+addition).splitlines(True),fromfile='343_geometry_v2/geom342.py',tofile='344_geometry/geom342.py')))
