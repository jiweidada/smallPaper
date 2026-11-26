"""Numpy-only diagnostic geometry. These are coverage measures, not AP."""
import numpy as np

def xyxy(boxes):
    a=np.asarray(boxes,dtype=np.float64)
    return np.concatenate([a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2],axis=1)

def iou_matrix(gt,pr):
    gt=np.asarray(gt,dtype=np.float64);pr=np.asarray(pr,dtype=np.float64)
    wh=np.maximum(0,np.minimum(gt[:,None,2:],pr[None,:,2:])-np.maximum(gt[:,None,:2],pr[None,:,:2]))
    inter=wh.prod(axis=-1)
    ga=np.maximum(0,gt[:,2:]-gt[:,:2]).prod(axis=-1)
    pa=np.maximum(0,pr[:,2:]-pr[:,:2]).prod(axis=-1)
    return inter/np.maximum(ga[:,None]+pa[None,:]-inter,1e-12)

def physical_query_matching(gt_boxes,gt_labels,boxes,labels,qids,threshold):
    """Maximum bipartite matching; a query cannot cover two GTs via class duplicates."""
    gt_boxes=np.asarray(gt_boxes).reshape(-1,4)
    gt_labels=np.asarray(gt_labels)
    labels=np.asarray(labels);qids=np.asarray(qids)
    overlap=iou_matrix(gt_boxes,boxes)
    eligible=(overlap>=threshold)&(gt_labels[:,None]==labels[None,:])
    edges=[np.unique(qids[row]).astype(int).tolist() for row in eligible]
    owner={}
    def augment(g,seen):
        for q in edges[g]:
            if q in seen:continue
            seen.add(q)
            if q not in owner or augment(owner[q],seen):
                owner[q]=g
                return True
        return False
    for g in range(len(gt_boxes)):augment(g,set())
    matched=np.full(len(gt_boxes),-1,dtype=np.int64)
    for q,g in owner.items():matched[g]=q
    return matched

def windows_xyxy(shape,input_size=(800,800)):
    h,w=shape;ih,iw=input_size
    y,x=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    return np.stack([x*iw/w,y*ih/h,(x+1)*iw/w,(y+1)*ih/h],axis=-1).reshape(-1,4)

def window_area_fraction(gt_boxes,shape):
    a=np.asarray(gt_boxes,dtype=np.float64).reshape(-1,4)
    w=windows_xyxy(shape)
    wh=np.maximum(0,np.minimum(a[:,None,2:],w[None,:,2:])-np.maximum(a[:,None,:2],w[None,:,:2]))
    area=np.maximum(0,a[:,2:]-a[:,:2]).prod(-1)
    return wh.prod(-1)/np.maximum(area[:,None],1e-12)

def masks_for_case(native,gt_boxes,tiny,case_index):
    native=np.asarray(native,dtype=bool);k=int(native.sum());n=native.size
    random=np.zeros(n,dtype=bool)
    random[np.random.default_rng(20261030+1000*case_index).choice(n,k,replace=False)]=True
    # Non-overlapping fixed windows: additive area coverage gain gives the same
    # deterministic order as residual greedy; this known GT-assisted control is
    # not a new algorithm and not a mathematical detection-performance upper bound.
    gain=window_area_fraction(gt_boxes,native.shape)[np.asarray(tiny,dtype=bool)].sum(0)
    order=np.lexsort((np.arange(n),-gain))
    coverage=np.zeros(n,dtype=bool);coverage[order[:k]]=True
    return {'REPLAY_NATIVE':native.copy(),'RANDOM_SAME_K':random.reshape(native.shape),
            'GT_COVERAGE_SAME_K':coverage.reshape(native.shape),'ALL_WINDOWS':np.ones_like(native)}

def coverage_stats(mask,gt_boxes,tiny):
    mask=np.asarray(mask,dtype=bool);a=np.asarray(gt_boxes,dtype=np.float64)
    tiny=np.asarray(tiny,dtype=bool)
    h,w=mask.shape;c=(a[:,:2]+a[:,2:])/2
    col=np.clip(np.floor(c[:,0]/800*w).astype(int),0,w-1)
    row=np.clip(np.floor(c[:,1]/800*h).astype(int),0,h-1)
    inside=mask[row,col]
    fraction=window_area_fraction(a,mask.shape)[:,mask.reshape(-1)].sum(1)
    return {'selected_windows':int(mask.sum()),'total_windows':int(mask.size),
      'tiny_gt':int(tiny.sum()),'tiny_center_covered':int(inside[tiny].sum()),
      'tiny_mean_area_covered':float(fraction[tiny].mean()) if tiny.any() else None,
      'tiny_uncovered_uids_indices':np.flatnonzero(tiny&~inside).tolist()}

def diagnostic_counts(gt_boxes,gt_labels,size,boxes,labels,qids):
    out={}
    for name,keep in [('all',np.ones(len(size),bool)),('lt8',size<8),('8to16',(size>=8)&(size<16))]:
        for threshold in [0.5,0.75]:
            m=physical_query_matching(gt_boxes[keep],gt_labels[keep],boxes,labels,qids,threshold)
            out[f'{name}_iou{int(100*threshold)}']={'gt':int(keep.sum()),'matched':int((m>=0).sum())}
    return out

def selfcheck():
    # Two different class detections from the same physical query cannot count
    # twice, even when both are perfect boxes. A second query restores capacity.
    gt=np.array([[0,0,10,10],[0,0,10,10]],float)
    labels=np.array([1,2])
    assert (physical_query_matching(gt,labels,gt,labels,np.array([7,7]),0.75)>=0).sum()==1
    assert (physical_query_matching(gt,labels,gt,labels,np.array([7,8]),0.75)>=0).sum()==2
    assert (physical_query_matching(gt,labels,np.array([[11,11,21,21]]),np.array([1]),np.array([0]),0.5)>=0).sum()==0
    # A box spanning two adjacent disjoint windows has exactly split area,
    # and all windows conserve object area.
    a=np.array([[350,0,450,100],[0,0,10,10]],float)
    f=window_area_fraction(a,(2,2))
    assert np.allclose(f.sum(1),1)
    assert np.allclose(f[0],[0.5,0.5,0,0])
    native=np.array([[True,False],[False,False]])
    masks=masks_for_case(native,a,np.ones(2,bool),0)
    assert all(int(masks[k].sum())==1 for k in ['REPLAY_NATIVE','RANDOM_SAME_K','GT_COVERAGE_SAME_K'])
    return {'status':'PASS_PHYSICAL_QUERY_CAPACITY_AND_DISJOINT_WINDOW_GEOMETRY','new_detector_forwards':0}

if __name__=='__main__':
    import json
    print(json.dumps(selfcheck()))
