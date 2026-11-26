from __future__ import annotations
import numpy as np


def iou_matrix(a, b):
    a=np.asarray(a,dtype=np.float64).reshape(-1,4)
    b=np.asarray(b,dtype=np.float64).reshape(-1,4)
    wh=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0)
    inter=wh.prod(2)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1)
    bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    union=aa[:,None]+bb[None,:]-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)


def ignored_fraction(boxes, regions):
    """Continuous rectangle-union area inside each box, not sum of overlaps."""
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    regions=np.asarray(regions,dtype=np.float64).reshape(-1,4)
    result=np.zeros(len(boxes),dtype=np.float64)
    for i,box in enumerate(boxes):
        area=np.maximum(box[2:]-box[:2],0).prod()
        if area<=0 or not len(regions): continue
        r=np.concatenate([np.maximum(regions[:,:2],box[:2]),np.minimum(regions[:,2:],box[2:])],1)
        r=r[np.all(r[:,2:]>r[:,:2],axis=1)]
        if not len(r): continue
        xs=np.unique(r[:,[0,2]])
        total=0.
        for left,right in zip(xs[:-1],xs[1:]):
            active=r[(r[:,0]<right)&(r[:,2]>left)]
            intervals=sorted((float(q[1]),float(q[3])) for q in active)
            end=-np.inf; length=0.
            for low,high in intervals:
                length+=max(0.,high-max(low,end)); end=max(end,high)
            total+=(right-left)*length
        result[i]=total/area
    return result


def maximum_cardinality(edges, n_right):
    """Deterministic augmenting paths; edge order is priority, not global min cost."""
    right=np.full(n_right,-1,dtype=np.int64)
    def augment(left,seen):
        for q in edges[left]:
            if seen[q]: continue
            seen[q]=True
            if right[q]<0 or augment(int(right[q]),seen):
                right[q]=left
                return True
        return False
    for left in range(len(edges)): augment(left,np.zeros(n_right,dtype=bool))
    matched=np.full(len(edges),-1,dtype=np.int64)
    for q,left in enumerate(right):
        if left>=0: matched[left]=q
    return matched


def geometric_assignment(gt, predictions, usable=None):
    overlaps=iou_matrix(gt,predictions)
    usable=np.ones(len(predictions),dtype=bool) if usable is None else np.asarray(usable,dtype=bool)
    edges=[]
    for row in overlaps:
        ids=np.flatnonzero((row>=.5)&usable).tolist()
        edges.append(sorted(ids,key=lambda q:(-float(row[q]),q)))
    return maximum_cardinality(edges,len(predictions)),overlaps


def prepare_annotations(raw, width, height):
    raw=np.asarray(raw,dtype=np.float64).reshape(-1,8)
    xyxy=np.c_[raw[:,:2],raw[:,:2]+raw[:,2:4]]/np.array([width,height,width,height])
    xyxy=np.clip(xyxy,0.,1.)
    nonzero=np.all(xyxy[:,2:]>xyxy[:,:2],axis=1)
    classes=raw[:,5].astype(np.int64)
    ignore_mask=nonzero&((classes==0)|(raw[:,4]<=0))
    ignores=xyxy[ignore_mask]
    all_real=nonzero&(classes>=1)&(classes<=11)
    eligible=all_real&(raw[:,4]>0)&(ignored_fraction(xyxy,ignores)<.5)
    ids=np.flatnonzero(eligible)
    primary=(classes[ids]>=1)&(classes[ids]<=10)&(raw[ids,2]*raw[ids,3]<=1024.)&(raw[ids,6]==0)&(raw[ids,7]==0)
    return dict(boxes=xyxy,classes=classes,ignore_regions=ignores,all_real_boxes=xyxy[all_real],
                eligible_ids=ids,eligible_boxes=xyxy[ids],primary_flags=primary)


def background_mask(predictions, annotations):
    predictions=np.asarray(predictions,dtype=np.float64)
    overlaps=iou_matrix(predictions,annotations['all_real_boxes'])
    clear=np.max(overlaps,axis=1)<.1 if overlaps.shape[1] else np.ones(len(predictions),dtype=bool)
    return clear&np.all(predictions[:,2:]>predictions[:,:2],axis=1)&(ignored_fraction(predictions,annotations['ignore_regions'])<.5)


def pair_queries(capture):
    boxes=np.clip(np.asarray(capture['query_boxes_xyxy'],dtype=np.float64),0,1)
    scores=np.asarray(capture['native_probabilities'],dtype=np.float64)
    annotation=prepare_annotations(capture['annotations'],int(capture['width']),int(capture['height']))
    q_ignore=ignored_fraction(boxes,annotation['ignore_regions'])
    usable=(q_ignore<.5)&np.all(boxes[:,2:]>boxes[:,:2],axis=1)
    assign,overlaps=geometric_assignment(annotation['eligible_boxes'],boxes,usable)
    labels=np.asarray(capture['native_labels'],dtype=np.int64)
    qids=np.asarray(capture['native_query_ids'],dtype=np.int64)
    top_scores=np.asarray(capture['native_scores'],dtype=np.float64)
    valid_native=(labels>=1)&(labels<=10)&(top_scores>=.5)&(q_ignore[qids]<.5)
    selected=np.flatnonzero(valid_native)
    supports=[]
    for row,gid in zip(overlaps,annotation['eligible_ids']):
        supports.append(any(labels[k]==annotation['classes'][gid] and row[qids[k]]>=.5 for k in selected))
    supports=np.asarray(supports,dtype=bool)
    positive_rows=np.flatnonzero(annotation['primary_flags']&(assign>=0))
    positive_q=assign[positive_rows]
    bg=np.flatnonzero(background_mask(boxes,annotation))
    wh=np.maximum(boxes[:,2:]-boxes[:,:2],1e-12)
    side=np.sqrt(wh.prod(1))*800.
    log_area=np.log2(np.maximum(wh.prod(1),1e-12))
    log_aspect=np.log2(wh[:,0]/wh[:,1])
    max_score=scores[:,1:11].max(1)
    sb=np.minimum(np.floor(max_score*20).astype(np.int64),19)
    ab=np.digitize(side,[8,16,32,64,128])
    rb=np.digitize(log_aspect,[-1,0,1])
    bins=np.stack([ab,rb,sb],1)
    edges=[]
    for q in positive_q:
        ids=bg[np.all(bins[bg]==bins[q],axis=1)&(np.abs(max_score[bg]-max_score[q])<=.025)]
        ids=sorted(ids.tolist(),key=lambda k:(abs(float(max_score[k]-max_score[q])),
                     abs(float(log_area[k]-log_area[q])),abs(float(log_aspect[k]-log_aspect[q])),k))
        edges.append(ids)
    negative_q=maximum_cardinality(edges,len(boxes))
    pairs=[]
    for j,r in enumerate(positive_rows):
        if negative_q[j]<0: continue
        gid=int(annotation['eligible_ids'][r]); q=int(positive_q[j]); neg=int(negative_q[j])
        pairs.append(dict(gt_index=gid,label=int(annotation['classes'][gid]),positive_query=q,
            background_query=neg,native_support=bool(supports[r]),iou=float(overlaps[r,q]),
            absolute_score_difference=float(abs(max_score[q]-max_score[neg]))))
    stats=dict(real_annotation_boxes=len(annotation['all_real_boxes']),eligible_annotation_boxes=len(assign),
        primary_small_visible_gt=int(annotation['primary_flags'].sum()),
        primary_geometry_covered=int(len(positive_rows)),background_query_candidates=len(bg),
        matched_pairs=len(pairs),unsupported_primary_pairs=sum(not r['native_support'] for r in pairs))
    geometry=np.c_[log_area,log_aspect,max_score,(boxes[:,:2]+boxes[:,2:])*.5]
    return pairs,stats,geometry


def auc(labels,scores):
    labels=np.asarray(labels,dtype=np.int64); scores=np.asarray(scores,dtype=np.float64)
    pos=labels==1; npos=int(pos.sum()); nneg=len(pos)-npos
    if npos==0 or nneg==0: return float('nan')
    order=np.argsort(scores,kind='stable'); s=scores[order]; p=pos[order]
    boundaries=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1,len(s)]
    below=0; wins=0.
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        np_=int(p[left:right].sum()); nn=right-left-np_
        wins+=np_*(below+.5*nn); below+=nn
    return wins/(npos*nneg)


def standardize_fit(x):
    x=np.asarray(x,dtype=np.float64); mean=x.mean(0); scale=np.maximum(x.std(0),1e-6)
    return (x-mean)/scale,mean,scale


def ridge_fit(x,y,regularization=1.):
    z,mean,scale=standardize_fit(x); y=np.asarray(y,dtype=np.float64)
    intercept=y.mean(0); rhs=z.T@(y-intercept)
    weight=np.linalg.solve(z.T@z+regularization*np.eye(z.shape[1]),rhs)
    return dict(mean=mean,scale=scale,weight=weight,intercept=intercept)


def ridge_predict(model,x):
    return ((np.asarray(x,dtype=np.float64)-model['mean'])/model['scale'])@model['weight']+model['intercept']


def pooling_coordinates(box,k):
    box=np.asarray(box,dtype=np.float64)
    t=(np.arange(k,dtype=np.float64)+.5)/k
    yy,xx=np.meshgrid(t,t,indexing='ij')
    return box[:2]+np.stack([xx,yy],-1)*(box[2:]-box[:2])


def bilinear_numpy(feature,coordinates):
    feature=np.asarray(feature,dtype=np.float64); c,h,w=feature.shape
    xy=np.asarray(coordinates,dtype=np.float64)
    x=np.clip(xy[...,0]*w-.5,0,w-1); y=np.clip(xy[...,1]*h-.5,0,h-1)
    x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,w-1); y1=np.minimum(y0+1,h-1)
    dx=x-x0; dy=y-y0
    return (feature[:,y0,x0]*(1-dx)*(1-dy)+feature[:,y0,x1]*dx*(1-dy)
            +feature[:,y1,x0]*(1-dx)*dy+feature[:,y1,x1]*dx*dy)


def pool_numpy(feature,box):
    box=np.clip(np.asarray(box,dtype=np.float64),0,1)
    inside=bilinear_numpy(feature,pooling_coordinates(box,7)).mean((1,2))
    center=(box[:2]+box[2:])*.5; wh=box[2:]-box[:2]
    t=(np.arange(15,dtype=np.float64)+.5)/15*2-1
    yy,xx=np.meshgrid(t,t,indexing='ij')
    coords=center+np.stack([xx,yy],-1)*wh
    ring=((np.abs(xx)>.5)|(np.abs(yy)>.5))&np.all((coords>=0)&(coords<=1),axis=-1)
    if not ring.any(): return inside,np.zeros_like(inside),0
    outside=bilinear_numpy(feature,coords)[:,ring].mean(1)
    return inside,inside-outside,int(ring.sum())
