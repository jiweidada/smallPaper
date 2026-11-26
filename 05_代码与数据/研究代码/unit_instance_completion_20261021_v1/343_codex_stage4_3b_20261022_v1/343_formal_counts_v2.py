"""Same frozen matching policy; class-index lists avoid repeated tensor scalars."""
import importlib
V=importlib.import_module('343_formal_counts');C=V.C;F=V.F;G=V.G;metrics=V.metrics
import numpy as np

def match(out,gt,labels,annotation_ids,threshold=.5):
    assert threshold in (.5,.75) and len(gt)==len(labels)==len(annotation_ids)
    assert len(set(annotation_ids))==len(annotation_ids)
    iou=G.pairwise_iou(out['boxes'].double(),gt.double()).numpy()
    byclass={};labs=labels.tolist();predlabs=out['labels'].tolist()
    for g,c in enumerate(labs):byclass.setdefault(c,[]).append(g)
    scores=out['scores'].tolist();flat=out['flat_index'].tolist()
    order=sorted(range(len(scores)),key=lambda j:(-scores[j],flat[j]))
    owner=np.full(len(scores),-1,dtype=np.int64);used=set()
    for j in order:
        eligible=[g for g in byclass.get(predlabs[j],[]) if g not in used and iou[j,g]>=threshold]
        if eligible:
            g=min(eligible,key=lambda g:(-float(iou[j,g]),annotation_ids[g]));owner[j]=g;used.add(g)
    return owner

def independent_match(out,gt,labels,aids,threshold):
    boxes=out['boxes'].double().tolist();targets=gt.double().tolist()
    score=out['scores'].tolist();flat=out['flat_index'].tolist();predlabs=out['labels'].tolist()
    classes={}
    for j,(label,target) in enumerate(zip(labels.tolist(),targets)):classes.setdefault(label,[]).append((j,target))
    order=sorted(range(len(boxes)),key=lambda k:(-score[k],flat[k]));owner=[-1]*len(boxes);used=[]
    for p in order:
        x,y,r,b=boxes[p];possibilities=[]
        for j,(u,v,s,t) in classes.get(predlabs[p],[]):
            if j in used:continue
            inter=max(0,min(r,s)-max(x,u))*max(0,min(b,t)-max(y,v))
            overlap=inter/max((r-x)*(b-y)+(s-u)*(t-v)-inter,1e-12)
            if overlap>=threshold:possibilities.append((overlap,aids[j],j))
        if possibilities:
            possibilities.sort(key=lambda x:(-x[0],x[1]));j=possibilities[0][2];owner[p]=j;used.append(j)
    return np.asarray(owner,dtype=np.int64)
