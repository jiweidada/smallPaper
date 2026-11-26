"""Frozen formal GT matching/counts. No GT group is used before matching."""
import importlib
C=importlib.import_module('343_context');F=importlib.import_module('343_fusion')
import torch,numpy as np
G=C.load_geometry(C.HERE/'343_geometry_v2/geom342.py');F.G=G

def match(out,gt,labels,annotation_ids,threshold=.5):
    assert threshold in (.5,.75) and len(gt)==len(labels)==len(annotation_ids)
    assert len(set(annotation_ids))==len(annotation_ids)
    iou=G.pairwise_iou(out['boxes'].double(),gt.double()).numpy()
    order=sorted(range(len(out['scores'])),key=lambda j:(-float(out['scores'][j]),int(out['flat_index'][j])))
    owner=np.full(len(out['scores']),-1,dtype=np.int64);used=set()
    for j in order:
        eligible=[g for g in range(len(gt)) if g not in used and int(labels[g])==int(out['labels'][j]) and iou[j,g]>=threshold]
        if eligible:
            g=min(eligible,key=lambda g:(-float(iou[j,g]),annotation_ids[g]));owner[j]=g;used.add(g)
    return owner

def independent_match(out,gt,labels,aids,threshold):
    # Python scalar area/intersection and per-prediction greedy selection.
    boxes=out['boxes'].double().tolist();targets=gt.double().tolist();order=list(range(len(boxes)))
    order.sort(key=lambda k:(-float(out['scores'][k]),int(out['flat_index'][k])))
    owner=[-1]*len(boxes);used=[]
    for p in order:
        x,y,r,b=boxes[p];possibilities=[]
        for j,(u,v,s,t) in enumerate(targets):
            if j in used or int(labels[j])!=int(out['labels'][p]):continue
            intersection=max(0,min(r,s)-max(x,u))*max(0,min(b,t)-max(y,v))
            union=(r-x)*(b-y)+(s-u)*(t-v)-intersection
            overlap=intersection/max(union,1e-12)
            if overlap>=threshold:possibilities.append((overlap,aids[j],j))
        if possibilities:
            possibilities.sort(key=lambda item:(-item[0],item[1]));j=possibilities[0][2];owner[p]=j;used.append(j)
    return np.asarray(owner,dtype=np.int64)

def metrics(out,owner,baseout,baseowner,groups,aids,pre=None):
    selected=set(int(i) for i in owner if i>=0);base=set(int(i) for i in baseowner if i>=0)
    added=selected-base;lost=base-selected
    result={'TP':len(selected),'FP':int((owner<0).sum()),
        'GT_added_vs_global_baseline':len(added),'GT_lost_vs_global_baseline':len(lost),
        'GT_added_annotation_ids':[aids[g] for g in sorted(added)],'GT_lost_annotation_ids':[aids[g] for g in sorted(lost)],
        'matched_annotation_ids':[aids[g] for g in sorted(selected)]}
    for name,mask in groups.items():
        eligible=set(np.flatnonzero(np.asarray(mask,dtype=bool)).tolist())
        result[name+'_TP']=len(selected&eligible);result[name+'_added_TP']=len(added&eligible)
    result['B']=result['tiny_complete_Geo05_added_TP']
    current_global=set(int(f) for f in out['flat_index'] if int(f)<3000)
    baseline_global=set(int(f) for f in baseout['flat_index'])
    new=set(int(f) for f in out['flat_index'] if int(f)>=3000)
    result['new_candidate_instances_final_selected']=len(new)
    result['new_candidate_instances_final_unselected']=40-len(new)
    result['global_candidate_instances_lost_vs_baseline']=len(baseline_global-current_global)
    result['global_candidate_instances_lost_flat_indices']=sorted(baseline_global-current_global)
    if pre is not None:
        entered=set(int(f) for f in pre['flat_index'] if int(f)>=3000)
        result['new_candidate_instances_top300_selected']=len(entered)
        result['new_candidate_instances_top300_unselected']=40-len(entered)
        result['new_candidate_instances_removed_by_threshold_or_perclass100']=len(entered-new)
    return result
