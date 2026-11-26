from pathlib import Path
from collections import deque,Counter
import hashlib,json,sys
import numpy as np
from semantic_math import prepare_annotations,geometric_assignment,background_mask

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;E=D/'downloaded_evidence';A=D/'analysis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def union_reference(box,regions):
    area=max(0,box[2]-box[0])*max(0,box[3]-box[1])
    if area==0 or len(regions)==0:return 0.
    r=np.array([[max(box[0],r[0]),max(box[1],r[1]),min(box[2],r[2]),min(box[3],r[3])] for r in regions])
    r=r[(r[:,2]>r[:,0])&(r[:,3]>r[:,1])]
    if not len(r):return 0.
    xs=np.unique(r[:,[0,2]]);ys=np.unique(r[:,[1,3]])
    x=(xs[:-1]+xs[1:])/2;y=(ys[:-1]+ys[1:])/2
    xx,yy=np.meshgrid(x,y,indexing='ij')
    covered=np.zeros(xx.shape,dtype=bool)
    for q in r:covered|=(xx>=q[0])&(xx<=q[2])&(yy>=q[1])&(yy<=q[3])
    return float((np.outer(np.diff(xs),np.diff(ys))*covered).sum()/area)
def overlaps_reference(gt,pred):
    result=np.zeros((len(gt),len(pred)))
    pa=np.maximum(0,pred[:,2]-pred[:,0])*np.maximum(0,pred[:,3]-pred[:,1])
    for i,b in enumerate(gt):
        widths=np.maximum(0,np.minimum(b[2],pred[:,2])-np.maximum(b[0],pred[:,0]))
        heights=np.maximum(0,np.minimum(b[3],pred[:,3])-np.maximum(b[1],pred[:,1]))
        intersection=widths*heights;area=max(0,b[2]-b[0])*max(0,b[3]-b[1])
        union=area+pa-intersection
        result[i]=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
    return result
def cardinality_reference(edges,n_right):
    """Hopcroft-Karp, separate from the frozen DFS augmenting-path implementation."""
    left=[-1]*len(edges);right=[-1]*n_right;dist=[]
    def bfs():
        nonlocal dist
        dist=[0 if left[u]<0 else -1 for u in range(len(edges))]
        queue=deque(u for u in range(len(edges)) if left[u]<0);found=False
        while queue:
            u=queue.popleft()
            for v in edges[u]:
                peer=right[v]
                if peer<0:found=True
                elif dist[peer]<0:dist[peer]=dist[u]+1;queue.append(peer)
        return found
    def dfs(u):
        for v in edges[u]:
            peer=right[v]
            if peer<0 or (dist[peer]==dist[u]+1 and dfs(peer)):
                left[u]=v;right[v]=u;return True
        dist[u]=-1;return False
    while bfs():
        for u in range(len(edges)):
            if left[u]<0:dfs(u)
    return sum(v>=0 for v in left)

assert cardinality_reference([[0,1],[0]],2)==2
records=read(A/'pair_records.json');source_sample=read(D/'combined_cases_FROZEN.json')
by_case={i:[] for i in range(224)}
for r in records['rows']:by_case[r['case_index']].append(r)
control=np.load(E/'first_control_memory_and_head.npz',allow_pickle=False)
weight=control['head_weight'].astype(np.float64);bias=control['head_bias'].astype(np.float64)
counts=Counter();train_images=set();val_images=set();train_prefixes=set();val_prefixes=set();max_head_error=0.;max_iou_error=0.
for row in source_sample['cases']:
    i=row['index'];path=Path(row['capture_path']) if row['capture_phase']=='A38' else E/row['capture_relative'];c=dict(np.load(path,allow_pickle=False));assert int(c['index'])==row['capture_index']
    assert c['file'].item()==row['file'] and c['split'].item()==row['split']
    raw=np.loadtxt(row['annotation_path'],delimiter=',',ndmin=2).reshape(-1,8)
    np.testing.assert_array_equal(c['annotations'],raw)
    boxes=np.clip(c['query_boxes_xyxy'].astype(np.float64),0,1);n=len(boxes)
    x=raw[:,0];y=raw[:,1];w=raw[:,2];h=raw[:,3]
    gt=np.clip(np.stack([x/row['width'],y/row['height'],(x+w)/row['width'],(y+h)/row['height']],1),0,1)
    nonzero=(gt[:,2]>gt[:,0])&(gt[:,3]>gt[:,1]);cls=raw[:,5].astype(int)
    ignored=gt[nonzero&((cls==0)|(raw[:,4]<=0))]
    qignored=np.array([union_reference(b,ignored) for b in boxes])
    gignored=np.array([union_reference(b,ignored) for b in gt])
    eligible=np.flatnonzero(nonzero&(cls>=1)&(cls<=11)&(raw[:,4]>0)&(gignored<.5))
    ref_overlap=overlaps_reference(gt,boxes)
    source_ann=prepare_annotations(raw,row['width'],row['height'])
    np.testing.assert_array_equal(source_ann['eligible_ids'],eligible)
    usable=np.all(boxes[:,2:]>boxes[:,:2],1)&(qignored<.5)
    assign,source_overlap=geometric_assignment(source_ann['eligible_boxes'],boxes,usable)
    np.testing.assert_allclose(source_overlap,ref_overlap[eligible],atol=2e-14,rtol=0)
    if source_overlap.size:max_iou_error=max(max_iou_error,float(np.max(np.abs(source_overlap-ref_overlap[eligible]))))
    fg_edges=[np.flatnonzero((ref_overlap[g]>=.5)&usable).tolist() for g in eligible]
    assert int((assign>=0).sum())==cardinality_reference(fg_edges,n)
    real=np.flatnonzero(nonzero&(cls>=1)&(cls<=11))
    clear=ref_overlap[real].max(0)<.1 if len(real) else np.ones(n,dtype=bool)
    bg_ref=clear&usable
    np.testing.assert_array_equal(background_mask(boxes,source_ann),bg_ref)
    primary=np.flatnonzero((cls[eligible]<=10)&(w[eligible]*h[eligible]<=1024)&(raw[eligible,6]==0)&(raw[eligible,7]==0)&(assign>=0))
    posq=assign[primary];bg=np.flatnonzero(bg_ref)
    wh=np.maximum(boxes[:,2:]-boxes[:,:2],1e-12);area=wh.prod(1);ratio=np.log2(wh[:,0]/wh[:,1])
    score=c['native_probabilities'][:,1:11].astype(np.float64).max(1)
    bins=np.stack([np.digitize(np.sqrt(area)*800,[8,16,32,64,128]),np.digitize(ratio,[-1,0,1]),np.minimum(np.floor(score*20).astype(int),19)],1)
    edges=[bg[(bins[bg]==bins[q]).all(1)&(np.abs(score[bg]-score[q])<=.025)].tolist() for q in posq]
    assert cardinality_reference(edges,n)==len(by_case[i])
    assert len({r['positive_query'] for r in by_case[i]})==len(by_case[i])
    assert len({r['background_query'] for r in by_case[i]})==len(by_case[i])
    labels=c['native_labels'];qids=c['native_query_ids'];nscores=c['native_scores']
    np.testing.assert_array_equal(nscores,c['native_probabilities'][qids,labels])
    np.testing.assert_array_equal(nscores,np.sort(c['native_probabilities'].ravel())[-n:][::-1])
    for r in by_case[i]:
        g=r['gt_index'];q=r['positive_query'];b=r['background_query']
        assert int(cls[g])==r['label'] and 1<=cls[g]<=10 and w[g]*h[g]<=1024 and raw[g,6]==raw[g,7]==0
        assert g in eligible and ref_overlap[g,q]>=.5 and usable[q] and bg_ref[b]
        np.testing.assert_array_equal(bins[q],bins[b]);assert abs(score[q]-score[b])<=.025
        native=((labels==cls[g])&(nscores>=.5)&(qignored[qids]<.5)&(ref_overlap[g,qids]>=.5)).any()
        assert bool(native)==r['native_support']
        counts[row['split']+'_pairs']+=1
        if row['split']=='train':train_images.add(i);train_prefixes.add(row['prefix'])
        elif not native:counts['primary_val_pairs']+=1;val_images.add(i);val_prefixes.add(row['prefix'])
    raw_ref=c['query_features'].astype(np.float64)@weight.T+bias
    np.testing.assert_allclose(raw_ref,c['raw_class_logits'],atol=1e-5,rtol=1e-6)
    max_head_error=max(max_head_error,float(np.max(np.abs(raw_ref-c['raw_class_logits']))))
result=read(A/'semantic_analysis_result.json')
assert counts['train_pairs']==result['eligibility']['train_pairs']==508
assert counts['primary_val_pairs']==result['eligibility']['primary_val_pairs']==199
assert len(train_images)==result['eligibility']['train_images_with_pairs']==100
assert len(val_images)==result['eligibility']['val_images_with_primary_pairs']==60
assert len(train_prefixes)==60 and len(val_prefixes)==45
fits=[json.loads(x) for x in (A/'probe_fit_entries.jsonl').read_text().splitlines()]
assert len(fits)==result['probe_fit_entries']==7
out=dict(status='PASS_INDEPENDENT_MATCHING_SUPPORT_AND_NATIVE_HEAD_AUDIT',cases=224,
    rows_checked=len(records['rows']),train_pairs=counts['train_pairs'],all_val_pairs=counts['val_pairs'],primary_val_pairs=199,
    train_images_with_pairs=100,val_images_with_primary_pairs=60,train_prefixes_with_pairs=60,val_prefixes_with_primary_pairs=45,
    iou_reference_max_error=max_iou_error,all_head_reconstruction_max_error=max_head_error,
    independent_checks=['2Dcell-unionignorefractions','Allquerybackgroundexclusionincludingclass11','HopcroftKarpgeometryandbackgroundcardinality',
        'Everychosenpairlabel/visibility/scorebin/native-support','AllnativeTopNscoreprovenance','All224queryLinearheadreconstructions'],
    data_probe_fits_observed=7,verifier_new_probe_fits=0,new_author_model_calls=0,mainline_validated=False)
with (D/'independent_support_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
