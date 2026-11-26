"""Independent saved-array/source-annotation analysis; no model/optimizer calls."""
from pathlib import Path
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import json
import numpy as np
import zlib

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/assignment_mechanism_diagnostic_20261010_v1'
RESULT=OUT/'local_analysis_verified_v2'
COCO_PACKED_SHA='660013db5600c93c328eafd305b1a284281b6432c670d86015fe6c2b3befe9ee'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))

def write(name,value):
    (RESULT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def xyxy(b):
    s=np.maximum(b[:,2:],0)/2
    return np.concatenate((b[:,:2]-s,b[:,:2]+s),axis=1)

def pair_geometry(b,t):
    b=xyxy(b);t=xyxy(t)
    wh=np.maximum(np.minimum(b[:,2:],t[:,2:])-np.maximum(b[:,:2],t[:,:2]),0)
    inter=wh[:,0]*wh[:,1]
    bw=b[:,2:]-b[:,:2];tw=t[:,2:]-t[:,:2]
    union=bw[:,0]*bw[:,1]+tw[:,0]*tw[:,1]-inter
    enclosure=np.maximum(np.maximum(b[:,2:],t[:,2:])-np.minimum(b[:,:2],t[:,:2]),0)
    area=enclosure[:,0]*enclosure[:,1]
    iou=inter/union
    return iou,iou-(area-union)/area

class Dual:
    """Native FP32 primal branches with independent FP64 derivative propagation."""
    def __init__(self,value,grad=None):
        self.v=np.asarray(value,dtype=np.float32)
        self.d=np.zeros(self.v.shape+(4,)) if grad is None else grad
    def __add__(self,other):
        o=other if isinstance(other,Dual) else Dual(np.full(self.v.shape,other))
        return Dual(self.v+o.v,self.d+o.d)
    __radd__=__add__
    def __neg__(self):return Dual(-self.v,-self.d)
    def __sub__(self,other):return self+-other if isinstance(other,Dual) else self+-float(other)
    def __mul__(self,other):
        o=other if isinstance(other,Dual) else Dual(np.full(self.v.shape,other))
        return Dual(self.v*o.v,self.d*o.v[...,None]+o.d*self.v[...,None])
    __rmul__=__mul__
    def __truediv__(self,other):
        o=other if isinstance(other,Dual) else Dual(np.full(self.v.shape,other))
        return Dual(self.v/o.v,(self.d*o.v[...,None]-o.d*self.v[...,None])/(o.v[...,None]**2))
    def clamp0(self):return Dual(np.maximum(self.v,0),self.d*(self.v>=0)[...,None])

def maximum(a,b):
    f=np.where(a.v>b.v,1.,np.where(a.v<b.v,0.,.5))
    return Dual(np.maximum(a.v,b.v),a.d*f[...,None]+b.d*(1-f)[...,None])

def minimum(a,b):return -maximum(-a,-b)

def analytic_gradient(boxes,targets,norm):
    b=[];t=[]
    for j in range(4):
        g=np.zeros_like(boxes,dtype=np.float64);g[:,j]=1
        b.append(Dual(boxes[:,j],g));t.append(Dual(targets[:,j]))
    def coordinates(p):
        w=p[2].clamp0();h=p[3].clamp0()
        return [p[0]-.5*w,p[1]-.5*h,p[0]+.5*w,p[1]+.5*h]
    b=coordinates(b);t=coordinates(t)
    w=(minimum(b[2],t[2])-maximum(b[0],t[0])).clamp0()
    h=(minimum(b[3],t[3])-maximum(b[1],t[1])).clamp0()
    inter=w*h
    union=(b[2]-b[0])*(b[3]-b[1])+(t[2]-t[0])*(t[3]-t[1])-inter
    ew=(maximum(b[2],t[2])-minimum(b[0],t[0])).clamp0()
    eh=(maximum(b[3],t[3])-minimum(b[1],t[1])).clamp0()
    enc=ew*eh
    giou=inter/union-(enc-union)/enc
    return (5*np.sign(boxes-targets)-2*giou.d)/norm

def raw_loss(b,t,norm):
    _,g=pair_geometry(b.astype(np.float64),t.astype(np.float64))
    return {'loss_bbox':float(np.abs(b.astype(np.float64)-t).sum()/norm),'loss_giou':float((1-g).sum()/norm)}

def pairs(data,event):
    p=event['prefix']+'.batch0'
    q=data[p+'.query_index'];g=data[p+'.GT_row_index']
    assert len(np.unique(q))==len(q)
    return q,g

def main():
    RESULT.mkdir(exist_ok=False)
    packed=base64.b64decode(''.join((OUT/'panel_original_coco.zlib.b64').read_text().split()),validate=True)
    assert hashlib.sha256(packed).hexdigest()==COCO_PACKED_SHA
    coco=json.loads(zlib.decompress(packed))
    assert coco['source_annotation_sha256']=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    write('original_panel_annotations.json',coco)
    image_meta={r['id']:r for r in coco['images']}
    annotations=defaultdict(list)
    for r in coco['annotations']:annotations[r['image_id']].append(r)
    all_conflicts=[];all_stages=[];summaries=[];sample_reports=[];max_grad_abs=0.;max_loss_abs=0.;verified_arrays=0;grad_checks=0
    identities=[]
    for seed in (1,2):
        capture=OUT/f'collection_seed{seed}/frozen_panel_v1'
        identity=read(capture/'panel_identity.json');identities.append(identity)
        assert identity['image_ids']==coco['image_ids']
        image_identity={r['image_id']:r for r in identity['images']}
        completion=read(capture/'completion.json')
        original_samples=read(capture/'sample_summaries.json')
        original_conflicts={(r['endpoint'],r['image_id'],r['query']):r for r in read(capture/'conflicts.json')}
        for endpoint in (12,132):
            samples=[];conflicts=[]
            for image_id in identity['image_ids']:
                p=capture/f'endpoint{endpoint}/image_{image_id:06d}'
                evidence=read(p/'evidence.json');assert evidence['status']=='CAPTURED' and not evidence['errors']
                with np.load(p/'arrays.npz',allow_pickle=False) as a:data={k:a[k] for k in a.files}
                assert len(data)==evidence['arrays'];verified_arrays+=len(data)
                assert all(np.isfinite(v).all() for v in data.values())
                rows=[r for r in annotations[image_id] if not r.get('iscrowd',0)]
                meta=image_meta[image_id]
                raw=np.asarray([r['bbox'] for r in rows],dtype=np.float64).reshape(-1,4);raw[:,2:]+=raw[:,:2]
                raw[:,0::2]=np.clip(raw[:,0::2],0,meta['width']);raw[:,1::2]=np.clip(raw[:,1::2],0,meta['height'])
                keep=(raw[:,2]>raw[:,0])&(raw[:,3]>raw[:,1]);rows=[r for r,k in zip(rows,keep) if k];raw=raw[keep]
                target=np.concatenate(((raw[:,:2]+raw[:,2:])/2,raw[:,2:]-raw[:,:2]),axis=1)/np.asarray([meta['width'],meta['height']]*2)
                assert np.allclose(target,data['target.0.boxes'],rtol=0,atol=1e-6)
                assert [r['id'] for r in rows]==data['target.0.verified_source_annotation_ids'].tolist()==image_identity[image_id]['annotation_ids']
                assert [r['category_id'] for r in rows]==data['target.0.labels'].tolist()
                small=np.asarray([float(r['area'])<=1024 for r in rows])
                normal=[e for e in evidence['events'] if e['kind']=='loss_call' and not e['is_DN']]
                matcher=[e for e in evidence['events'] if e['kind']=='matcher']
                votes=Counter()
                for e in matcher:
                    q,g=pairs(data,e);votes.update(zip(q.tolist(),g.tolist()))
                winners=defaultdict(list)
                for (q,g),count in votes.items():winners[q].append((g,count))
                for stage in sorted({e['stage'] for e in normal}):
                    ce=next(e for e in normal if e['stage']==stage and e['loss']=='mal')
                    re=next(e for e in normal if e['stage']==stage and e['loss']=='boxes')
                    cq,cg=pairs(data,ce);uq,ug=pairs(data,re)
                    own=dict(zip(cq.tolist(),cg.tolist()));union=dict(zip(uq.tolist(),ug.tolist()))
                    assert set(own)<=set(union) and set(union)==set(winners)
                    for q,g in union.items():assert votes[q,g]==max(count for _,count in winners[q])
                    me=next(e for e in matcher if e['stage']==stage)
                    mq,mg=pairs(data,me);assert np.array_equal(cq,mq) and np.array_equal(cg,mg)
                    conflict_q=[q for q,g in union.items() if q in own and own[q]!=g]
                    all_stages.append(dict(seed=seed,endpoint=endpoint,image_id=image_id,stage=stage,classification_pairs=len(own),regression_pairs=len(union),regression_only_queries=len(union)-len(own),GT_conflicts=len(conflict_q)))
                    if stage!='model':continue
                    b=data['model.pred_boxes'][0];t=data['target.0.boxes'];logits=data['model.pred_logits'][0]
                    own_iou,_=pair_geometry(b[cq],t[cg])
                    correct=logits[cq].argmax(-1)==data['target.0.labels'][cg]
                    protected=(own_iou>=.75)&correct&small[cg]
                    with np.load(p/'counterfactual_leaf_gradients.npz',allow_pickle=False) as a:leaf={k:a[k] for k in a.files}
                    norms=read(p/'counterfactual_leaf_losses.json');norm=norms['num_boxes_normalizer']
                    assert norm==len(uq)==re['num_boxes_normalizer']
                    assert norms['loss_weights']['loss_bbox']==5 and norms['loss_weights']['loss_giou']==2
                    anchored=np.asarray([own.get(q,g) for q,g in zip(uq,ug)])
                    assert np.array_equal(leaf['union_query'],uq) and np.array_equal(leaf['union_GT'],ug) and np.array_equal(leaf['anchored_GT'],anchored)
                    assert np.array_equal(leaf['own_query'],cq) and np.array_equal(leaf['own_GT'],cg)
                    gu=leaf['union_gradient'];ga=leaf['anchored_gradient'];go=leaf['own_gradient']
                    assert np.array_equal(gu[uq[ug==anchored]],ga[uq[ug==anchored]])
                    assert np.array_equal(ga[cq],go[cq])
                    for name,q,g,gradient in [('union',uq,ug,gu),('anchored_union',uq,anchored,ga),('own',cq,cg,go)]:
                        expected_loss=raw_loss(b[q],t[g],norm)
                        for key,value in expected_loss.items():
                            error=abs(value-norms[name]['raw'][key]);max_loss_abs=max(max_loss_abs,error)
                            assert error<2e-5,(seed,endpoint,image_id,name,key,error)
                        weighted=5*expected_loss['loss_bbox']+2*expected_loss['loss_giou']
                        assert abs(weighted-norms[name]['weighted'])<1e-4
                        expected_grad=analytic_gradient(b[q].astype(np.float64),t[g].astype(np.float64),norm)
                        max_grad_abs=max(max_grad_abs,float(np.max(np.abs(expected_grad-gradient[q]))))
                        assert np.allclose(expected_grad,gradient[q],rtol=.002,atol=.002),(seed,endpoint,image_id,name,float(np.max(np.abs(expected_grad-gradient[q]))))
                        nonmatched=np.setdiff1d(np.arange(300),q);assert np.all(gradient[nonmatched]==0)
                        grad_checks+=1
                    assert abs(norms['union']['raw']['loss_bbox']-re['native_unweighted_loss_scalars']['loss_bbox'])<2e-5
                    assert abs(norms['union']['raw']['loss_giou']-re['native_unweighted_loss_scalars']['loss_giou'])<2e-5
                    for q in conflict_q:
                        og=own[q];ug0=union[q]
                        iou1,_=pair_geometry(b[[q]],t[[og]]);iou2,_=pair_geometry(b[[q]],t[[ug0]])
                        dot=float(np.dot(gu[q],go[q]));cos=dot/float(np.linalg.norm(gu[q])*np.linalg.norm(go[q]))
                        r=original_conflicts[endpoint,image_id,q]
                        assert abs(float(iou1[0])-r['own_IoU'])<1e-6 and abs(float(iou2[0])-r['union_IoU'])<1e-6
                        assert np.isclose(dot,r['union_vs_own_bbox_gradient_dot'],rtol=1e-5,atol=1e-7)
                        assert (dot<0)==r['first_order_union_descent_increases_own_bbox_loss']
                        assert r['own_GT_annotation']==rows[og]['id'] and r['union_GT_annotation']==rows[ug0]['id']
                        assert r['own_class']==rows[og]['category_id'] and r['union_class']==rows[ug0]['category_id']
                        assert r['own_COCO_small']==bool(small[og])
                        assert r['own_strict_correct']==bool(iou1[0]>=.75 and logits[q].argmax()==rows[og]['category_id'])
                        assert abs(cos-r['union_vs_own_bbox_gradient_cosine'])<1e-5
                        r=dict(r,own_minus_union_IoU=float(iou1[0]-iou2[0]),original_union_vote_count=votes[q,ug0],final_own_vote_count=votes[q,og])
                        conflicts.append(r);all_conflicts.append(r)
                    s=dict(endpoint=endpoint,image_id=image_id,GT_rows=len(rows),GT_small_rows=int(small.sum()),final_own_pairs=len(cq),final_own_small_pairs=int(small[cg].sum()),final_strict_correct_small_pairs=int(protected.sum()),final_union_pairs=len(uq),final_GT_conflicts=len(conflict_q),final_strict_correct_conflicts=sum(r['own_strict_correct'] for r in conflicts if r['image_id']==image_id),unchanged_union_rows_gradient_equal=True)
                    original=next(s0 for s0 in original_samples if s0['endpoint']==endpoint and s0['image_id']==image_id)
                    for k,v in s.items():assert v==original[k],(k,v,original[k])
                    samples.append(s)
                    sample_reports.append(dict(seed=seed,**s,union_weighted_bbox_loss=norms['union']['weighted'],anchored_weighted_bbox_loss=norms['anchored_union']['weighted'],anchored_minus_union_weighted_bbox_loss=norms['anchored_union']['weighted']-norms['union']['weighted']))
                assert all_stages[-1]['seed']==seed
            strict=sum(s['final_strict_correct_small_pairs'] for s in samples)
            opposed=[r for r in conflicts if r['own_COCO_small'] and r['own_strict_correct'] and r['first_order_union_descent_increases_own_bbox_loss']]
            summary=dict(seed=seed,endpoint=endpoint,images=len(samples),GT_rows=sum(s['GT_rows'] for s in samples),GT_small_rows=sum(s['GT_small_rows'] for s in samples),classification_pairs=sum(s['final_own_pairs'] for s in samples),union_pairs=sum(s['final_union_pairs'] for s in samples),GT_conflicts=len(conflicts),cross_class_conflicts=sum(r['own_class']!=r['union_class'] for r in conflicts),COCO_small_conflicts=sum(r['own_COCO_small'] for r in conflicts),own_strict_correct_conflicts=sum(r['own_strict_correct'] for r in conflicts),own_strict_correct_small_conflicts=sum(r['own_strict_correct'] and r['own_COCO_small'] for r in conflicts),gradient_opposed_conflicts=sum(r['first_order_union_descent_increases_own_bbox_loss'] for r in conflicts),gradient_opposed_strict_correct_small_conflicts=len(opposed),strict_correct_small_denominator=strict,protected_small_opposed_fraction=len(opposed)/strict,own_target_higher_IoU_conflicts=sum(r['own_minus_union_IoU']>0 for r in conflicts),own_target_lower_IoU_conflicts=sum(r['own_minus_union_IoU']<0 for r in conflicts),mean_own_minus_union_IoU=float(np.mean([r['own_minus_union_IoU'] for r in conflicts])),anchored_minus_union_bbox_loss_mean=float(np.mean([s['anchored_minus_union_weighted_bbox_loss'] for s in sample_reports if s['seed']==seed and s['endpoint']==endpoint])))
            for k,v in completion['summaries'][str(endpoint)].items():assert summary[k]==v,(k,v,summary[k])
            summaries.append(summary)
    assert identities[0]==identities[1]
    write('summaries.json',summaries);write('all_conflicts.json',all_conflicts);write('all_stage_summaries.json',all_stages);write('all_sample_summaries.json',sample_reports)
    for name,rows in [('conflicts.csv',all_conflicts),('summaries.csv',summaries),('stage_summaries.csv',all_stages),('sample_summaries.csv',sample_reports)]:
        with (RESULT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    decision=dict(predeclared_mature_both_seed_boundary_met=all(s['gradient_opposed_strict_correct_small_conflicts']>0 for s in summaries if s['endpoint']==132),mechanism='Native majority localization union sometimes selects a different GT from final classification matching and opposes its leaf-box gradient.',priority='PROTECTED_SMALL_CASES_RARE_NO_LEADING_BOTTLENECK_CLAIM',candidate_status='Final-own-anchored union qualifies as a bounded mechanism hypothesis under the frozen protocol; efficacy and novelty remain unverified. No automatic training launch.',root_cause_established=False,AP_evidence=False,actual_training_history=False,detector_training_launched=False)
    write('decision.json',decision)
    write('verification.json',dict(status='PASS_SAVED_ARRAY_IDENTITY_GEOMETRY_LOSSES_ANALYTIC_GRADIENTS_AND_SUMMARIES',samples=128,arrays_verified=verified_arrays,leaf_gradient_sets_independently_checked=grad_checks,maximum_numpy_vs_native_loss_abs_error=max_loss_abs,maximum_analytic_vs_saved_gradient_abs_error=max_grad_abs,analytic_gradient_tolerance={'rtol':.002,'atol':.002},original_annotation_compressed_sha256=COCO_PACKED_SHA,all_native_union_winners_max_vote_verified=True,unchanged_query_gradients_exact=True,anchored_own_query_gradients_equal_own_exact=True,all_original_conflicts_and_completion_summaries_verified=True,model_forwards_added=0,model_backwards_added=0,optimizer_steps=0,training_epochs_added=0,verifier_repair='Initial pure FP64 primal falsely separated two edges which native FP32 rounds to an exact tie, selecting the wrong min/max derivative branch. Preserved failure identifies q64 y2 equality at image0 endpoint12. Now use native FP32 primal branch values and independent FP64 derivative propagation, keeping original tolerances; no model rerun.',limits='Independent array recomputation and analytic gradient checks; no re-execution of model captures, no parameter-gradient or AP inference. Original captured forward/loss quantities FP32; independent derivative propagation FP64 on matching FP32 primal values with explicit rounding tolerance.'))
    print(json.dumps(dict(status='PASS_LOCAL_ANALYSIS',summaries=summaries,decision=decision,max_gradient_abs_error=max_grad_abs),ensure_ascii=False))

if __name__=='__main__':main()
