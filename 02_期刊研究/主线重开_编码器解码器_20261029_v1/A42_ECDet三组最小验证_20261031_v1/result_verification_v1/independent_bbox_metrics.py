"""Independent NumPy recomputation for this pilot's bbox/no-crowd COCO metrics.

Does not import the training evaluator, torch, or pycocotools. COCO matching
conventions were checked against the archived installed cocoeval reference.
Only the actual maxDet=300 output is implemented; unsupported crowd/ignore GT
are rejected rather than approximated. Inputs are the frozen evaluated records.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,pathlib,time
import numpy as np

IOU=np.linspace(.5,.95,10)
RECALL=np.linspace(0,1,101)
AREAS=[(0,1e10),(0,1024),(1024,9216),(9216,1e10)]

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def matrix_iou(dt,gt):
    if not len(dt) or not len(gt):return np.zeros((len(dt),len(gt)),dtype=np.float64)
    d=np.asarray([v['bbox']for v in dt],dtype=np.float64)
    g=np.asarray([v['bbox']for v in gt],dtype=np.float64)
    lo=np.maximum(d[:,None,:2],g[None,:,:2])
    hi=np.minimum(d[:,None,:2]+d[:,None,2:],g[None,:,:2]+g[None,:,2:])
    wh=np.maximum(hi-lo,0);inter=wh[:,:,0]*wh[:,:,1]
    return inter/(d[:,None,2]*d[:,None,3]+g[None,:,2]*g[None,:,3]-inter)

def evaluate_image(gt,dt,ious,area):
    ga=np.asarray([v['area']for v in gt],dtype=np.float64)
    ignore=(ga<area[0])|(ga>area[1])
    order=np.argsort(ignore,kind='stable');ignore=ignore[order];ious=ious[:,order]
    ng=len(gt);nd=len(dt)
    matches=np.zeros((len(IOU),nd),dtype=bool)
    di=np.zeros_like(matches)
    for ti,threshold in enumerate(IOU):
        taken=np.zeros(ng,dtype=bool)
        if not ng:continue
        for j in np.flatnonzero(np.any(ious>=min(threshold,1-1e-10),axis=1)):
            allowed=(ious[j]>=min(threshold,1-1e-10))&~taken
            regular=allowed&~ignore
            candidates=np.flatnonzero(regular if regular.any() else allowed)
            if not len(candidates):continue
            values=ious[j,candidates]
            # The original greedy scan keeps the last GT on an exact IoU tie.
            k=int(candidates[np.flatnonzero(values==values.max())[-1]])
            taken[k]=True;matches[ti,j]=True;di[ti,j]=ignore[k]
    da=np.asarray([v['bbox'][2]*v['bbox'][3]for v in dt],dtype=np.float64)
    di|=(~matches)&((da<area[0])|(da>area[1]))[None,:]
    return matches,di,int(np.count_nonzero(~ignore))

def recompute(gt,predictions):
    images=sorted(v['id']for v in gt['images'])
    assert len(set(images))==len(images)
    categories=sorted(v['id']for v in gt['categories'])
    assert categories==list(range(1,11)),categories
    image_set=set(images);category_set=set(categories)
    groups_g=collections.defaultdict(list);groups_d=collections.defaultdict(list)
    gtids=set()
    for v in gt['annotations']:
        assert v['id']>0 and v['id']not in gtids;gtids.add(v['id'])
        assert not v.get('iscrowd',0)and not v.get('ignore',0)
        assert v['image_id']in image_set and v['category_id']in category_set
        box=np.asarray(v['bbox'],dtype=np.float64)
        assert np.isfinite(box).all()and box[2]>0 and box[3]>0
        assert v['area']==float(box[2]*box[3])
        groups_g[v['image_id'],v['category_id']].append(v)
    counts=collections.Counter()
    for v in predictions:
        assert v['image_id']in image_set and v['category_id']in category_set
        box=np.asarray(v['bbox'],dtype=np.float64)
        assert np.isfinite(box).all()and box[2]>0 and box[3]>0
        assert np.isfinite(v['score'])and 0<=v['score']<=1
        groups_d[v['image_id'],v['category_id']].append(v);counts[v['image_id']]+=1
    assert max(counts.values(),default=0)<=300
    precision=np.full((len(IOU),len(RECALL),len(categories),len(AREAS)),-1.,dtype=np.float64)
    recall=np.full((len(IOU),len(categories),len(AREAS)),-1.,dtype=np.float64)
    for ci,category in enumerate(categories):
        cases=[]
        for image in images:
            gs=groups_g[image,category]
            ds=groups_d[image,category]
            if not gs and not ds:continue
            order=np.argsort([-v['score']for v in ds],kind='stable')[:300]
            ds=[ds[i]for i in order]
            cases.append((gs,ds,matrix_iou(ds,gs)))
        for ai,area in enumerate(AREAS):
            scores=[];mt=[];ig=[];npos=0
            for gs,ds,ious in cases:
                match,ignore,n=evaluate_image(gs,ds,ious,area)
                scores.extend(v['score']for v in ds);mt.append(match);ig.append(ignore);npos+=n
            if not npos:continue
            order=np.argsort(-np.asarray(scores),kind='stable')
            match=np.concatenate(mt,axis=1)[:,order]
            ignore=np.concatenate(ig,axis=1)[:,order]
            tp=np.cumsum(match&~ignore,axis=1,dtype=np.float64)
            fp=np.cumsum(~match&~ignore,axis=1,dtype=np.float64)
            for ti in range(len(IOU)):
                rc=tp[ti]/npos
                pr=tp[ti]/(tp[ti]+fp[ti]+np.spacing(1))
                q=np.zeros(len(RECALL),dtype=np.float64)
                if len(rc):
                    recall[ti,ci,ai]=rc[-1]
                    envelope=np.maximum.accumulate(pr[::-1])[::-1]
                    idx=np.searchsorted(rc,RECALL,side='left');valid=idx<len(envelope)
                    q[valid]=envelope[idx[valid]]
                else:recall[ti,ci,ai]=0
                precision[ti,:,ci,ai]=q
    def ap(area=0,iou=None):
        p=precision[:,:,:,area]
        if iou is not None:p=p[np.isclose(IOU,iou)]
        values=p[p>=0]
        return float(values.mean()*100)if len(values)else None
    z=recall[:,:,1];z=z[z>=0]
    result={'AP':ap(),'AP50':ap(iou=.5),'AP75':ap(iou=.75),'APs':ap(1),'APs75':ap(1,.75),'APm':ap(2),'APl':ap(3),'ARs':float(z.mean()*100)if len(z)else None,'evaluated_GT':len(gt['annotations']),'predictions_after_ignore':len(predictions),'max_detections':300}
    return result,precision,recall

def self_test():
    cats=[{'id':i}for i in range(1,11)]
    def g(boxes):return {'images':[{'id':1}],'categories':cats,'annotations':[{'id':i+1,'image_id':1,'category_id':1,'bbox':b,'area':b[2]*b[3],'iscrowd':0}for i,b in enumerate(boxes)]}
    def d(boxes):return [{'image_id':1,'category_id':1,'bbox':b,'score':1-i*.1}for i,b in enumerate(boxes)]
    p,*_=recompute(g([[0,0,10,10]]),d([[0,0,10,10]]));assert abs(p['AP']-100)<1e-12
    p,*_=recompute(g([[0,0,10,10]]),d([[30,0,10,10],[0,0,10,10]]));assert abs(p['AP']-50)<1e-12
    p,*_=recompute(g([[0,0,100,100]]),d([[0,0,100,100]]));assert p['APs']is None and abs(p['APl']-100)<1e-12
    p,*_=recompute(g([[0,0,10,10],[5,0,10,10]]),d([[2.5,0,10,10],[0,0,10,10]]));assert abs(p['AP50']-100)<1e-12
    p,*_=recompute(g([[0,0,10,10],[30,30,100,100]]),d([[30,30,100,100],[0,0,10,10]]));assert abs(p['APs']-100)<1e-12
    return 'PASS five independently specified metric cases'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gt',type=pathlib.Path);parser.add_argument('--predictions',type=pathlib.Path);parser.add_argument('--reference-metrics',type=pathlib.Path);parser.add_argument('--reference-matrix',type=pathlib.Path);parser.add_argument('--output',type=pathlib.Path);parser.add_argument('--self-test',action='store_true')
    a=parser.parse_args()
    if a.self_test:print(self_test(),flush=True)
    if not a.gt:return
    assert a.predictions and a.output
    t=time.monotonic();gt=json.loads(a.gt.read_text(encoding='utf-8'));pred=json.loads(a.predictions.read_text(encoding='utf-8'))
    result,p,r=recompute(gt,pred)
    verification={'status':'PASS','implementation':'standalone NumPy bbox/no-crowd greedy matching +101-point interpolation; no training-evaluator/pycocotools/torch import','metrics':result,'gt_sha256':sha(a.gt),'predictions_sha256':sha(a.predictions),'implementation_sha256':sha(__file__),'metric_reference_comparisons':{},'seconds':time.monotonic()-t}
    if a.reference_metrics:
        ref=json.loads(a.reference_metrics.read_text(encoding='utf-8'))
        for k,v in result.items():
            if k not in ref:continue
            delta=None if v is None or ref[k]is None else v-ref[k]
            assert (v is None and ref[k]is None)or(delta is not None and abs(delta)<=1e-10),(k,v,ref[k])
            verification['metric_reference_comparisons'][k]=delta
    if a.reference_matrix:
        ref=np.load(a.reference_matrix,allow_pickle=False)
        pr=ref['precision'][...,-1];rr=ref['recall'][...,-1]
        assert pr.shape==p.shape and rr.shape==r.shape
        dp=float(np.max(np.abs(pr-p)));dr=float(np.max(np.abs(rr-r)))
        assert dp<=1e-12 and dr<=1e-12,(dp,dr)
        verification['all_precision_cells_max_abs_difference']=dp
        verification['all_recall_cells_max_abs_difference']=dr
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(a.output.with_suffix('.npz'),precision=p,recall=r,iou_thresholds=IOU)
    print(json.dumps(verification),flush=True)

if __name__=='__main__':main()
