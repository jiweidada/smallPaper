"""Read cached B4 arrays/predictions only. No model, optimizer or COCO rerun."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import numpy as np

P=Path(__file__).resolve().parent
D=P/'B_epoch04_remote_v1'
def load(f):return json.loads(f.read_text('utf-8'))
def quantiles(x):return [float(v) for v in np.quantile(x,[0,.1,.5,.9,1])]
def main():
    arm=D/'arm_B';gt=load(arm/'epoch04_RAW_GT.json');pred=load(arm/'epoch04_RAW_predictions.json')
    assert gt==load(P/'validation_GT_FROZEN_v1.json')
    assert len(gt['images'])==548 and len(gt['annotations'])==38602 and len(pred)==158221
    m=load(arm/'epoch04_RAW_metrics.json')
    arrays=np.load(arm/'epoch04_RAW_precision_recall.npz',allow_pickle=False)
    precision,recall,ths=[arrays[n] for n in ['precision','recall','iou_thresholds']]
    assert precision.shape==(10,101,10,4,3) and recall.shape==(10,10,4,3)
    def ap(area=0,iou=None,cls=None):
        v=precision[:,:,:,area,-1]
        if iou is not None:v=v[np.isclose(ths,iou)]
        if cls is not None:v=v[:,:,cls]
        v=v[v>=0];return float(v.mean()*100) if len(v) else None
    v=recall[:,:,1,-1];v=v[v>=0]
    derived={'AP':ap(),'AP50':ap(iou=.5),'AP75':ap(iou=.75),'APs':ap(1),'APs75':ap(1,.75),'APm':ap(2),'APl':ap(3),'ARs':float(v.mean()*100)}
    for k in derived:assert abs(derived[k]-m[k])<1e-12,k
    rows=[json.loads(s) for s in (D/'B_epoch04_evaluation_events_prefix.jsonl').read_text('utf-8').splitlines()]
    assert sum(s['event']=='model_forward_entered' for s in rows)==6609
    assert sum(s['event']=='model_forward_completed' for s in rows)==6609
    assert sum(s['event']=='optimizer_step_returned' for s in rows)==3236
    assert sum(s['event']=='optimizer_step_attempt' for s in rows)==3236
    assert [s['epoch'] for s in rows if s['event']=='training_epoch_completed']==[1,2,3,4]
    gs=defaultdict(list);ps=defaultdict(list)
    for x in gt['annotations']:gs[x['image_id']].append(x)
    for x in pred:ps[x['image_id']].append(x)
    assert set(ps).issubset({x['id'] for x in gt['images']})
    assert {x['category_id'] for x in pred}.issubset(set(range(1,11)))
    best=[];best_class=[];pred_sizes=[];gt_sizes=[];scores=[];unique=[];confusion=np.zeros((10,10),dtype=np.int64)
    for image in gt['images']:
        iid=image['id'];g=gs[iid];p=ps[iid]
        gxy=np.array([x['bbox'] for x in g],dtype=np.float64).reshape(-1,4)
        pxy=np.array([x['bbox'] for x in p],dtype=np.float64).reshape(-1,4)
        assert np.isfinite(pxy).all() and (pxy[:,2:]>0).all()
        scores.extend(x['score'] for x in p)
        unique.append(len({tuple(x['bbox']) for x in p}))
        scale=np.array([image['width'],image['height']])
        pred_sizes.extend((pxy[:,2:]/scale).tolist());gt_sizes.extend((gxy[:,2:]/scale).tolist())
        if not len(g):continue
        if not len(p):best.extend([0.]*len(g));best_class.extend([0.]*len(g));continue
        ga=gxy[:,2:].prod(1);pa=pxy[:,2:].prod(1)
        inter=np.maximum(0,np.minimum(gxy[:,None,:2]+gxy[:,None,2:],pxy[None,:,:2]+pxy[None,:,2:])-np.maximum(gxy[:,None,:2],pxy[None,:,:2])).prod(2)
        iou=inter/(ga[:,None]+pa[None,:]-inter+1e-12)
        gc=np.array([x['category_id'] for x in g]);pc=np.array([x['category_id'] for x in p])
        b=iou.max(1);bc=np.where(gc[:,None]==pc[None,:],iou,0).max(1)
        best.extend(b.tolist());best_class.extend(bc.tolist())
        closest=iou.argmax(1)
        for k in np.flatnonzero(b>=.5):confusion[gc[k]-1,pc[closest[k]]-1]+=1
    best=np.array(best);best_class=np.array(best_class);assert len(best)==38602
    scores=np.array(scores);assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
    out={'status':'PASS_CACHED_METRIC_AND_FORMAT_AUDIT','metrics_recomputed_from_saved_precision':derived,
         'GT_semantically_identical':True,'predictions':len(pred),'predicted_images':len(ps),
         'event_prefix':{'model_entries':6609,'model_completions':6609,'train_forwards':6472,'val_forwards':137,'updates_returned':3236,'epochs':4},
         'prediction_category_counts':dict(sorted(Counter(x['category_id'] for x in pred).items())),
         'GT_category_counts':dict(sorted(Counter(x['category_id'] for x in gt['annotations']).items())),
         'score_quantiles_min_p10_median_p90_max':quantiles(scores),
         'unique_predicted_boxes_per_image_quantiles':quantiles(unique),
         'prediction_normalized_width_quantiles':quantiles(np.array(pred_sizes)[:,0]),'prediction_normalized_height_quantiles':quantiles(np.array(pred_sizes)[:,1]),
         'GT_normalized_width_quantiles':quantiles(np.array(gt_sizes)[:,0]),'GT_normalized_height_quantiles':quantiles(np.array(gt_sizes)[:,1]),
         'GT_best_IoU_quantiles':quantiles(best),'GT_same_class_best_IoU_quantiles':quantiles(best_class),
         'coverage_upper_bound':{str(t):{'class_agnostic_GT_percent':float((best>=t).mean()*100),'class_aware_GT_percent':float((best_class>=t).mean()*100)} for t in [.1,.5,.75]},
         'coverage_scope':'Loose GT coverage; one prediction may cover several GT. This is neither COCO AP nor one-to-one recall. Uses already selected top300 after ignore filtering.',
         'best_geometry_confusion_IoU_at_least_05':confusion.tolist(),
         'per_category_AP_and_APs':{str(i+1):{'AP':ap(cls=i),'APs':ap(1,cls=i)} for i in range(10)},
         'additional_model_forwards':0,'additional_optimizer_updates':0,'additional_COCO_evaluate_calls':0,
         'performance_gate_changed':False,'training_code_changed':False,'causal_explanation_established':False}
    with (P/'B_epoch04_cached_audit_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:out[k] for k in ['status','metrics_recomputed_from_saved_precision','prediction_category_counts','score_quantiles_min_p10_median_p90_max','unique_predicted_boxes_per_image_quantiles','coverage_upper_bound','prediction_normalized_width_quantiles','GT_normalized_width_quantiles']}))


if __name__=='__main__':main()
