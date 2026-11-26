"""Frozen geometric proxies, valid-GT COCO-style@500, and paired-prefix decisions."""
from fractions import Fraction
import contextlib,io
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

ARMS=['original','pixels_N','pixels_B','state_N','mean_N','both_N','state_B','mean_B','both_B','normstate_N','normstate_B']
def ious(boxes,gt):
    boxes=np.asarray(boxes,dtype=np.float64);g=np.asarray(gt,dtype=np.float64)
    lo=np.maximum(boxes[:,:2],g[:2]);hi=np.minimum(boxes[:,2:],g[2:])
    intersection=np.maximum(hi-lo,0).prod(1)
    area=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1);ga=np.maximum(g[2:]-g[:2],0).prod()
    return intersection/np.maximum(area+ga-intersection,1e-12)
def proxies(output,case):
    b=np.asarray(output['pred_boxes'],dtype=np.float64).reshape(-1,4)*800
    assert b.shape==(500,4)
    xyxy=np.concatenate([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],1)
    a=[float(Fraction(x)) for x in case['target_box800_exact']]
    n=[float(Fraction(x)) for x in case['neighbour_box800_exact']]
    ia,ib=ious(xyxy,a),ious(xyxy,n)
    matrix=np.stack([ia,ib]);rows,cols=linear_sum_assignment(-matrix)
    logits=np.asarray(output['pred_logits'],dtype=np.float64).reshape(500,12)
    eligible=ia>=0.5
    return {'J':float(ia.max()),'P':float(matrix[rows,cols].sum()/2),
            'classification':float(expit(logits[eligible,int(case['category'])]).max()) if eligible.any() else 0.0}

def paired_decision(cases):
    assert len(cases) in [27,6]
    assert len({x['prefix'] for x in cases})==len(cases)
    vectors=[]
    for c in cases:
        a=c['arms'];assert set(a)==set(ARMS)
        vectors.append([a['both_N']['J']-a['both_B']['J'],a['both_N']['J']-a['original']['J'],
                        a['normstate_N']['J']-a['normstate_B']['J'],a['normstate_N']['P']-a['original']['P']])
    x=np.asarray(vectors,dtype=np.float64);assert np.isfinite(x).all()
    draw=np.random.default_rng(114021).integers(0,len(cases),size=(10000,len(cases)))
    means=x.mean(0);interval=np.percentile(x[draw].mean(1),[2.5,97.5],axis=0)
    gates=[means[0]>=.05 and interval[0,0]>0,means[1]>=.03 and interval[0,1]>0,
           means[2]>=.03 and interval[0,2]>0,means[3]>=0 and interval[0,3]>=-.02]
    return {'n':len(cases),'contrasts':{k:{'mean':float(means[i]),'CI95':interval[:,i].tolist(),'gate':bool(gates[i])} for i,k in enumerate(['D','G','R','H'])},
            'all_gates_pass':bool(all(gates)),'bootstrap_seed':114021,'bootstrap_replicates':10000,
            'scope':'Conditional localization proxy screening; not AP improvement, convergence or confirmed novel algorithm'}

def coco_metrics(gt_dataset,detections):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=gt_dataset;gt.createIndex()
        if detections:dt=gt.loadRes(detections)
        else:
            dt=COCO();dt.dataset={'images':gt_dataset['images'],'categories':gt_dataset['categories'],'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.maxDets=[1,10,500]
        ev.params.catIds=list(range(1,11));ev.params.imgIds=[r['id'] for r in gt_dataset['images']]
        ev.evaluate();ev.accumulate()
    precision=ev.eval['precision']
    assert precision.shape[0]==10 and precision.shape[-1]==3 and ev.params.maxDets[-1]==500
    def average(x):
        x=x[x>-1];return float(x.mean()) if x.size else -1.0
    return {'AP':average(precision[:,:,:,0,2]),'AP50':average(precision[0,:,:,0,2]),
       'APsmall':average(precision[:,:,:,1,2]),'maxDets':500,'valid_categories':list(range(1,11)),
       'area_scope':'original-pixel COCO small<32^2; distinct from resized800 area<64 cohort',
       'ignore_scope':'Only score>0/class1..10 positive clipped GT; official VisDrone ignored-region benchmark not reproduced'}

def learning_gate(metrics,cohort_outputs,cohort):
    values=[proxies(cohort_outputs[c['image']],c)['J'] for c in cohort if not c['filename_prefix_also_in_train']]
    assert len(values)==27
    high=sum(v>=.5 for v in values);low=sum(v<.5 for v in values)
    return {'qualified':bool(metrics['AP50']>=.2 and metrics['APsmall']>=.05 and high>=6 and low>=6),
       'AP50_floor':.2,'APsmall_floor':.05,'primary_J_ge05':high,'primary_J_lt05':low,
       'scope':'Pilot-qualification engineering floors, not proof of convergence or architecture failure'}
