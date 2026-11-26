from pathlib import Path
import argparse,json,sys
import numpy as np
from semantic_math import (auc,background_mask,geometric_assignment,ignored_fraction,iou_matrix,
                           maximum_cardinality,pool_numpy,prepare_annotations,ridge_fit,ridge_predict)

sys.stdout.reconfigure(encoding='utf-8')
parser=argparse.ArgumentParser(); parser.add_argument('--torch-cpu',action='store_true'); parser.add_argument('--out',required=True)
args=parser.parse_args()
checks={}
boxes=np.array([[0,0,2,2],[1,1,3,3],[0,0,0,1.]])
np.testing.assert_allclose(iou_matrix(boxes,boxes[:1]).ravel(),[1,1/7,0],rtol=0,atol=1e-15)
checks['iou_hand_computed']=True
np.testing.assert_allclose(ignored_fraction([[0,0,2,2]],[[0,0,2,1.5],[0,0,1.5,2]]),[.9375],rtol=0,atol=1e-15)
checks['overlapping_ignore_union_not_sum']=True
assert maximum_cardinality([[0,1],[0]],2).tolist()==[1,0]
checks['augmenting_matching_recovers_second_assignment']=True
ann=prepare_annotations([[20,20,20,20,1,11,0,0]],100,100)
assert background_mask([[.2,.2,.4,.4],[.6,.6,.8,.8]],ann).tolist()==[False,True]
checks['unscored_class11_excluded_from_background']=True
assert auc([1,0,1,0],[.9,.8,.5,.5])==.625
assert auc([1,0,1,0],[1,1,1,1])==.5
assert auc([1,0],[0,1])==0
checks['auc_ties_and_reversal_hand_computed']=True
x=np.array([[-1.],[0.],[1.]])
model=ridge_fit(x,np.array([-1.,0.,1.]),1.)
np.testing.assert_allclose(ridge_predict(model,x),[-.75,0,.75],rtol=0,atol=1e-14)
np.testing.assert_allclose(ridge_predict(model,[[100.]]),[75.],rtol=0,atol=1e-12)
checks['ridge_hand_computed_and_training_only_scaling']=True
y,xg=np.mgrid[:4,:4]; ramp=(10*y+xg).astype(np.float64)[None]
mean,_,_=pool_numpy(ramp,[.25,.25,.5,.5])
np.testing.assert_allclose(mean,[11.],rtol=0,atol=1e-14)
mean,_,_=pool_numpy(ramp,[0,0,1,1])
np.testing.assert_allclose(mean,[16.5],rtol=0,atol=1e-14)
checks['bilinear_affine_field_known_means']=True
if args.torch_cpu:
    import torch
    from pool_tensor import pool_tensor
    feature=np.stack([10*y+xg,((xg+y)%2)*3],0).astype(np.float32)
    test_boxes=np.array([[.25,.25,.5,.5],[0,0,1,1],[0,0,.3,.2],[.99,.98,1,1],[.2,.3,.6,.7],[.5,.5,.5,.5]],dtype=np.float32)
    with torch.inference_mode(): observed=pool_tensor(torch.from_numpy(feature)[None],torch.from_numpy(test_boxes),2)
    reference=[pool_numpy(feature,b) for b in test_boxes]
    fine=np.stack([r[0] for r in reference]); contrast=np.stack([r[1] for r in reference]); counts=np.array([r[2] for r in reference])
    np.testing.assert_allclose(observed[0].numpy(),fine,rtol=0,atol=1e-5)
    np.testing.assert_allclose(observed[1].numpy(),contrast,rtol=0,atol=1e-5)
    np.testing.assert_array_equal(observed[2].numpy(),counts)
    checks['torch_cpu_grid_sample_vs_independent_numpy']=dict(fine_max_error=float(np.max(np.abs(observed[0].numpy()-fine))),contrast_max_error=float(np.max(np.abs(observed[1].numpy()-contrast))),boxes=len(test_boxes))
result=dict(status='PASS_ANALYTIC_ONLY_NO_AUTHOR_MODEL',numpy=np.__version__,torch_cpu=args.torch_cpu,checks=checks,
            author_detector_calls=0,probe_fits=1,probe_fit_scope='One synthetic analytic ridge fixture,not a fitted detector or data probe.')
with Path(args.out).open('x',encoding='utf-8') as output: json.dump(result,output,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
