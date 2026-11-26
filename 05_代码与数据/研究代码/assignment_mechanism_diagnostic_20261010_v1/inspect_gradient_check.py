from pathlib import Path
import json
import numpy as np
import analyze_saved as a

p=a.OUT/'collection_seed1/frozen_panel_v1/endpoint12/image_000000'
d=np.load(p/'arrays.npz');g=np.load(p/'counterfactual_leaf_gradients.npz')
norm=a.read(p/'counterfactual_leaf_losses.json')['num_boxes_normalizer']
q=g['union_query'];gt=g['union_GT'];boxes=d['model.pred_boxes'][0,q];targets=d['target.0.boxes'][gt]
expected=a.analytic_gradient(boxes.astype(np.float64),targets.astype(np.float64),norm)
actual=g['union_gradient'][q]
bad=~np.isclose(expected,actual,rtol=.002,atol=.002)
rows=[]
for i,j in np.argwhere(bad)[:12]:
    rows.append(dict(query=int(q[i]),coordinate=int(j),expected=float(expected[i,j]),saved=float(actual[i,j]),pred_box=boxes[i].tolist(),target_box=targets[i].tolist(),pred_edges_FP64=a.xyxy(boxes[[i]].astype(np.float64))[0].tolist(),target_edges_FP64=a.xyxy(targets[[i]].astype(np.float64))[0].tolist(),pred_edges_FP32=a.xyxy(boxes[[i]])[0].tolist(),target_edges_FP32=a.xyxy(targets[[i]])[0].tolist()))
result=dict(status='INITIAL_NUMPY_ANALYTIC_VERIFICATION_FAILED_NO_MODEL_RETRY',bad_coordinates=int(bad.sum()),rows=rows,scope='Saved-array mathematical verifier; no model/optimizer execution. Source recording and download verification remain valid.')
(a.OUT/'initial_analytic_verification_failure.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
