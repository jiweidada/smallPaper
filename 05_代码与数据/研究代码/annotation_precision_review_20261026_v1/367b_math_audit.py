"""Independent vectorized audit of the 367 numerical grid test. No models/data inference."""
import numpy as np, random, json, itertools
from pathlib import Path
H=Path(__file__).resolve().parent
rng=random.Random(0)
E=np.array(list(itertools.product(np.linspace(-.5,.5,11),repeat=4)))
def scores(p,b):
    wh=np.maximum(0,np.minimum(p[2:],b[:,2:])-np.maximum(p[:2],b[:,:2]))
    I=wh[:,0]*wh[:,1]
    A=np.prod(p[2:]-p[:2]);B=np.prod(b[:,2:]-b[:,:2],axis=1);U=A+B-I
    iou=I/U
    C=np.prod(np.maximum(p[2:],b[:,2:])-np.minimum(p[:2],b[:,:2]),axis=1)
    return iou,iou-(C-U)/C
max_grid_over_projection=-float('inf');max_projection_over_grid=-float('inf');max_giou_gap=-float('inf');max_model_gap=0
worst_proj=None;worst_giou=None
for z in range(400):
    w=rng.uniform(1.5,8);h=rng.uniform(4,20);x=rng.uniform(0,10);y=rng.uniform(0,10)
    g=np.array([x,y,x+w,y+h]);px=x+rng.uniform(-3,3);py=y+rng.uniform(-3,3)
    p=np.array([px,py,px+w*rng.uniform(.4,2.5),py+h*rng.uniform(.6,1.6)])
    projected=np.clip(p,g-.5,g+.5)
    ip,gp=[v[0] for v in scores(p,projected[None])]
    edge=g+E;ie,ge=scores(p,edge);mi=float(ie.max());mg=float(ge.max())
    xywh=np.array([x,y,w,h])+E;xywh[:,2:]+=xywh[:,:2];mx=float(scores(p,xywh)[0].max())
    max_grid_over_projection=max(max_grid_over_projection,mi-ip)
    if ip-mi>max_projection_over_grid:
        max_projection_over_grid=float(ip-mi);worst_proj={'pred':p.tolist(),'gt':g.tolist(),'projection_iou':float(ip),'grid_iou':mi}
    if mg-gp>max_giou_gap:
        max_giou_gap=float(mg-gp);worst_giou={'pred':p.tolist(),'gt':g.tolist(),'projection_giou':float(gp),'better_grid_giou':mg,'better_grid_box':edge[int(ge.argmax())].tolist()}
    max_model_gap=max(max_model_gap,abs(mx-mi))
# Exact interpretable counterexamples; neither relies on identifying a continuous optimum by grid.
p=np.array([0.,0.,1.,1.]);g_proj=np.array([[9.5,0,10.5,1.]]);g_better=np.array([[9.5,0,11.5,1.]])
simple_giou={'pred':p.tolist(),'original_gt':[10,0,11,1],'delta':.5,'projection_box':g_proj[0].tolist(),'better_feasible_box':g_better[0].tolist(),'projection_giou':float(scores(p,g_proj)[1][0]),'better_feasible_giou':float(scores(p,g_better)[1][0])}
result={'cases':400,'grid_points_per_case':len(E),'edge_delta':.5,'max_grid_iou_minus_projection':max_grid_over_projection,'max_projection_iou_minus_grid':max_projection_over_grid,'grid_not_exact_projection_example':worst_proj,'max_giou_grid_advantage':max_giou_gap,'giou_counterexample':worst_giou,'max_xywh_grid_vs_edge_grid_difference':max_model_gap,'simple_giou_counterexample':simple_giou,'interpretation':'No grid counterexample to IoU projection. The grid need not contain the projected point. GIoU counterexample disproves projection; grid maximum is not a certified continuous maximum. Analytic IoU proof supplied separately.','model_forwards':0,'optimizer_steps':0}
with (H/'367b_math_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result),flush=True)
