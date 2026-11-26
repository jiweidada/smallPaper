"""NumPy-only coordinate/score components for bounded cached diagnostics."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent;N=R.parent
A30=N/'A30_YOLO26双头原生读出核查_v1';D=A30/'downloaded_evidence'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def functions(path,names,environment):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];assert {x.name for x in nodes}==set(names)
    env=dict(environment);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);return {k:env[k] for k in names}
def region_functions():
    return functions(N/'A27_查询扩容同预算任务诊断_v1'/'o27_metrics.py',['matlab_round','ignore_integral','region_keep'],{'np':np})
def original_from_input(records,meta):
    g=meta['geometry'];s=g['ratio'];l,t=g['pad_ltrb'][:2];out=[]
    for x in records:
        a,b,w,h=x['bbox'];corners=np.asarray([(a-l)/s,(b-t)/s,(a+w-l)/s,(b+h-t)/s],dtype=np.float32).astype(np.float64)
        out.append(dict(image_id=x['image_id'],category_id=x['category_id'],bbox=[corners[0],corners[1],corners[2]-corners[0],corners[3]-corners[1]],score=x['score']))
    return out
def raw_original_corners(boxes,case):
    # NativeCUDA scalar divide is implemented through multiplication by the scalar reciprocal.
    g=case['geometry'];l,t=g['pad_ltrb'][:2];x=np.asarray(boxes,dtype=np.float32).copy();x-=np.asarray([l,t,l,t],dtype=np.float32);x*=np.float32(1.)/np.float32(g['ratio'])
    x[:,[0,2]]=np.clip(x[:,[0,2]],0,case['width']);x[:,[1,3]]=np.clip(x[:,[1,3]],0,case['height'])
    return x.astype(np.float64)
def probabilities(logits):return 1/(1+np.exp(-np.clip(np.asarray(logits,dtype=np.float64),-80,80)))
def bilinear(field,x,y):
    # field=C,H,W;pixel-index coordinates, borderreplication, not normalized grid coords.
    c,h,w=field.shape;x=np.clip(np.asarray(x,dtype=np.float64),0,w-1);y=np.clip(np.asarray(y,dtype=np.float64),0,h-1)
    x0=np.floor(x).astype(np.int64);y0=np.floor(y).astype(np.int64);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1);dx=x-x0;dy=y-y0
    return field[:,y0,x0]*(1-dx)*(1-dy)+field[:,y0,x1]*dx*(1-dy)+field[:,y1,x0]*(1-dx)*dy+field[:,y1,x1]*dx*dy
def score_flow(probs,anchors,boxes_original,case,strides,shapes,alpha):
    g=case['geometry'];l,t=g['pad_ltrb'][:2];s=g['ratio'];centers=(boxes_original[:,:2]+boxes_original[:,2:])/2*s+np.asarray([l,t]);positions=anchors+alpha*(centers-anchors);out=np.empty_like(probs);clamped=0;offset=0
    for stride,(h,w) in zip(strides,shapes):
        count=h*w;q=positions[offset:offset+count]/stride-.5;clamped+=int(np.sum((q[:,0]<0)|(q[:,0]>w-1)|(q[:,1]<0)|(q[:,1]>h-1)))
        field=probs[:,offset:offset+count].reshape(10,h,w);out[:,offset:offset+count]=bilinear(field,q[:,0],q[:,1]);offset+=count
    assert offset==8400 and np.isfinite(out).all()
    return out,clamped
def topk(scores):
    # Native two-stage topk definition with an explicit deterministic stable tie order.
    selected=np.argsort(-scores.max(axis=0),kind='stable')[:500];flat=scores[:,selected].T.reshape(-1);indices=np.argsort(-flat,kind='stable')[:500];a=selected[indices//10];c=indices%10;p=scores[c,a];keep=p>=.001
    return a[keep],c[keep],p[keep]
def prediction_records(image_id,corners,anchors,classes,scores):
    return [dict(image_id=image_id,category_id=int(c),bbox=[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],score=float(s)) for b,c,s in zip(corners[anchors],classes,scores)]
