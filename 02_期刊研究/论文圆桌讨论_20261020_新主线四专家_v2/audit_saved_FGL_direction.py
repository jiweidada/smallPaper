"""Exploratory output-space FGL/geometry directional check, development only.

No network execution or differentiation; GIoU coordinate derivatives use
checked finite differences. Output-space antagonism is not parameter conflict.
"""
import hashlib, json, tarfile
from pathlib import Path
import numpy as np
from audit_saved_bin_response import load_arrays, DATA, OUT, ROOT
from audit_saved_support import xyxy, ceiling, iou

def giou_loss(x,y):
    wh=np.maximum(0,np.minimum(x[...,2:],y[...,2:])-np.maximum(x[...,:2],y[...,:2]))
    inter=wh.prod(-1); union=(x[...,2:]-x[...,:2]).prod(-1)+(y[...,2:]-y[...,:2]).prod(-1)-inter
    enclosing=(np.maximum(x[...,2:],y[...,2:])-np.minimum(x[...,:2],y[...,:2])).prod(-1)
    return 1-inter/union+(enclosing-union)/enclosing

def geometry_derivative(x,y):
    cx=(x[...,0]+x[...,2])/2;cy=(x[...,1]+x[...,3])/2
    gt_cx=(y[...,0]+y[...,2])/2;gt_cy=(y[...,1]+y[...,3])/2
    a=np.sign(cx-gt_cx);b=np.sign(cy-gt_cy)
    c=np.sign((x[...,2]-x[...,0])-(y[...,2]-y[...,0]));d=np.sign((x[...,3]-x[...,1])-(y[...,3]-y[...,1]))
    l1=np.stack((a/2-c,b/2-d,a/2+c,b/2+d),-1)
    eps=np.minimum(1e-8,np.minimum((x[...,2:]-x[...,:2]).min(-1),(y[...,2:]-y[...,:2]).min(-1))*1e-5)
    assert np.all(eps>0)
    gradients=[]
    for ratio in [1.,.5]:
        grad=[]
        for e in range(4):
            delta=np.zeros_like(x);delta[...,e]=eps*ratio
            grad.append((giou_loss(x+delta,y)-giou_loss(x-delta,y))/(2*eps*ratio))
        gradients.append(np.stack(grad,-1))
    err=np.max(np.abs(gradients[0]-gradients[1])/(1+np.abs(gradients[1])))
    assert err<.002,err
    return 5*l1+2*gradients[1],float(err)

def main():
    spec={'scope':'BASE24 ordinary final native union; two seeds development only',
          'quantity':'dot of analytic FGL logit gradient with logit gradient of 5L1+2GIoU, ignoring common normalization',
          'FGL_weight':.15,'stop_gradient_IoU':True,
          'limitations':['not a network parameter gradient','omits MAL/LQE, DDF, aux and DN','not optimizer trajectory or AP','exploratory, no significance or acceptance gate'],
          'new_forwards':0,'new_backwards':0,'optimizer_updates':0,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'19_FGL方向探索口径.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
    result={'spec':spec,'summaries':[],'sources':{},'finite_difference_max_relative_discrepancy':0.}
    for account,seed in [('A',1),('B',2)]:
        rows=[]
        with tarfile.open(DATA/f'{account}_frozen_predictions.tar.gz') as tf,tarfile.open(DATA/f'{account}_development.tar.gz') as ta:
            seal=json.load(tf.extractfile('development/seal.json'))
            for item in seal['files']:
                blob=tf.extractfile('development/'+item['relative']).read(); digest=hashlib.sha256(blob).hexdigest();assert digest==item['sha256']
                result['sources'][account+'/'+item['relative']]=digest
                an=json.load(ta.extractfile(item['relative'].replace('_predictions.pt','_analysis.json')));assert an['prediction_sha256']==digest
                d=load_arrays(blob);o=d['outputs'];regmax=o['pred_corners'].shape[-1]//4-1
                scale=float(o['reg_scale'].ravel()[0]);up=float(o['up'].ravel()[0]);step=(abs(up)*scale+1)**(2/(regmax-2))
                bins=np.array([-2*abs(up)*scale]+[1-step**i for i in range(regmax//2-1,0,-1)]+[0.]+[step**i-1 for i in range(1,regmax//2)]+[2*abs(up)*scale])
                for bi,r in enumerate(an['inputs']):
                    b=r['branches']['final']['native_union']; qs=np.array(b['pair_query']);gs=np.array(b['pair_GT'])
                    if not len(qs):continue
                    pred=o['pred_boxes'][bi,qs].astype(float); gt=d['targets'][bi]['boxes'][gs].astype(float)
                    x=xyxy(pred);y=xyxy(gt);ref=o['ref_points'][bi,qs].astype(float)
                    z=o['pred_corners'][bi,qs].astype(float).reshape(-1,4,regmax+1)
                    p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);mu=(p*bins).sum(-1)
                    wanted=np.concatenate(((ref[:,:2]-y[:,:2])*scale/ref[:,2:]-scale/2,(y[:,2:]-ref[:,:2])*scale/ref[:,2:]-scale/2),-1)
                    ix=np.clip(np.searchsorted(bins,wanted,side='right')-1,0,regmax-1)
                    wr=np.clip((wanted-bins[ix])/(bins[ix+1]-bins[ix]),0,1)
                    target=np.zeros_like(p)
                    nn,ee=np.indices(ix.shape);target[nn,ee,ix]=1-wr;target[nn,ee,ix+1]+=wr
                    np.testing.assert_allclose(target.sum(-1),1,atol=1e-12)
                    geometry,error=geometry_derivative(x,y)
                    result['finite_difference_max_relative_discrepancy']=max(result['finite_difference_max_relative_discrepancy'],error)
                    jac=p*(bins-mu[...,None])*np.tile(ref[:,2:]/scale,(1,2))[...,None]*np.array([-1,-1,1,1])[None,:,None]
                    boxgrad=geometry[...,None]*jac
                    actual=np.array([iou(a,b) for a,b in zip(x,y)])
                    fglgrad=.15*actual[:,None,None]*(p-target)
                    dot=(boxgrad*fglgrad).sum((1,2)); denom=(boxgrad**2).sum((1,2))
                    cap=np.array([ceiling(a,b) for a,b in zip(ref,y)])
                    # Size is in a common diagnostic 640 coordinate, independent of fixture resize.
                    size=np.sqrt(gt[:,2]*gt[:,3])*640
                    for k in range(len(qs)):
                        rows.append({'iou':actual[k],'cap':cap[k],'size640':size[k],'dot':dot[k],
                                     'effect_ratio':dot[k]/denom[k] if denom[k]>0 else 0.,'geometry_norm':np.sqrt(denom[k]),
                                     'fgl_norm':np.linalg.norm(fglgrad[k]),'outside':bool(np.any(np.abs(wanted[k])>bins[-1]))})
        for size in ['ALL','<4','4-8','8-16','16-32','>=32']:
            for group in ['all','overlap_bad_reachable','overlap_bad_unreachable','good']:
                rs=[r for r in rows if (size=='ALL' or (size=='<4' and r['size640']<4) or (size=='4-8' and 4<=r['size640']<8) or (size=='8-16' and 8<=r['size640']<16) or (size=='16-32' and 16<=r['size640']<32) or (size=='>=32' and r['size640']>=32)) and (group=='all' or (group=='overlap_bad_reachable' and 0<r['iou']<.5 and r['cap']>=.5) or (group=='overlap_bad_unreachable' and 0<r['iou']<.5 and r['cap']<.5) or (group=='good' and r['iou']>=.5))]
                if not rs:continue
                result['summaries'].append({'seed':seed,'size640':size,'group':group,'pair_exposures':len(rs),'FGL_active':sum(r['iou']>0 for r in rs),
                    'opposing':sum(r['dot']<0 for r in rs),'overturns_geometry_direction':sum(r['effect_ratio']< -1 for r in rs),
                    'effect_ratio_q01_q10_q50_q90_q99':np.quantile([r['effect_ratio'] for r in rs],[.01,.1,.5,.9,.99]).tolist(),
                    'sum_dot_over_sum_geometry_norm2':sum(r['dot'] for r in rs)/sum(r['geometry_norm']**2 for r in rs)})
    (OUT/'20_CPU开发组FGL与几何方向.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda v:v.item()),encoding='utf8')
    print(json.dumps({**{k:v for k,v in result.items() if k not in ['sources','summaries']},'summaries':[r for r in result['summaries'] if r['size640']=='ALL']},ensure_ascii=False,default=lambda v:v.item()))

if __name__=='__main__':main()
