"""One training-annotation pass and a bounded box-geometry calculation, no models."""
from pathlib import Path
import collections
import hashlib
import json
import math
import signal
import sys
import time
import traceback
import numpy as np
from PIL import Image

STAGE=Path('/root/uav_a54_20261101_v1/A60_uploads')
OUT=Path('/root/uav_a54_20261101_v1/A60_geometry_D0_v1')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def clock(u,d):
    d=np.asarray(d,dtype=np.float64)
    den=np.where(d>1e-10,d,1.)
    a=1.-np.expm1((1.-u)*np.log1p(d))/den
    return np.where(d>1e-10,a,u)

def fraction(d,threshold,kind,reference):
    z=np.minimum(1.,threshold/np.maximum(d,1e-300))
    if kind=='linear':return z
    if kind=='global':return np.log1p(reference*z)/np.log1p(reference)
    if kind=='instance':return np.minimum(1.,np.log1p(threshold)/np.maximum(np.log1p(d),1e-300))
    raise ValueError(kind)

def trunc_normal(rng,n):
    # Deliberately specified N(0,1) truncated to [0,1]. Author code not reproduced.
    out=np.empty(n);written=0
    while written<n:
        x=rng.normal(size=max(64,(n-written)*4))
        x=x[(x>=0.)&(x<=1.)];take=min(len(x),n-written)
        out[written:written+take]=x[:take];written+=take
    return out

def main():
    scope=json.loads((STAGE/'geometry_scope_FROZEN_v1.json').read_text())
    assert sha(__file__)==scope['script_sha256']
    assert not OUT.exists();OUT.mkdir()
    save(OUT/'run_once.json',dict(model_constructors=0,model_forwards=0,optimizer_updates=0,
        annotation_passes_allowed=1,selected_image_headers_allowed=scope['sample_images']))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen CPU time cap')))
    signal.alarm(scope['seconds_cap']);start=time.monotonic()
    data=Path(scope['train_root']);ann=data/'annotations';images=data/'images'
    names=sorted(p.name for p in images.glob('*.jpg'))
    assert len(names)==6471
    selected=sorted(names,key=lambda s:hashlib.sha256((str(scope['seed'])+'|'+s).encode()).digest())[:scope['sample_images']]
    selected_set=set(selected);rows=[];meta=[];counts=[];stream=hashlib.sha256()
    for name in names:
        raw=(ann/(Path(name).stem+'.txt')).read_bytes()
        stream.update(name.encode()+b'\0'+raw+b'\0')
        valid=[]
        for line in raw.decode().splitlines():
            z=[float(v.strip()) for v in line.split(',') if v.strip()]
            assert len(z)>=8
            x,y,w,h,score,cat=z[:6]
            if score>0 and 1<=int(cat)<=10 and w>0 and h>0:
                valid.append((x,y,w,h,int(cat)))
        counts.append((len(valid),name))
        if name not in selected_set:continue
        with Image.open(images/name) as im:width,height=im.size
        kept=0
        for x,y,w,h,cat in valid:
            a=max(0.,min(width,x));b=max(0.,min(height,y))
            c=max(0.,min(width,x+w));e=max(0.,min(height,y+h))
            if c<=a or e<=b:continue
            rows.append((a/width,b/height,c/width,e/height,
                min((c-a)*640/width,(e-b)*640/height),(c-a)*(e-b),cat))
            kept+=1
        meta.append(dict(name=name,width=width,height=height,clipped_valid_GT=kept,
            annotation_sha256=hashlib.sha256(raw).hexdigest()))
    arr=np.asarray(rows,dtype=np.float64);assert arr.ndim==2 and arr.shape[1]==7
    np.savez_compressed(OUT/'selected_training_GT_geometry.npz',rows=arr)
    save(OUT/'selected_images.json',meta)
    geometry_checks={}
    dtest=np.array([0.,1e-12,.1,1.,10.,100.,1000.])
    assert np.allclose(clock(0.,dtest),0.,atol=1e-12) and np.allclose(clock(1.,dtest),1.,atol=1e-12)
    for u in [.1,.5,.9]:
        t=clock(u,dtest);assert np.all((0<=t)&(t<=1))
        assert np.allclose(np.log1p((1-t)*dtest),(1-u)*np.log1p(dtest),atol=2e-12)
    geometry_checks['endpoint_and_relative_error_clock_identity']=True
    priors={};rng=np.random.default_rng(scope['seed'])
    pair_GT=np.repeat(arr,scope['prior_draws_per_GT'],axis=0)
    n=len(pair_GT)
    for prior in scope['priors']:
        xywh=(trunc_normal(rng,n*4).reshape(n,4) if prior=='specified_truncated_gaussian' else rng.uniform(size=(n,4)))
        xywh[:,2:]=np.maximum(xywh[:,2:],1e-8)
        corners=np.concatenate((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2),axis=1)
        wh=pair_GT[:,2:4]-pair_GT[:,:2]
        norms=np.concatenate((wh,wh),axis=1)
        d=np.max(np.abs(corners-pair_GT[:,:4])/norms,axis=1)
        assert np.all(np.isfinite(d)) and np.all(d>=0)
        reference=float(np.median(d))
        masks={'all_GT':np.ones(n,dtype=bool),'small_original_area_le_1024':pair_GT[:,5]<=1024}
        short=pair_GT[:,4];small=masks['small_original_area_le_1024']
        masks.update({'small_short_le_4':small&(short<=4),'small_short_4_8':small&(short>4)&(short<=8),
            'small_short_8_16':small&(short>8)&(short<=16),'small_short_gt_16':small&(short>16)})
        stats={}
        for label,mask in masks.items():
            if not mask.any():continue
            v=d[mask];f={kind:fraction(v,scope['relative_edge_threshold'],kind,reference) for kind in ['linear','global','instance']}
            stats[label]=dict(GT_count=int(mask.sum()/scope['prior_draws_per_GT']),pairs=int(mask.sum()),
                d0_median=float(np.median(v)),fine_stage_time_fraction_mean={kind:float(z.mean()) for kind,z in f.items()},
                instance_over_global_mean_fraction=float(f['instance'].mean()/f['global'].mean()))
        # Secondary observed IoU at predeclared midpoint times; not AP or recall.
        bins={k:v for k,v in masks.items() if k in ['all_GT','small_short_le_4','small_short_4_8','small_short_8_16','small_short_gt_16']}
        hits={kind:np.zeros(n,dtype=np.int32) for kind in ['linear','global','instance']}
        ga=(pair_GT[:,2]-pair_GT[:,0])*(pair_GT[:,3]-pair_GT[:,1])
        for u in (np.arange(scope['midpoint_time_nodes'])+.5)/scope['midpoint_time_nodes']:
            for kind in hits:
                t=(np.full(n,u) if kind=='linear' else clock(u,reference if kind=='global' else d))
                box=corners+(pair_GT[:,:4]-corners)*np.asarray(t)[...,None]
                inter_wh=np.maximum(0.,np.minimum(box[:,2:],pair_GT[:,2:4])-np.maximum(box[:,:2],pair_GT[:,:2]))
                inter=inter_wh[:,0]*inter_wh[:,1]
                area=(box[:,2]-box[:,0])*(box[:,3]-box[:,1])
                iou=inter/(area+ga-inter)
                hits[kind]+=(iou>=.5)
        iou_stats={label:{kind:float(hit[mask].mean()/scope['midpoint_time_nodes']) for kind,hit in hits.items()} for label,mask in bins.items() if mask.any()}
        priors[prior]=dict(reference_d0_train_sample_median=reference,strata=stats,
            secondary_fraction_of_midpoint_times_with_IoU_ge_half=iou_stats)
        np.savez_compressed(OUT/(prior+'_derived_pairs.npz'),d0=d,GT=pair_GT,prior_corners=corners)
    assert 'torch' not in sys.modules
    result=dict(status='GEOMETRY_D0_COMPLETE_NOT_A_PERFORMANCE_TEST',source_train_images=6471,
        annotation_stream_sha256=stream.hexdigest(),raw_valid_count_upper_bound=sum(a for a,_ in counts),
        maximum_raw_valid_GT_upper_bound=max(counts)[0],largest_counts=sorted(counts,reverse=True)[:10],
        images_above_300=sum(a>300 for a,_ in counts),images_above_500=sum(a>500 for a,_ in counts),
        selected_image_headers=len(meta),selected_clipped_GT=len(rows),priors=priors,geometry_checks=geometry_checks,
        model_constructors=0,model_forwards=0,optimizer_updates=0,checkpoint_loads=0,torch_imported=False,
        elapsed_seconds=time.monotonic()-start,scope_sha256=sha(STAGE/'geometry_scope_FROZEN_v1.json'),script_sha256=sha(__file__),
        limits='A specified normalized-box prior and training-GT geometry calculation. Not native author-code reproduction, learned failure attribution, AP/recall, causal benefit or novelty evidence.')
    save(OUT/'result.json',result);save(OUT/'exit.json',dict(exit=0));print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:
        main()
    except BaseException:
        if OUT.is_dir() and not (OUT/'exit.json').exists():
            save(OUT/'exit.json',dict(exit=1,traceback=traceback.format_exc(),no_retry=True))
        traceback.print_exc()
        raise
