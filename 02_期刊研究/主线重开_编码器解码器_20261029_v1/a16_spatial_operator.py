"""FP64 reconstruction of the fixed native sampler and local finite responses.

No detector, torch, optimizer or fitting. Query/offset/attention stay fixed when
the sampling reference centre moves. This is a partial operator response, not
an image-shift experiment or the full decoder Jacobian.
"""
import numpy as np


def project(query, reference, state):
    q=np.asarray(query,dtype=np.float64)
    r=np.asarray(reference,dtype=np.float64).reshape(len(q),4)
    offsets=(q@state['sampling_offsets_weight'].astype(np.float64).T+
             state['sampling_offsets_bias'].astype(np.float64)).reshape(-1,8,16,2)
    logits=(q@state['attention_weights_weight'].astype(np.float64).T+
            state['attention_weights_bias'].astype(np.float64)).reshape(-1,8,16)
    exp=np.exp(logits-logits.max(-1,keepdims=True))
    weights=exp/exp.sum(-1,keepdims=True)
    scale=state['num_points_scale'].astype(np.float64).reshape(1,1,16,1)
    loc=r[:,None,None,:2]+offsets*scale*r[:,None,None,2:]*.5
    assert np.isfinite(loc).all() and np.isfinite(weights).all()
    return offsets,weights,loc


def pack(query,reference,values,shapes,state):
    offsets,weights,loc=project(query,reference,state)
    t=len(query);patch=np.zeros((t,8,16,3,3,32),dtype=np.float32)
    local=np.empty((t,8,16,2),dtype=np.float64)
    wh=np.repeat(np.asarray(shapes,dtype=np.int64)[:,::-1],4,axis=0)
    pixels=loc*wh[None,None,:,:]-.5
    assert np.abs(pixels).max(initial=0)<1e12
    nearest=np.floor(pixels+.5).astype(np.int64)
    local[:]=pixels-nearest+1
    for level,(height,width) in enumerate(shapes):
        sl=slice(level*4,(level+1)*4)
        v=np.asarray(values[level],dtype=np.float32).transpose(0,2,1)
        assert v.shape==(8,int(height)*int(width),32)
        for yy in range(3):
            for xx in range(3):
                ix=nearest[:,:,sl,0]+xx-1;iy=nearest[:,:,sl,1]+yy-1
                valid=(ix>=0)&(ix<width)&(iy>=0)&(iy<height)
                index=np.clip(iy,0,height-1)*width+np.clip(ix,0,width-1)
                got=v[np.arange(8)[None,:,None],index]
                patch[:,:,sl,yy,xx,:]=np.where(valid[...,None],got,0)
    return {'patch':patch,'local_coords':local,'point_wh':wh,
            'attention_weights':weights,'locations':loc,'offsets':offsets,
            'ca_query':np.asarray(query,dtype=np.float32),
            'reference':np.asarray(reference,dtype=np.float32).reshape(t,1,4),
            'spatial_shapes':np.asarray(shapes,dtype=np.int64)}


def read_points(data,delta=(0.,0.)):
    patch=np.asarray(data['patch'],dtype=np.float64)
    coords=data['local_coords']+np.asarray(delta,dtype=np.float64)*data['point_wh'][None,None,:,:]/800.
    assert np.isfinite(coords).all() and (coords>=0).all() and (coords<2).all()
    xy=np.floor(coords).astype(np.int64);fraction=coords-xy
    flat=patch.reshape(-1,3,3,32);ind=np.arange(len(flat))
    xx=xy[...,0].reshape(-1);yy=xy[...,1].reshape(-1)
    fx=fraction[...,0].reshape(-1,1);fy=fraction[...,1].reshape(-1,1)
    value=((1-fx)*(1-fy)*flat[ind,yy,xx]+fx*(1-fy)*flat[ind,yy,xx+1]+
           (1-fx)*fy*flat[ind,yy+1,xx]+fx*fy*flat[ind,yy+1,xx+1])
    return value.reshape(*patch.shape[:3],32)


def pooled(data,delta=(0.,0.)):
    return np.einsum('thp,thpc->thc',data['attention_weights'],read_points(data,delta)).reshape(-1,256)


def response_metrics(data,epsilon=.125):
    derivatives=np.stack(((read_points(data,(epsilon,0))-read_points(data,(-epsilon,0)))/(2*epsilon),
                          (read_points(data,(0,epsilon))-read_points(data,(0,-epsilon)))/(2*epsilon)),axis=-1)
    a=data['attention_weights']
    stack=np.einsum('thp,thpci,thpcj->tij',a,derivatives,derivatives)
    j=np.einsum('thp,thpci->thci',a,derivatives)
    pool=np.einsum('thci,thcj->tij',j,j)
    level=np.zeros_like(stack)
    for l in range(4):
        sl=slice(l*4,(l+1)*4);mass=a[:,:,sl].sum(-1)
        summed=np.einsum('thp,thpci->thci',a[:,:,sl],derivatives[:,:,sl])
        avg=np.divide(summed,mass[...,None,None],out=np.zeros_like(summed),where=mass[...,None,None]>0)
        level+=np.einsum('th,thci,thcj->tij',mass,avg,avg)
    ss=np.trace(stack,axis1=-2,axis2=-1);ll=np.trace(level,axis1=-2,axis2=-1);pp=np.trace(pool,axis1=-2,axis2=-1)
    scale=np.maximum(ss,1.)
    assert np.all(np.linalg.eigvalsh(stack-level)[:,0]>=-1e-10*scale)
    assert np.all(np.linalg.eigvalsh(level-pool)[:,0]>=-1e-10*scale)
    total=np.divide(pp,ss,out=np.ones_like(pp),where=ss>0)
    within=np.divide(ll,ss,out=np.ones_like(ll),where=ss>0)
    across=np.divide(pp,ll,out=np.ones_like(pp),where=ll>0)
    assert np.isfinite(total).all() and np.all(total>=-1e-10) and np.all(total<=1+1e-10)
    return {'G_stack':stack,'G_level':level,'G_pool':pool,'trace_stack':ss,'trace_level':ll,'trace_pool':pp,
            'cancellation':np.clip(1-total,0,1),'within_level_cancellation':np.clip(1-within,0,1),
            'across_level_cancellation':np.clip(1-across,0,1),'zero_response':ss==0}
