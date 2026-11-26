"""Finite CPU synthetic correctness allocation, not a UAV failure experiment."""
from pathlib import Path
import hashlib
import json
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent.parent/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from state_readout_v2 import materialized_read,encoded_feature_read,packed_query_read,stencil

COUNTS={'standalone_operator_forwards':0,'synthetic_backward_calls':0,
        'synthetic_component_optimizer_updates':0,'whole_detector_forwards':0,
        'detector_optimizer_updates':0,'real_images_used':0}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def call(fn,*args,**kwargs):
    COUNTS['standalone_operator_forwards']+=1
    assert COUNTS['standalone_operator_forwards']<=18
    return fn(*args,**kwargs)


def arguments(h,w,q):
    b,d,s,n=2,4,3,h*w
    def leaf(x):return x.detach().requires_grad_(True)
    return {'u':leaf(torch.randn(b,d,n,dtype=torch.float64)*.3),
            'delta':leaf(torch.rand(b,d,n,dtype=torch.float64)*.15+.01),
            'A':leaf(-torch.rand(d,s,dtype=torch.float64)*2-.1),
            'B':leaf(torch.randn(b,s,n,dtype=torch.float64)*.2),
            'z':leaf(torch.randn(b,d,n,dtype=torch.float64)),
            'D':leaf(torch.randn(d,dtype=torch.float64)),
            'Cquery':leaf(torch.randn(b,q,s,dtype=torch.float64)*.3),
            'reference_xy':leaf(torch.rand(b,q,2,dtype=torch.float64)*1.2-.1),
            'hw':(h,w)}


def main():
    assert not (P/'CPU_operator_result_v2.json').exists()
    manifest=json.loads((P/'preflight_inputs_FROZEN_v2.json').read_text('utf-8'))
    assert all(sha(P/n)==h for n,h in manifest['files_sha256'].items())
    torch.manual_seed(20261114)
    results=[]
    saved=None
    for h,w,q,reverse in [(2,3,3,False),(3,4,5,True),(1,5,2,False)]:
        a=arguments(h,w,q)
        cctx=torch.randn(2,3,h*w,dtype=torch.float64)*.3
        m_ref=call(materialized_read,**a,Ccontext=cctx,reverse=reverse)
        m_a={k:v for k,v in a.items() if k!='Cquery'}
        m_plain=call(encoded_feature_read,**m_a,Ccontext=cctx,reverse=reverse)
        q_ref=call(materialized_read,**a,reverse=reverse)
        q_packed=call(packed_query_read,**a,reverse=reverse)
        dm=float((m_ref-m_plain).abs().max().detach())
        dq=float((q_ref-q_packed).abs().max().detach())
        assert dm<1e-11 and dq<1e-11 and torch.isfinite(q_packed).all()
        ids,weights=stencil(a['reference_xy'],a['hw'])
        assert torch.allclose(weights.sum(-1),torch.ones_like(weights[:,:,0]),atol=1e-15,rtol=0)
        results.append({'hw':[h,w],'Q':q,'reverse':reverse,
                        'ordinary_encoded_vs_materialized_max_abs':dm,
                        'packed_query_vs_materialized_max_abs':dq})
        if saved is None:saved=(a,q_ref,q_packed)
    a,q_ref,q_packed=saved
    perm=torch.tensor([2,0,1])
    b=dict(a,Cquery=a['Cquery'][:,perm],reference_xy=a['reference_xy'][:,perm])
    permuted=call(packed_query_read,**b)
    permutation_error=float((permuted-q_packed[:,perm]).abs().max().detach())
    assert permutation_error<1e-11
    b=dict(a,Cquery=torch.cat([a['Cquery'],a['Cquery'][:,:1]],1),
           reference_xy=torch.cat([a['reference_xy'],a['reference_xy'][:,:1]],1))
    duplicated=call(packed_query_read,**b)
    duplication_error=float((duplicated[:,:3]-q_packed).abs().max().detach())
    assert duplication_error<1e-11
    c=torch.randn(2,3,1,dtype=torch.float64)
    const_m=call(encoded_feature_read,**{k:v for k,v in a.items() if k!='Cquery'},
                 Ccontext=c.expand(-1,-1,6))
    const_q=call(packed_query_read,**dict(a,Cquery=c.transpose(1,2).expand(-1,3,-1)))
    constant_C_error=float((const_m-const_q).abs().max().detach())
    assert constant_C_error<1e-11
    zero=call(packed_query_read,**dict(a,Cquery=a['Cquery']*0))
    ids,weights=stencil(a['reference_xy'],a['hw'])
    skip=a['u']*a['D'][None,:,None]*torch.nn.functional.silu(a['z'])
    expected=(skip.gather(2,ids.flatten(1)[:,None].expand(-1,4,-1)).reshape(2,4,3,4)*weights[:,None]).sum(-1).transpose(1,2)
    zero_C_skip_error=float((zero-expected).abs().max().detach())
    assert zero_C_skip_error<1e-11
    leaves=[v for v in a.values() if isinstance(v,torch.Tensor)]
    probe=torch.randn_like(q_ref)
    COUNTS['synthetic_backward_calls']+=1
    g_ref=torch.autograd.grad((q_ref*probe).sum(),leaves,retain_graph=True)
    COUNTS['synthetic_backward_calls']+=1
    g_packed=torch.autograd.grad((q_packed*probe).sum(),leaves)
    errors=[float((x-y).abs().max()) for x,y in zip(g_ref,g_packed)]
    assert max(errors)<1e-10 and all(torch.isfinite(g).all() for g in g_packed)
    assert COUNTS['standalone_operator_forwards']==17 and COUNTS['synthetic_backward_calls']==2
    result={'status':'PASS_SYNTHETIC_OPERATOR_ONLY','cases':results,
            'query_permutation_error':permutation_error,'existing_query_duplication_error':duplication_error,
            'constant_C_control_equivalence_error':constant_C_error,'zero_C_skip_error':zero_C_skip_error,
            'gradient_max_abs_error':max(errors),'finite_gradients':True,'counts':COUNTS,
            'dtype':'float64 CPU gold; not native CUDA or whole-model validation',
            'preflight_manifest_sha256':sha(P/'preflight_inputs_FROZEN_v2.json'),
            'limitations':['No UAV data or learned detector tested.','No AP, cause, novelty, or24GB full-model feasibility established.']}
    with (P/'CPU_operator_result_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        with (P/'CPU_operator_failure_v2.json').open('x',encoding='utf-8') as f:
            json.dump({'status':'FAILED_PRESERVED','counts':COUNTS,'traceback':traceback.format_exc()},f,indent=2)
        raise
