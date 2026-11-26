"""One bounded synthetic CUDA preflight; no detector, dataset, or optimizer."""
from pathlib import Path
import hashlib
import json
import time
import traceback

P=Path(__file__).resolve().parent
COUNTS={'GPU_operator_forwards':0,'GPU_backward_calls':0,'native_selective_scan_calls':0,
        'whole_detector_forwards':0,'detector_optimizer_updates':0,'real_images_used':0}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert P==Path('/root/uav_a57_20261102_v1')
    assert not (P/'CUDA_operator_result.json').exists() and not (P/'CUDA_operator_failure.json').exists()
    frozen=json.loads((P/'cuda_preflight_inputs_FROZEN_v1.json').read_text())
    assert all(sha(P/n)==h for n,h in frozen['files_sha256'].items())
    import torch
    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from state_readout_v2 import materialized_read,encoded_feature_read,packed_query_read
    torch.manual_seed(20261114)

    def args(h,w,q,b,d,s,grad=False):
        def tensor(x):return x.detach().requires_grad_(grad)
        return {'u':tensor(torch.randn(b,d,h*w,device='cuda')*.3),
                'delta':tensor(torch.rand(b,d,h*w,device='cuda')*.15+.01),
                'A':tensor(-torch.rand(d,s,device='cuda')*2-.1),
                'B':tensor(torch.randn(b,s,h*w,device='cuda')*.2),
                'z':tensor(torch.randn(b,d,h*w,device='cuda')),
                'D':tensor(torch.randn(d,device='cuda')),
                'Cquery':tensor(torch.randn(b,q,s,device='cuda')*.3),
                'reference_xy':tensor(torch.rand(b,q,2,device='cuda')*.8+.1),
                'hw':(h,w)}

    def scan(u,delta,A,B,C):
        COUNTS['native_selective_scan_calls']+=1
        return selective_scan_fn(u,delta,A,B,C,D=None,z=None,delta_bias=None,delta_softplus=False)

    def call(fn,**kw):
        COUNTS['GPU_operator_forwards']+=1
        assert COUNTS['GPU_operator_forwards']<=4
        return fn(**kw)

    a=args(3,4,5,2,8,4,grad=True)
    gold=call(materialized_read,**a,reverse=True)
    native=call(packed_query_read,**a,scan=scan,reverse=True)
    forward_error=float((native-gold).abs().max().detach())
    assert forward_error<2e-5,forward_error
    leaves=[v for v in a.values() if isinstance(v,torch.Tensor)]
    probe=torch.randn_like(gold)
    COUNTS['GPU_backward_calls']+=1
    grad_gold=torch.autograd.grad((gold*probe).sum(),leaves)
    COUNTS['GPU_backward_calls']+=1
    grad_native=torch.autograd.grad((native*probe).sum(),leaves)
    gradient_error=max(float((x-y).abs().max()) for x,y in zip(grad_gold,grad_native))
    assert gradient_error<2e-4 and all(torch.isfinite(g).all() for g in grad_native),gradient_error
    del gold,native,leaves,grad_gold,grad_native,a,probe
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    common=args(160,160,2400,1,512,16,grad=False)
    cctx=torch.randn(1,16,25600,device='cuda')*.3
    costs={}
    with torch.no_grad():
        for arm in ['M','Q']:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.monotonic()
            if arm=='M':
                out=call(encoded_feature_read,**{k:v for k,v in common.items() if k!='Cquery'},Ccontext=cctx,scan=scan)
            else:out=call(packed_query_read,**common,scan=scan)
            torch.cuda.synchronize()
            seconds=time.monotonic()-start
            assert out.shape==(1,2400,512) and torch.isfinite(out).all()
            costs[arm]={'seconds_single_untimed_warmup_forward':seconds,
                        'peak_allocated_bytes_including_common_inputs':torch.cuda.max_memory_allocated(),
                        'output_shape':list(out.shape),'finite':True}
            del out
    assert COUNTS['GPU_operator_forwards']==4 and COUNTS['GPU_backward_calls']==2
    result={'status':'PASS_CUDA_COMPONENT_ONLY','forward_max_abs_error':forward_error,
            'gradient_max_abs_error':gradient_error,'finite_gradients':True,
            'large_case':{'context_hw':[160,160],'queries':2400,'inner_channels':512,'state_size':16,
                          'directions_per_call':1,'costs':costs},'counts':COUNTS,
            'torch':torch.__version__,'mamba_ssm':mamba_ssm.__version__,'GPU':torch.cuda.get_device_name(),
            'frozen_manifest_sha256':sha(P/'cuda_preflight_inputs_FROZEN_v1.json'),
            'limits':['Synthetic FP32 components only.','No full-model24GB capacity proof or latency benchmark.','No UAV AP, learned task benefit, or novelty conclusion.']}
    with (P/'CUDA_operator_result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        with (P/'CUDA_operator_failure.json').open('x') as f:
            json.dump({'status':'FAILED_PRESERVED_NO_AUTOMATIC_RERUN','counts':COUNTS,'traceback':traceback.format_exc()},f,indent=2)
        raise
