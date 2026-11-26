"""Isolated CUDA AMP Stem check after both current queued dispatches are confirmed."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,subprocess,sys
import torch
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main(prefix,seed):
    assert prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    p=read(HERE/'protocol.json');assert sha(__file__)==p['preflight_sha256']
    root=HERE.parent;out=root/f'seed{seed}_preflight_v1';out.mkdir(exist_ok=False)
    prototype=Path(prefix)/'DEIM/experiments/warm_stem_prototype_20261011_v2'
    assert {f.name:sha(f) for f in (prototype/'code').iterdir() if f.is_file()}==p['prototype_files_sha256']
    cpu=read(prototype/'CPU_execution_v1.json')
    assert cpu['exit_code']==0 and cpu['report']['status']==p['CPU_pass_status']
    assert not cpu['training_launched'] and cpu['report']['optimizer_steps']==0
    old=Path(prefix)/'DEIM/experiments/stem_component12_20261011_v1'
    queue=Path(prefix)/'DEIM/experiments/stem_component_queue_pulse_20261011_v2'
    done=read(queue/'completion.json')
    assert set(done['queued_dispatches'])=={'attention_spatial_mean','early_final_downsample'}
    for arm in done['queued_dispatches']:
        assert (old/f'seed{seed}_{arm}/parallel_resource_receipt.json').exists()
        assert (old/f'seed{seed}_{arm}/candidate_start.json').exists()
    line=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    receipt=dict(UUID=line[0].strip(),free_MiB=int(line[1]),GPU_compute_apps=apps,
       UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_existing_process_signals=True)
    assert receipt['UUID']==p['GPU_uuid'][str(seed)]
    write(out/'resource_receipt.json',receipt)
    if receipt['free_MiB']<8192 or len(apps)>3:
        r=dict(status='DEFER_GPU_PREFLIGHT_CURRENT_CAPACITY',seed=seed,receipt=receipt,
               training_epochs=0,optimizer_steps=0,detector_forwards=0)
        write(out/'report.json',r);print(json.dumps(r));return
    assert 'engine' not in sys.modules
    original=Path(prefix)/'DEIM/experiments/original_stem12_20261011_v1'
    sys.path.append(str(original/'code'))
    from runtime import verify_bundle,activate
    _,plan,paths=verify_bundle(original/f'seed{seed}_original_stem/bundle')
    activate(paths['native_root']);sys.path.insert(0,str(prototype/'code'))
    from warm_stem_model import WarmStartEnhancedDEIM
    from initialization import load_warm_common
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=WarmStartEnhancedDEIM(plan['historical_resolved_config']['ResidualGeometryDEIM']['recipe'])
    provenance=load_warm_common(model,paths['common_init'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    stem=model.backbone.stem.cuda().train();reference=copy.deepcopy(stem.native).train()
    torch.cuda.reset_peak_memory_stats()
    counts=dict(prototype_detector=0,warm_stem=0,reference_native_stem=0,retained_native_stem=0,enhanced_stem=0)
    def hook(name):
        def count(module,args):counts[name]+=1
        return count
    handles=[model.register_forward_pre_hook(hook('prototype_detector')),
      stem.register_forward_pre_hook(hook('warm_stem')),reference.register_forward_pre_hook(hook('reference_native_stem')),
      stem.native.register_forward_pre_hook(hook('retained_native_stem')),
      stem.enhanced.register_forward_pre_hook(hook('enhanced_stem'))]
    cases=[]
    for side in (640,832):
        stem.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
        x=torch.linspace(-1,1,3*side*side,device='cuda').reshape(1,3,side,side).expand(4,-1,-1,-1).contiguous()
        with torch.autocast('cuda',dtype=torch.float16):
            expected=reference(x);actual=stem(x);loss=actual.float().square().mean()
        assert actual.shape==expected.shape==(4,16,side//4,side//4)
        assert torch.isfinite(actual).all() and torch.equal(actual,expected)
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in stem.parameters())
        assert torch.isfinite(stem.residual_gain.grad).all() and bool((stem.residual_gain.grad!=0).any())
        assert all(torch.equal(v.grad,torch.zeros_like(v.grad)) for v in stem.enhanced.parameters())
        cases.append(dict(side=side,output_dtype=str(actual.dtype),shape=list(actual.shape),
          zero_gain_output_exact=True,all_gradients_finite=True,gain_gradient_nonzero_channels=int((stem.residual_gain.grad!=0).sum())))
        del x,expected,actual,loss
    stem.zero_grad(set_to_none=True)
    with torch.no_grad():stem.residual_gain.fill_(0.1)
    x=torch.linspace(-1,1,3*640*640,device='cuda').reshape(1,3,640,640).expand(4,-1,-1,-1).contiguous()
    with torch.autocast('cuda',dtype=torch.float16):
        y=stem(x);loss=y.float().square().mean()
    assert torch.isfinite(y).all();loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in stem.parameters())
    active=int(sum(bool((v.grad!=0).any()) for v in stem.enhanced.parameters()));assert active>0
    torch.cuda.synchronize()
    for h in handles:h.remove()
    assert counts['prototype_detector']==0
    r=dict(status='PASS_CUDA_AMP_BOUND_STEM_SHAPES_ZERO_INITIAL_CONTRIBUTION_AND_ACTIVE_RESIDUAL_GRAPH',
      seed=seed,provenance=provenance,cases=cases,manually_nonzero_gate_enhanced_parameter_tensors_with_nonzero_gradient=active,
      actual_forward_counts=counts,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
      peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU_resource_receipt=receipt,
      evaluator_calls=0,optimizer_steps=0,training_epochs=0,full_detector_forward_validated=False,
      AP_improvement_proven=False,training_launched=False,protocol_sha256=sha(HERE/'protocol.json'),
      prototype_CPU_execution_sha256=sha(prototype/'CPU_execution_v1.json'),
      scope='Standalone Stem CUDA AMP engineering test at the bound native batch4 and640/832 training shapes, using actual common initial weights and synthetic inputs. Neither a full detector nor a saved BASE12 is forwarded. Manual nonzero-gate assignment probes gradients without an optimizer. This is not a training-memory forecast, complete detector preflight, AP result, causal proof or novelty claim.')
    write(out/'report.json',r);print(json.dumps(r))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prefix',required=True);a.add_argument('--seed',type=int,required=True)
    x=a.parse_args();main(x.prefix,x.seed)

