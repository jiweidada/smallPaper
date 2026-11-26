"""Exercise learned (nonzero) interactions at a padded boundary and real CPU gradients."""
import hashlib,json,sys
from pathlib import Path
import torch
from torch import nn
from dual_view import DualViewDEIM,P3Interaction
from geometry import alignment_grid,align_global_p3

def main():
    torch.set_num_threads(4);torch.manual_seed(17)
    results={}
    crop=torch.tensor([[0.,0.,640/300,640/400]])
    local=torch.randn(1,256,8,8,requires_grad=True)
    global_p3=torch.randn(1,256,8,8,requires_grad=True)
    rest=[torch.randn(1,512,4,4),torch.randn(1,1024,2,2)]
    m=DualViewDEIM(nn.Identity(),nn.Identity(),nn.Identity(),'local_residual')
    with torch.no_grad():m.interaction.restore.weight.normal_(0,.05)
    seen={}
    hook=m.interaction.register_forward_pre_hook(lambda module,args:seen.update(context=args[1].detach().clone(),valid=args[2].detach().clone()))
    output=m.interact([local,*rest],[global_p3,*rest],crop)
    hook.remove()
    _,valid=align_global_p3(global_p3,crop,local.shape[-2:])
    results['same_padding_valid_as_aligned']=torch.equal(seen['valid'],valid)
    results['context_zero_on_invalid']=bool((seen['context'].masked_select(~valid.expand_as(local))==0).all())
    results['context_exact_local_on_valid']=torch.equal(seen['context'].masked_select(valid.expand_as(local)),local.detach().masked_select(valid.expand_as(local)))
    results['invalid_output_exact_local']=torch.equal(output[0].masked_select(~valid.expand_as(local)),local.masked_select(~valid.expand_as(local)))
    results['valid_output_actually_learned']=not torch.equal(output[0].masked_select(valid.expand_as(local)),local.masked_select(valid.expand_as(local)))
    results['P4_P5_same_objects']=all(a is b for a,b in zip(output[1:],rest))
    # Isolate context leakage: hold the local input constant, perturb only
    # invalid context, then compare residuals next to the boundary.
    context=local.detach()*valid
    dirty=context+1000*(~valid)
    a=m.interaction(local.detach(),context,valid)
    b=m.interaction(local.detach(),dirty*valid,valid)
    results['invalid_context_cannot_leak_after_mask']=torch.equal(a,b)
    output[0].square().mean().backward()
    results['finite_local_and_parameter_gradients']=bool(torch.isfinite(local.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.interaction.parameters()))
    results['global_not_in_local_residual_graph']=global_p3.grad is None
    fullcrop=torch.tensor([[0.,0.,1.,1.]])
    expected=m.interaction(local.detach(),local.detach(),torch.ones_like(valid))
    actual=m.interact([local.detach(),*rest],[global_p3.detach(),*rest],fullcrop)[0]
    results['ordinary_nonpadded_behavior_exact']=torch.equal(expected,actual)
    independent=DualViewDEIM(nn.Identity(),nn.Identity(),nn.Identity(),'independent')
    called=[];hook=independent.interaction.register_forward_hook(lambda *args:called.append(True))
    out=independent.interact([local.detach(),*rest],[global_p3.detach(),*rest],crop);hook.remove()
    results['independent_never_executes_interaction']=not called and out[0].data_ptr()==local.data_ptr()
    aligned=DualViewDEIM(nn.Identity(),nn.Identity(),nn.Identity(),'aligned')
    aligned.interaction.load_state_dict(m.interaction.state_dict())
    c,v=align_global_p3(global_p3.detach(),crop,local.shape[-2:])
    results['aligned_behavior_unchanged']=torch.equal(aligned.interact([local.detach(),*rest],[global_p3.detach(),*rest],crop)[0],aligned.interaction(local.detach(),c,v))
    results['same_parameter_shapes_counts']=sum(p.numel() for p in m.interaction.parameters())==49728 and list(m.state_dict())==list(aligned.state_dict())==list(independent.state_dict())
    result={'status':'passed' if all(results.values()) else 'failed','checks':results,'CPU_only':True,
      'changed_source':{'path':str(Path(__file__).with_name('dual_view.py')),'sha256':hashlib.sha256(Path(__file__).with_name('dual_view.py').read_bytes()).hexdigest()},
      'does_not_prove_APs_or_GPU_capacity':True}
    Path(__file__).with_name('padding_cpu_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2));assert all(results.values())

if __name__=='__main__':main()
