"""Private synthetic branch checks only; no detector, data, CUDA or optimizer."""
from pathlib import Path
import hashlib,json,sys
import torch
from branch import LowRankP2Residual
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=json.loads((HERE/'protocol.json').read_bytes())
assert {n:sha(HERE/n) for n in spec['files']}==spec['files']
torch.set_num_threads(2)
torch.manual_seed(int(sys.argv[1]));rng=torch.get_rng_state().clone()
with torch.random.fork_rng(devices=[]):branch=LowRankP2Residual().eval()
assert torch.equal(rng,torch.get_rng_state())
assert sum(p.numel() for p in branch.parameters())==6176
assert len(branch.state_dict())==8
assert int(torch.linalg.matrix_rank(branch.mix.weight.view(16,64).double()))==16
assert torch.count_nonzero(branch.effective_weight())==0
cases=[]
for h,w in ((20,20),(21,23),(24,24)):
    p2=torch.linspace(-1,1,64*h*w).reshape(1,64,h,w)
    p3=torch.linspace(-.5,.5,256*((h+1)//2)*((w+1)//2)).reshape(1,256,(h+1)//2,(w+1)//2)
    branch.zero_grad(set_to_none=True);y=branch(p2,p3)
    assert torch.equal(y,p3);y.square().mean().backward()
    assert torch.isfinite(branch.restore.weight.grad).all() and (branch.restore.weight.grad!=0).any()
    assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in [branch.mix.weight,*branch.reduce.parameters()])
    cases.append(dict(input_hw=[h,w],zero_output_exact=True,restore_gradient_active=True,mix_and_reduce_initial_gradient_zero=True))
# This is a private synthetic weight perturbation, not an optimizer step or
# a trained model. It checks that the upstream factors can learn once the
# zero-initialized final factor becomes nonzero.
with torch.no_grad():
    branch.restore.weight.copy_(torch.linspace(-.02,.03,256*16).reshape(256,16,1,1))
branch.zero_grad(set_to_none=True);branch(p2,p3).square().mean().backward()
assert (branch.mix.weight.grad!=0).any()
assert all(p.grad is not None and torch.isfinite(p.grad).all() and (p.grad!=0).any() for p in branch.reduce.parameters())
assert torch.linalg.matrix_rank(branch.effective_weight().double()).item()<=16
# Unshuffle followed by the two linear maps still collapses to a rank-limited
# stride2 convolution after reduction; this algebraic identity is not novelty.
detail=branch.reduce(p2)
combined=branch.effective_weight().reshape(256,16,2,2)
direct=torch.nn.functional.conv2d(detail,combined,stride=2)
assert torch.allclose(branch(p2,p3)-p3,direct,atol=1e-7,rtol=2e-5)
try:branch(p2,p3[:,:,:-1]);raise AssertionError('Wrong P3 accepted')
except ValueError:pass
assert not torch.cuda.is_initialized()
report=dict(status='PASS_PRIVATE_RANK16_BRANCH_ZERO_PATH_AND_GRADIENT_CPU_REVIEW',seed=int(sys.argv[1]),files_sha256=spec['files'],protocol_sha256=sha(HERE/'protocol.json'),cases=cases,
    parameters=6176,parameters_saved_vs_original_P2=11264,upstream_gradients_after_private_weight_perturbation_active=True,RNG_preserved_by_fork=True,
    rank_upper_bound=16,equivalent_rank_limited_stride2_conv_verified=True,wrong_spatial_shape_rejected=True,
    full_detector_forwards=0,BASE_forwards=0,CUDA_initialized=False,optimizer_updates=0,training_epochs=0,
    synthetic_branch_forwards=6,private_manual_weight_perturbations=1,training_allowed=False,training_efficacy_proven=False,novelty_proven=False,
    limitations='Rank-limited linear mixing is known machinery; the initialization and gradient checks do not establish AP benefit. Neither factor is initialized from the trained validation-selected SVD.')
with (HERE.parent/f'seed{int(sys.argv[1])}_CPU_report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
