from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from distribution_relation import characteristic_features,pair_features,DistributionRelation
torch.set_num_threads(4);torch.manual_seed(61)
p=torch.softmax(torch.randn(2,6,4,9,dtype=torch.float64),-1)
x=torch.randn(2,6,4,9,dtype=torch.float64)*.012+.4
b=torch.rand(2,6,4,dtype=torch.float64);b[...,2:]=b[...,2:]*.1+.01
checks={}
one=torch.zeros_like(p);one[...,3]=1
a=pair_features(one,x,b,'M');q=pair_features(one,x,b,'Q')
checks['point_limit_max_error']=float((a-q).abs().max());assert checks['point_limit_max_error']<1e-12
perm=torch.tensor([3,0,5,2,4,1]);r=pair_features(p,x,b,'Q')
rp=pair_features(p[:,perm],x[:,perm],b[:,perm],'Q')
checks['permutation_max_error']=float((rp-r[:,perm][:,:,perm]).abs().max());assert checks['permutation_max_error']<1e-12
shift=.071;b2=b.clone();b2[...,:2]+=shift
shifted=pair_features(p,x+shift,b2,'Q')
checks['translation_max_error']=float((r-shifted).abs().max());assert checks['translation_max_error']<2e-12
# Same first two moments, unequal higher shape; no Monte Carlo noise.
pp=torch.tensor([[[[.5,0,.5],[.125,.75,.125]]]],dtype=torch.float64).reshape(1,1,2,3)
xx=torch.tensor([[[[-1.,0,1],[-2.,0,2]]]],dtype=torch.float64)*.02+.3
m=(pp*xx).sum(-1);v=(pp*(xx-m[...,None]).square()).sum(-1)
assert torch.allclose(m[...,0],m[...,1],atol=1e-15) and torch.allclose(v[...,0],v[...,1],atol=1e-15)
mc,ms=characteristic_features(pp,xx,'M');qc,qs=characteristic_features(pp,xx,'Q')
checks['same_moments_Gaussian_feature_difference']=float((mc[:,:,0]-mc[:,:,1]).abs().max())
checks['same_moments_categorical_feature_difference']=float((qc[:,:,0]-qc[:,:,1]).abs().max())
assert checks['same_moments_Gaussian_feature_difference']<1e-12 and checks['same_moments_categorical_feature_difference']>.1
# Actual mask dimensions, DN isolation, gradients and zero-output initialization.
module=DistributionRelation(8,4,300,'Q',4)
B,N,K=1,306,33
corners=torch.randn(B,N,4*K,requires_grad=True)
refs=torch.rand(B,N,4);refs[...,2:]=.05
boxes=refs.clone();project=torch.linspace(-4,4,K)
target=torch.randn(B,N,192,requires_grad=True)
forbidden=torch.zeros(N,N,dtype=torch.bool);forbidden[6:,:6]=True
mask=module.mask(0,(corners,refs,project,boxes),forbidden,target)
assert mask.shape==(8,N,N) and torch.isneginf(mask[:,6:,:6]).all()
assert torch.equal(mask[:,-300:,-300:],torch.zeros_like(mask[:,-300:,-300:]))
weights=torch.randn_like(mask[:,6:,6:]);(mask[:,6:,6:]*weights).sum().backward()
assert corners.grad is None and module.maps[0].last.weight.grad.abs().max()>0
assert all(v.grad is None for v in module.maps[1:].parameters())
checks['new_parameters']=sum(v.numel() for v in module.parameters());assert checks['new_parameters']==6480
checks['DN_hard_mask_preserved']=True;checks['no_gradient_to_previous_distribution']=True
checks['initial_relation_output_exact_zero']=True
result={'status':'PASS','checks':checks,'detector_model_forwards':0,'detector_optimizer_updates':0,
 'scope':'Mathematical/module checks only; no evidence of detector AP, calibration, novelty, or trainability of a full detector.'}
path=ROOT/'math_check_v1.json';assert not path.exists();path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

