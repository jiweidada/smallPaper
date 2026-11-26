from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parent
ENG=ROOT.parent.parent
runtime=ENG/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'
sys.path.insert(0,str(runtime))
import torch
from edge_shape_matching import expected_error
support=torch.tensor([-3.,-2.,-1.,0.,1.,2.,3.],dtype=torch.float64)
p=torch.tensor([.16875,0,.28125,.1,.28125,0,.16875],dtype=torch.float64)
q=torch.tensor([0,.45,0,.1,0,.45,0],dtype=torch.float64)
probs=torch.stack([p,q])[:,None,:].expand(-1,4,-1)
supports=support[None,None,:].expand(2,4,-1)
# All GT edges at zero; normalization is eps, so multiply back in this mathematical toy.
gt=torch.zeros((1,4),dtype=torch.float64)
exact=expected_error(probs,supports,gt,'Q')[:,0]*1e-6
gauss=expected_error(probs,supports,gt,'M')[:,0]*1e-6
mean=(probs*supports).sum(-1);var=(probs*(supports-mean[...,None]).square()).sum(-1)
assert torch.allclose(mean,torch.zeros_like(mean),atol=1e-14)
assert torch.allclose(var,torch.full_like(var,3.6),atol=1e-14)
assert torch.allclose(exact,torch.tensor([1.575,1.8],dtype=torch.float64),atol=1e-14)
assert torch.allclose(gauss,torch.full_like(gauss,math.sqrt(3.6)*math.sqrt(2/math.pi)),atol=1e-14)
# Dirac limit agrees for both approximations and is target-dependent.
prob=torch.zeros((1,4,7),dtype=torch.float64);prob[:,:,3]=1
gt=torch.tensor([[-1.,-2.,1.,2.]],dtype=torch.float64)
m=expected_error(prob,supports[:1],gt,'M');v=expected_error(prob,supports[:1],gt,'Q')
assert torch.equal(m,v) and torch.equal(m,torch.tensor([[.5]],dtype=torch.float64))
result={'status':'PASS','scope':'mathematical cost checks only; no detector execution or accuracy evidence',
 'same_mean':0,'same_variance':3.6,'same_GT_bin_probability':.1,
 'exact_expected_absolute_error':exact.tolist(),'moment_Gaussian_expected_absolute_error':gauss.tolist(),
 'Dirac_limit_both':m.tolist(),'model_forwards':0,'optimizer_updates':0}
(ROOT/'math_check_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
