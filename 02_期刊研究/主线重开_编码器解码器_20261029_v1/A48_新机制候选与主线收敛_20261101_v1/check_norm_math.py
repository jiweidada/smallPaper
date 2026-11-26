from pathlib import Path
import hashlib, json, sys

ROOT = Path(__file__).resolve().parent
if sys.platform == 'win32':
    sys.path.insert(0, str(ROOT.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from instance_moment_norm import MomentContext, BoxMomentBN, install, ordinary_inference, SITES

torch.set_num_threads(1)
torch.manual_seed(41048)
checks = []
target = [{'boxes':torch.tensor([[.5,.5,1.,1.]])} for _ in range(2)]
ctx = MomentContext('Q');ctx.set(target,1618)
a = torch.nn.BatchNorm2d(3)
b = BoxMomentBN(a,ctx)
x = torch.randn(2,3,4,5,requires_grad=True)
xx = x.detach().clone().requires_grad_(True)
g = torch.randn_like(x)
y,z = a(x),b(xx)
(y*g).sum().backward();(z*g).sum().backward()
for name,v,w in [('uniform output',y,z),('uniform input gradient',x.grad,xx.grad),
                 ('uniform running mean',a.running_mean,b.running_mean),
                 ('uniform running variance',a.running_var,b.running_var),
                 ('uniform affine gradient',a.weight.grad,b.weight.grad)]:
    assert torch.allclose(v,w,atol=3e-6,rtol=3e-6), name
    checks.append({'check':name,'max_abs_error':float((v-w).abs().max())})
ctx.clear()
boxes=[{'boxes':torch.tensor([[.125,.5,.25,1.],[.75,.5,.5,1.]])}]
weights={}
for mode in ['M','Q']:
    c=MomentContext(mode);c.set(boxes,1618)
    weights[mode]=c.weights(torch.zeros(1,3,2,4))
    assert torch.allclose(weights[mode].sum(),torch.tensor(1.))
    fg=(weights[mode]-.5/8)/.5
    expected_left=1/3 if mode=='M' else .5
    assert abs(float(fg[:,:,:,0].sum())-expected_left)<1e-6
    checks.append({'check':mode+' smaller box foreground allocation','actual':float(fg[:,:,:,0].sum()),'expected':expected_left})
    c.clear()
c=MomentContext('Q');c.set(boxes,1618)
layer=BoxMomentBN(torch.nn.BatchNorm2d(3),c)
x=torch.randn(1,3,2,4,requires_grad=True);g=torch.randn_like(x)
out=layer(x);(out*g).sum().backward()
with torch.no_grad():
    w=weights['Q'];mu=(w*x).sum((0,2,3))[None,:,None,None]
    d=x-mu;v=(w*d.square()).sum((0,2,3))[None,:,None,None]
    analytic=torch.rsqrt(v+layer.eps)*(g-w*g.sum((0,2,3),keepdim=True)-w*d*(g*d).sum((0,2,3),keepdim=True)/(v+layer.eps))
    assert torch.allclose(analytic,x.grad,atol=3e-6,rtol=3e-6)
    checks.append({'check':'weighted BN analytic gradient','max_abs_error':float((analytic-x.grad).abs().max())})
c.clear();layer.eval()
state=layer.state_dict();plain=torch.nn.BatchNorm2d(3);plain.load_state_dict(state);plain.eval()
assert torch.equal(layer(x.detach()),plain(x.detach()))
checks.append({'check':'eval is exact ordinary BN, no targets','pass':True})
empty=MomentContext('Q');empty.set([{'boxes':torch.empty(0,4)}],1618)
assert torch.equal(empty.weights(torch.zeros(1,3,2,4)),torch.full((1,1,2,4),1/8))
checks.append({'check':'empty GT uniform fallback','pass':True})
toy=torch.nn.Module();toy.encoder=torch.nn.Module()
for family in ['fpn_blocks','pan_blocks']:
    blocks=torch.nn.ModuleList()
    for _ in range(2):
        block=torch.nn.Module();block.cv4=torch.nn.Module();block.cv4.norm=torch.nn.BatchNorm2d(3)
        blocks.append(block)
    setattr(toy.encoder,family,blocks)
install(toy,'Q');toy.eval()
identities=[id(p) for p in toy.parameters()]
before={k:v.clone() for k,v in toy.state_dict().items()}
reference=[toy.get_submodule(site)(x.detach()) for site in SITES]
with ordinary_inference(toy):
    assert all(type(toy.get_submodule(site)) is torch.nn.BatchNorm2d for site in SITES)
    for site,ref in zip(SITES,reference):assert torch.equal(toy.get_submodule(site)(x.detach()),ref)
assert identities==[id(p) for p in toy.parameters()]
assert all(torch.equal(v,before[k]) for k,v in toy.state_dict().items())
checks.append({'check':'ordinary inference graph and original training parameter identities restored','pass':True})
result={'status':'PASS','checks':checks,'torch':torch.__version__,'detector_forwards':0,'optimizer_updates':0,
        'operator_sha256':hashlib.sha256((ROOT/'instance_moment_norm.py').read_bytes()).hexdigest()}
(ROOT/'norm_math_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
