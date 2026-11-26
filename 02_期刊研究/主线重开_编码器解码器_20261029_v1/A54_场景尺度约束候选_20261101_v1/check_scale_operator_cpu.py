from pathlib import Path
import sys,json,hashlib,math
P=Path(__file__).resolve().parent;ENG=P.parents[1]
sys.path.insert(0,str(ENG/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from scene_scale_prior import SceneScalePrior
def main():
    torch.set_num_threads(2);torch.manual_seed(71);checks=[]
    def ck(name,ok):
        checks.append({'name':name,'passed':bool(ok)})
        assert ok,name
    offsets=torch.linspace(-.5,.5,10);M=SceneScalePrior(192,'M',offsets,-3.6);Q=SceneScalePrior(192,'Q',offsets,-3.6);Q.load_state_dict(M.state_dict())
    ck('same trainable parameter count',sum(p.numel() for p in M.parameters())==sum(p.numel() for p in Q.parameters())==6210)
    raw=torch.tensor([[[.4,-.7,-3.2,-4.1],[.2,.3,-2.2,-2.9]]],requires_grad=True);scores=torch.randn(1,2,10)
    fmap=torch.full((1,1,9,13),-3.6,requires_grad=True)
    ck('zero gain preserves raw references exactly M',torch.equal(M.correct(fmap,raw,scores),raw))
    ck('zero gain preserves raw references exactly Q',torch.equal(Q.correct(fmap,raw,scores),raw))
    ck('uniform initial map yields equal M/Q references',torch.equal(M.correct(fmap,raw,scores),Q.correct(fmap,raw,scores)))
    out=Q.correct(fmap,raw,scores);out[...,2:].sum().backward()
    ck('zero-initialized gain receives a main-path gradient',Q.gain.grad is not None and Q.gain.grad.abs()>0)
    Q.gain.data.fill_(.5);out=Q.correct(fmap,raw,scores)
    ck('reference centers unchanged',torch.equal(out[...,:2],raw[...,:2]))
    ratio=out.sigmoid()[...,2]/out.sigmoid()[...,3];base=raw.sigmoid()[...,2]/raw.sigmoid()[...,3]
    ck('isotropic size correction preserves aspect ratio away from clipping',torch.allclose(ratio,base,atol=1e-6,rtol=1e-6))
    x,y=Q.basis(fmap);plane=(-3.2+.6*x[None,None,None,:]-.3*y[None,None,:,None]).expand(1,1,9,13).clone()
    coeff=Q.coefficients(plane);expected=torch.tensor([[-3.2,.6*float(x.square().mean()/(x.square().mean()+Q.ridge)),-.3*float(y.square().mean()/(y.square().mean()+Q.ridge))]])
    ck('fixed ridge affine projection matches analytic plane coefficients',torch.allclose(coeff,expected,atol=1e-6))
    point=torch.tensor([[[.25,.4],[.8,.8]]]);reflected=point.clone();reflected[...,0]=1-reflected[...,0]
    ck('shared field commutes with horizontal flip',torch.allclose(Q.read(plane,point),Q.read(plane.flip(-1),reflected),atol=1e-6))
    changed=plane.clone();changed[0,0,0,0]+=2
    ck('Q query value depends on distant scene values',not torch.allclose(Q.read(plane,point),Q.read(changed,point),atol=1e-6))
    ck('M local query value unaffected by distant pixel',torch.equal(M.read(plane,point),M.read(changed,point)))
    targets=[{'boxes':torch.tensor([[.3,.6,.01,.02],[.7,.2,.05,.07]]),'labels':torch.tensor([0,8])}]
    for arm,mod in [('M',M),('Q',Q)]:
        mod.zero_grad();features=torch.randn(1,192,9,13);pred=mod.map(features);loss=mod.auxiliary_loss(pred,targets);loss.backward()
        ck(arm+' auxiliary task supplies finite, nonzero head learning signal',torch.isfinite(loss) and mod.map[-1].weight.grad.abs().sum()>0)
    result={'status':'PASSED','checks':checks,'new_detector_whole_model_forwards':0,'new_detector_optimizer_updates':0,'scope':'Standalone operator and gradient properties, not a detector integration or memory/capacity test.','torch':torch.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'module_sha256':hashlib.sha256((P/'scene_scale_prior.py').read_bytes()).hexdigest()}
    (P/'04_算子CPU核查_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
