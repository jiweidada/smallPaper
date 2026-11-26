"""Pre-data numerical-test correction. Original script and failure are preserved.

Only change: use a dot-product rounding bound for an algebraic float64 identity.
No candidate/data/seed/threshold/hyperparameter changes from protocol19.
"""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('boundary_v1_preserved',ROOT/'20_boundary_frozen.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
torch,F,math=b.torch,b.F,b.math
b.OUT=ROOT/'23_boundary_frozen_v2'
b.__file__=__file__


def synthetic_v2():
    torch.manual_seed(78021)
    fake=torch.ones(2,5,4,2,3,dtype=torch.float64)
    assert bool((b.arm_features(fake)['difference']==0).all())
    fake=torch.randn_like(fake);feats=b.arm_features(fake)
    w=torch.randn(4,12,dtype=torch.float64)
    expanded=torch.stack([w.reshape(4,4,3)/math.sqrt(2),-w.reshape(4,4,3)/math.sqrt(2)],-2).flatten(-3)
    err=float((F.linear(feats['difference'],w)-F.linear(feats['full_concat'],expanded)).abs().max())
    bound=float(64*torch.finfo(torch.float64).eps*F.linear(feats['full_concat'].abs(),expanded.abs()).max())
    assert err<=bound,(err,bound)
    group={'amsgrad':False,'betas':[.9,.999],'lr':.0004,'weight_decay':.0001,'eps':1e-8,'maximize':False}
    g=torch.tensor([[.2,-.3,0.]],dtype=torch.float64)
    got,detail=b.fresh_update(g,group);clipped=g*detail['clip_factor']
    expected=-group['lr']*clipped/(clipped.abs()+group['eps'])
    assert float((got-expected).abs().max())<1e-16
    a=torch.tensor([[.3,.4,.2,.1],[.2,.7,.04,.08]],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([[.4,.5,.1,.2],[.8,.7,.04,.08]],dtype=torch.float64)
    direct,_=b.paired_terms(a,target)
    from engine.deim.box_ops import generalized_box_iou
    exact=5*(a-target).abs().sum(-1)+2*(1-generalized_box_iou(b.n.box_cxcywh_to_xyxy(a),b.n.box_cxcywh_to_xyxy(target)).diagonal())
    assert float((direct-exact).abs().max())<1e-14
    ga=torch.autograd.grad(direct.sum(),a,retain_graph=True)[0]
    gb=torch.autograd.grad(exact.sum(),a)[0]
    assert float((ga-gb).abs().max())<1e-14
    return {'constant_diff_zero':True,'concat_contains_difference':True,
            'algebraic_identity_float64_error':err,'rounding_bound':bound,
            'fresh_AdamW_matches_closed_first_step':True,
            'aligned_geometry_value_and_gradient_match_native':True}


b.synthetic=synthetic_v2
if __name__=='__main__':
    try:
        b.main()
        b.save('03_predata_revision.json',{
            'reason':'v1 stopped on synthetic float64 identity 2.6645352591e-15 vs absolute2e-15; no real predictions/features/endpoint outcomes had been read',
            'change':'Replace only this numerical identity assertion with64*eps*sum(abs(products)); all scientific decisions, arms, data, seeds unchanged',
            'original_script_sha256':b.n.sha(ROOT/'20_boundary_frozen.py'),
            'correction_script_sha256':b.n.sha(__file__),
            'v1_failure_sha256':b.n.sha(ROOT/'21_boundary_frozen/failure.json'),
            'no_identical_retry':True})
        b.save('04_final_manifest.json',{p.name:b.n.sha(p) for p in b.OUT.iterdir() if p.is_file()})
    except BaseException as e:
        if b.OUT.exists():b.save('failure.json',{'error':repr(e),'traceback':b.traceback.format_exc(),'retry_identical_script':False})
        raise
