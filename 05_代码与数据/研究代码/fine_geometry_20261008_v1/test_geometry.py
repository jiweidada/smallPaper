"""CPU engineering tests on synthetic tensors and fresh models, without updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import time
import traceback
import torch
from runtime import HERE, activate_snapshot, build_components, verify_sources, register_components

torch.set_num_threads(4)
CHECKS = []


def check(name, value):
    assert value, name
    CHECKS.append(name)


def test_helpers():
    from geometry import selected_geometry_memory, grid_reference
    torch.manual_seed(810086)
    semantic = torch.randn(2,3,3,requires_grad=True)
    fine = torch.randn(2,3,2,2,requires_grad=True)
    ids = torch.tensor([[0,3,4],[1,2,5]])
    valid = torch.ones(1,6,1,dtype=torch.bool);valid[0,3] = False
    out = selected_geometry_memory(semantic,ids,fine,valid,lambda x: 2*x+1)
    flat = fine.flatten(2).transpose(1,2)
    check('P3 uses the selected pre-fusion token',torch.equal(out[0,0],2*flat[0,0]+1))
    check('Invalid P3 masking happens before shared projection',torch.equal(out[0,1],torch.ones(3)))
    check('P4/P5 retain their semantic vectors exactly',torch.equal(out[:,2],semantic[:,2]))
    gf,gs = torch.autograd.grad(out.sum(),(fine,semantic))
    check('P3 geometry gradients use fine features, not selected semantic features',torch.count_nonzero(gs[:,:2]).item()==0 and gf[0,:,0,0].eq(2).all().item())
    check('Invalid fine features receive zero geometry gradient',gf[0,:,1,1].eq(0).all().item())
    check('Coarse selected vectors keep their gradient',gs[:,2].eq(1).all().item())
    coarse = selected_geometry_memory(semantic,torch.full((2,3),5),fine,valid,lambda x:x)
    check('No selected P3 is a valid no-op',coarse is semantic)
    anchors = torch.logit(torch.tensor([[[.2,.3,.05,.05],[.4,.5,.1,.1],[.7,.8,.05,.05]]]))
    anchors[0,2] = torch.inf
    raw = torch.tensor([[[.4,-.7,.2,-.3],[.2,.3,.4,.5],[1.,1.,1.,1.]]],requires_grad=True)
    token_ids=torch.tensor([[20,6420,22]])
    ref=grid_reference(raw,anchors,token_ids,(80,80))
    native=raw+anchors
    check('Grid rule preserves all width/height logits',torch.equal(ref[...,2:],native[...,2:]))
    check('Grid rule preserves P4/P5 references',torch.equal(ref[:,1],native[:,1]))
    check('Grid rule preserves invalid-anchor infinities',torch.equal(ref[:,2],native[:,2]))
    center=ref[0,0,:2].sigmoid()
    expected=anchors[0,0,:2].sigmoid()+raw[0,0,:2]/80
    check('Grid center matches the specified formula',torch.allclose(center,expected,atol=1e-7,rtol=0))
    gradient=torch.autograd.grad(center.sum(),raw)[0]
    check('Interior center derivative is one grid cell',torch.allclose(gradient[0,0,:2],torch.full((2,),1/80),atol=2e-8,rtol=0))
    for h,w in ((60,60),(80,80),(100,100),(60,100)):
        a=anchors[:,:1].clone();r=raw.detach()[:,:1]
        xy=a[0,0,:2].sigmoid();shift=torch.tensor([1/w,1/h])
        shifted=a.clone();shifted[0,0,:2]=torch.logit(xy+shift)
        base=grid_reference(r,a,torch.tensor([[20]]),(h,w)).sigmoid()
        moved=grid_reference(r,shifted,torch.tensor([[21]]),(h,w)).sigmoid()
        assert torch.allclose((moved-base)[0,0,:2],shift,atol=2e-7,rtol=0)
        assert torch.equal(moved[...,2:],base[...,2:])
    check('Interior translation property holds across square and rectangular grids',True)
    shifted=anchors[:,:1].clone();shifted[0,0,:2]=torch.logit(anchors[0,0,:2].sigmoid()+1/80)
    old_delta=(raw.detach()[:,:1]+shifted).sigmoid()-(raw.detach()[:,:1]+anchors[:,:1]).sigmoid()
    check('Native sigmoid-offset counterexample remains visible',not torch.allclose(old_delta[0,0,:2],torch.full((2,),1/80),atol=1e-5,rtol=0))
    extreme=torch.tensor([[[1e3,-1e3,0.,0.]]],requires_grad=True)
    bounded=grid_reference(extreme,anchors[:,:1],torch.tensor([[20]]),(80,80)).sigmoid()
    gb=torch.autograd.grad(bounded[...,:2].sum(),extreme)[0]
    check('Boundary clamp is finite and explicitly saturates center gradients',torch.isfinite(bounded).all().item() and torch.count_nonzero(gb).item()==0)
    zero=grid_reference(torch.zeros_like(anchors[:,:1]),anchors[:,:1],torch.tensor([[20]]),(80,80)).sigmoid()
    check('Zero offsets recover anchors within predeclared FP32 tolerance',torch.allclose(zero,anchors[:,:1].sigmoid(),atol=1e-7,rtol=0))
    half=grid_reference(raw.detach()[:,:1].half(),anchors[:,:1],torch.tensor([[20]]),(80,80))
    check('Half raw inputs use finite float32 coordinate arithmetic',half.dtype==torch.float32 and torch.isfinite(half).all().item())


def test_models():
    from geometry import ARMS
    from engine.core.workspace import GLOBAL_CONFIG,create
    register_components()
    plans={arm:json.loads((HERE/f'configs/{arm}_seed1.json').read_text(encoding='utf-8')) for arm in ARMS}
    models,criteria={},{}
    states,rngs={},{}
    for arm in ARMS:
        torch.manual_seed(810087)
        model,criterion,_=build_components(plans[arm])
        models[arm],criteria[arm]=model.eval(),criterion
        rngs[arm]=torch.get_rng_state().clone()
        states[arm]=model.state_dict()
    check('All arms consume the same initialization RNG',all(torch.equal(rngs['baseline'],v) for v in rngs.values()))
    check('All arms retain exactly the native state keys and tensors',all(set(s)==set(states['baseline']) and all(torch.equal(v,s[k]) for k,v in states['baseline'].items()) for s in states.values()))
    check('No new model parameters',len({sum(p.numel() for p in m.parameters()) for m in models.values()})==1)
    registry=dict(GLOBAL_CONFIG)
    registry['GeometryDEIM']=dict(registry['GeometryDEIM'],recipe=plans['baseline']['model_recipe'],arm='baseline')
    torch.manual_seed(810087);built=create('GeometryDEIM',registry)
    check('Native registry constructs the exact baseline',all(torch.equal(v,built.state_dict()[k]) for k,v in states['baseline'].items()))
    del built
    image=torch.rand(1,3,640,640)
    with torch.no_grad():
        backbone=models['baseline'].backbone(image)
        plain=models['baseline'].encoder(backbone)
        bundle=models['fine_only'].encoder(backbone)
        check('Explicit encoder bundle preserves every fused feature bitwise',all(torch.equal(a,b) for a,b in zip(plain,bundle['features'])))
        check('Fine branch exposes the actual input projection',torch.equal(bundle['fine_p3'],models['fine_only'].encoder.input_proj[0](backbone[0])))
        outputs={arm:model(image) for arm,model in models.items()}
    check('Fine-only zero box head preserves full native outputs bitwise',all(torch.equal(v,outputs['fine_only'][k]) for k,v in outputs['baseline'].items()))
    for arm in ('grid_only','fine_grid'):
        assert all(torch.allclose(v,outputs[arm][k],atol=1e-4,rtol=0) for k,v in outputs['baseline'].items()),arm
    check('Grid arms zero-head outputs stay within predeclared 1e-4 FP32 tolerance',True)
    check('Eval outputs retain only native prediction fields',all(set(o)==set(outputs['baseline']) for o in outputs.values()))
    check('Fine feature references are released after forward',models['fine_grid'].decoder._fine_p3 is None and models['fine_only'].decoder._fine_p3 is None)
    # Direct decoder entrance checks on two non-640 grids; no full detector forwards.
    a,b=models['baseline'].decoder,models['fine_grid'].decoder
    a.train();b.train();b.audit_trace=True
    for shapes in ([[20,28],[10,14],[5,7]],[[24,32],[12,16],[6,8]]):
        n=sum(h*w for h,w in shapes)
        memory=torch.randn(2,n,256)
        fine=torch.randn(2,256,*shapes[0])
        dn_content=torch.randn(2,8,256);dn_reference=torch.randn(2,8,4)
        b._fine_p3=fine
        na=a._get_decoder_input(memory,shapes,dn_content,dn_reference)
        nb=b._get_decoder_input(memory,shapes,dn_content,dn_reference)
        assert torch.equal(na[0],nb[0]) and torch.equal(na[4],nb[4])
        assert torch.equal(na[1][:,:8],nb[1][:,:8])
        assert torch.equal(nb[0][:,:8],dn_content)
        assert not nb[1].requires_grad and nb[2][0].requires_grad
        p3=b.last_trace['ids'] < shapes[0][0]*shapes[0][1]
        assert p3.any() and (~p3).any()
        assert torch.equal(b.last_trace['geometry'][~p3],b.last_trace['semantic'][~p3])
        b._fine_p3=None
    check('Multi-scale batch2 entrance preserves semantic scores/content, coarse vectors and DN prefix',True)
    check('Encoder boxes retain gradients but decoder references remain detached',True)
    b.audit_trace=False;b.last_trace=None
    base,candidate=models['baseline'],models['fine_grid']
    before={k:p.detach().clone() for k,p in candidate.named_parameters()}
    base.train();candidate.train()
    for model in (base,candidate):
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    target=[{'labels':torch.tensor([3]),'boxes':torch.tensor([[.5,.5,.2,.2]])}]
    with torch.no_grad():
        torch.manual_seed(810088);native_out=base(image,target)
        native_losses=criteria['baseline'](native_out,target)
    torch.manual_seed(810088);candidate_out=candidate(image,target)
    losses=criteria['fine_grid'](candidate_out,target)
    check('Training preserves native output and loss keys',set(candidate_out)==set(native_out) and set(losses)==set(native_losses))
    check('Training keeps DN and all auxiliary outputs',candidate_out['dn_meta']['dn_num_split']==native_out['dn_meta']['dn_num_split'] and len(candidate_out['aux_outputs'])==len(native_out['aux_outputs']))
    check('Native losses remain finite on full synthetic training forward',all(torch.isfinite(v).all().item() for v in losses.values()))
    sum(losses.values()).backward()
    check('Full native backward reaches the encoder box head with finite gradients',all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in candidate.decoder.enc_bbox_head.parameters()))
    check('No model parameter was assigned or updated by tests',all(torch.equal(p,before[k]) for k,p in candidate.named_parameters()))
    check('Shared source binding remains intact',bool(verify_sources()))


if __name__=='__main__':
    begin=time.time();receipt={'status':'RUNNING','checks':CHECKS,'optimizer_constructions':0,'parameter_updates':0,'checkpoint_loads':0,'scientific_efficacy_claim':False}
    path=HERE/('engineering_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    try:
        activate_snapshot();test_helpers();test_models()
        receipt.update(status='PASS',full_synthetic_detector_forwards=6,
                       additional_components={'backbone_forwards':1,'encoder_forwards':2,'input_projection_forwards':1,'decoder_entrance_calls':4},elapsed_seconds=time.time()-begin)
    except Exception:
        receipt.update(status='FAIL',traceback=traceback.format_exc(),elapsed_seconds=time.time()-begin)
        raise
    finally:
        with path.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
        print(json.dumps(receipt,ensure_ascii=False,indent=2))
