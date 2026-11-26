"""Engineering only: fresh CPU models and synthetic tensors, no updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import time
import traceback
import torch
from torch.nn import functional as F
from runtime import HERE, activate_snapshot, build_components, register_components, verify_sources

torch.set_num_threads(4)
CHECKS = []
FULL_FORWARDS = 0
DECODER_FORWARDS = 0
DETAILS = {}

def check(name, condition):
    assert condition, name
    CHECKS.append(name)

def helpers():
    from residual import selected_patches, LocalGeometryResidual
    torch.manual_seed(810209)
    x=torch.arange(2*4*3*5,dtype=torch.float32).reshape(2,4,3,5).requires_grad_()
    bi=torch.tensor([0,0,1]); ids=torch.tensor([0,7,14])
    patches=selected_patches(x,bi,ids)
    expected=torch.zeros(3,9,4)
    for k,(b,i) in enumerate(zip(bi.tolist(),ids.tolist())):
        for j,(dy,dx) in enumerate(( (dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) )):
            yy,xx=i//5+dy,i%5+dx
            if 0<=yy<3 and 0<=xx<5: expected[k,j]=x.detach()[b,:,yy,xx]
    check('Selected patches equal independent border-aware loop on rectangular grid',torch.equal(patches,expected))
    before_rng=torch.get_rng_state().clone()
    block=LocalGeometryResidual(4,3)
    check('Residual construction does not consume global RNG',torch.equal(before_rng,torch.get_rng_state()))
    pooled=(patches*block.kernel[None]).sum(1)
    conv=F.conv2d(x,block.kernel.transpose(0,1).reshape(4,1,3,3),padding=1,groups=4)
    reference=conv.flatten(2).transpose(1,2)[bi,ids]
    check('Selected-only pooling matches independent grouped convolution',torch.allclose(pooled,reference,atol=1e-5,rtol=1e-6))
    src_grad=torch.autograd.grad(pooled.sum(),x,retain_graph=True)[0]
    check('Local neighborhood gathering remains differentiable',torch.isfinite(src_grad).all().item() and src_grad.abs().sum().item()>0)
    semantic=torch.randn(2,3,4,requires_grad=True)
    ids2=torch.tensor([[0,7,15],[14,3,16]])
    valid=torch.ones(1,20,1,dtype=torch.bool); valid[0,3]=False
    delta=block(semantic,x,ids2,valid)
    check('Output initialization is exactly zero for arbitrary finite semantic/source inputs',torch.equal(delta,torch.zeros_like(delta)))
    coarse=(ids2>=15)
    gc=torch.autograd.grad(delta[coarse].sum(),block.output_bias,retain_graph=True)[0]
    gp=torch.autograd.grad(delta[~coarse].sum(),block.output_bias,retain_graph=True)[0]
    check('P4/P5 entries have no residual-bias path',gc.eq(0).all().item())
    check('Only valid P3 entries receive the residual-bias path',gp.eq(3).all().item())
    gradients=torch.autograd.grad(delta.sum(),(block.output_weight,block.mix_weight,block.kernel),retain_graph=True)
    check('Zero output layer receives a finite nonzero initial gradient',torch.isfinite(gradients[0]).all().item() and gradients[0].abs().sum().item()>0)
    check('Upstream new layers are gated initially by only the zero output layer',gradients[1].eq(0).all().item() and gradients[2].eq(0).all().item())
    empty=block(semantic,x,torch.full_like(ids2,15),valid)
    check('No selected P3 is a valid empty correction',torch.equal(empty,torch.zeros_like(empty)))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        low=block(semantic,x,ids2,valid)
    check('Local block supports finite CPU bfloat16 autocast',torch.isfinite(low).all().item() and low.eq(0).all().item())

def models():
    global FULL_FORWARDS
    from geometry import ARMS
    from engine.core.workspace import GLOBAL_CONFIG,create
    register_components()
    plans={a:json.loads((HERE/f'configs/{a}_seed1.json').read_text(encoding='utf-8')) for a in ARMS}
    models,criteria,states,rngs={},{},{},{}
    for a in ARMS:
        torch.manual_seed(810210)
        model,criterion,_=build_components(plans[a])
        models[a],criteria[a]=model.eval(),criterion
        states[a]=model.state_dict()
        rngs[a]=torch.get_rng_state().clone()
    native=states['baseline']
    check('All native tensors start identically across all three arms',all(
        all(torch.equal(v,s[k]) for k,v in native.items()) for s in states.values()))
    check('All arms retain the same global initialization RNG state',all(torch.equal(rngs['baseline'],r) for r in rngs.values()))
    expected={'decoder.geometry_residual.'+k for k in ('kernel','mix_weight','mix_bias','output_weight','output_bias')}
    check('Candidate/control add exactly the five reviewed parameter tensors',all(set(states[a])-set(native)==expected for a in ARMS[1:]))
    check('Candidate/control added tensors start identically',all(torch.equal(states['fine_residual'][k],states['fused_residual'][k]) for k in expected))
    baseline_count=sum(p.numel() for p in models['baseline'].parameters())
    check('Both residual arms add exactly 35396 parameters',all(sum(p.numel() for p in models[a].parameters())-baseline_count==35396 for a in ARMS[1:]))
    registry=dict(GLOBAL_CONFIG)
    registry['ResidualGeometryDEIM']=dict(registry['ResidualGeometryDEIM'],recipe=plans['baseline']['model_recipe'],arm='baseline')
    torch.manual_seed(810210); built=create('ResidualGeometryDEIM',registry)
    check('Native registry can construct the exact baseline',all(torch.equal(v,built.state_dict()[k]) for k,v in native.items()))
    del built
    image=torch.rand(1,3,640,640)
    with torch.no_grad():
        features=models['baseline'].backbone(image)
        plain=models['baseline'].encoder(features)
        bundle=models['fine_residual'].encoder(features)
        check('Feature bundle keeps every original fused feature bitwise',all(torch.equal(a,b) for a,b in zip(plain,bundle['features'])))
        check('Fine source is the actual pre-fusion projection',torch.equal(bundle['fine_p3'],models['fine_residual'].encoder.input_proj[0](features[0])))
        outputs={}
        for a,m in models.items():
            outputs[a]=m(image); FULL_FORWARDS+=1
    check('Zero residual preserves full native evaluation predictions bitwise',all(
        set(o)==set(outputs['baseline']) and all(torch.equal(v,o[k]) for k,v in outputs['baseline'].items())
        for o in outputs.values()))
    check('Per-forward source references are released',all(models[a].decoder._geometry_map is None for a in ARMS[1:]))
    # Native entrance on two rectangular grids; no trained model or image data.
    for shapes in ([[20,28],[10,14],[5,7]],[[24,32],[12,16],[6,8]]):
        memory=torch.randn(2,sum(h*w for h,w in shapes),256)
        source=torch.randn(2,256,*shapes[0])
        dc=torch.randn(2,8,256); dr=torch.randn(2,8,4)
        models['baseline'].decoder.train()
        original=models['baseline'].decoder._get_decoder_input(memory,shapes,dc,dr)
        for a in ARMS[1:]:
            d=models[a].decoder; d.train(); d.audit_trace=True; d._geometry_map=source
            out=d._get_decoder_input(memory,shapes,dc,dr)
            assert all(torch.equal(original[i],out[i]) for i in (0,1,4))
            assert torch.equal(original[2][0],out[2][0]) and torch.equal(original[3][0],out[3][0])
            assert not out[1].requires_grad and out[2][0].requires_grad
            assert torch.equal(out[0][:,:8],dc) and torch.equal(out[1][:,:8],dr)
            assert d.last_trace['delta'].eq(0).all()
            d._geometry_map=None; d.audit_trace=False; d.last_trace=None
    check('Batch2 rectangular entrance retains scores, content, references and DN prefix exactly',True)
    check('Encoder auxiliary boxes carry gradients while decoder references stay detached',True)
    target=[{'labels':torch.tensor([3,6]),'boxes':torch.tensor([[.45,.46,.03,.04],[.62,.55,.02,.03]])}]
    losses_by_arm={}; output_keys={}; dn_meta={}
    for a,m in models.items():
        m.train()
        for module in m.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
        before={k:p.detach().clone() for k,p in m.named_parameters()}
        torch.manual_seed(810211)
        if a=='baseline':
            with torch.no_grad():
                out=m(image,target); FULL_FORWARDS+=1
                losses=criteria[a](out,target)
        else:
            m.decoder.audit_trace=True
            out=m(image,target); FULL_FORWARDS+=1
            losses=criteria[a](out,target)
        output_keys[a]=set(out)
        dn_meta[a]=out['dn_meta']['dn_num_split']
        losses_by_arm[a]={k:v.detach().clone() for k,v in losses.items()}
        assert all(torch.isfinite(v).all() for v in losses.values()),a
        if a!='baseline':
            sum(losses.values()).backward()
            b=m.decoder.geometry_residual
            p3_count=int((m.decoder.last_trace['ids']<6400).sum())
            DETAILS[a+'_whole_detector_selected_P3']=p3_count
            if p3_count==0:
                check(a+' whole-detector fixture correctly gives zero residual gradient when no P3 is selected',
                      b.output_weight.grad is not None and b.output_weight.grad.eq(0).all().item())
            else:
                check(a+' whole-detector residual gradient is finite',b.output_weight.grad is not None and torch.isfinite(b.output_weight.grad).all().item())
            check(a+' native encoder box head retains finite gradients',all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.decoder.enc_bbox_head.parameters()))
            check(a+' every new parameter has a finite gradient tensor',all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in b.parameters()))
        check(a+' parameters unchanged after forward/backward',all(torch.equal(p,before[k]) for k,p in m.named_parameters()))
    check('Training outputs, DN split and loss keys match the native baseline',all(output_keys[a]==output_keys['baseline'] and dn_meta[a]==dn_meta['baseline'] and set(losses_by_arm[a])==set(losses_by_arm['baseline']) for a in ARMS))
    check('All native loss values start exactly equal, no new loss terms',all(all(torch.equal(v,losses_by_arm[a][k]) for k,v in losses_by_arm['baseline'].items()) for a in ARMS))
    check('Shared native sources still match their SHA binding',bool(verify_sources()))

def decoder_integration():
    """Full native readout/decoder/criterion with synthetic multilevel features.

The v1 image fixture selects P5 only; this separately exercises the P3 path.
No weights, logits, query indices, matcher, or losses are replaced.
"""
    global DECODER_FORWARDS
    from geometry import ARMS
    torch.manual_seed(810212)
    feats=[torch.randn(1,256,h,w) for h,w in ((80,80),(40,40),(20,20))]
    fine=torch.randn(1,256,80,80)
    target=[{'labels':torch.tensor([3,6]),'boxes':torch.tensor([[.45,.46,.03,.04],[.62,.55,.02,.03]])}]
    references={}; losses_saved={}; grads={}
    for a in ARMS:
        plan=json.loads((HERE/f'configs/{a}_seed1.json').read_text(encoding='utf-8'))
        torch.manual_seed(810210)
        model,criterion,_=build_components(plan)
        decoder=model.decoder.train()
        before={k:p.detach().clone() for k,p in decoder.named_parameters()}
        if a!='baseline':decoder.audit_trace=True
        torch.manual_seed(810213)
        inputs=feats if a=='baseline' else {'features':feats,'fine_p3':fine}
        out=decoder(inputs,target); DECODER_FORWARDS+=1
        enc=out['enc_aux_outputs'][0]['pred_boxes']; enc.retain_grad()
        losses=criterion(out,target)
        losses_saved[a]={k:v.detach().clone() for k,v in losses.items()}
        references[a]=enc.detach().clone()
        sum(losses.values()).backward()
        if a!='baseline':
            ids=decoder.last_trace['ids']; mask=ids<6400
            count=int(mask.sum()); active=int((enc.grad[mask].abs().sum(-1)>0).sum())
            DETAILS[a+'_decoder_selected_P3']=count
            DETAILS[a+'_decoder_supervised_P3']=active
            check(a+' nonempty P3 native selection and regression precondition',count>0 and active>0)
            block=decoder.geometry_residual
            check(a+' complete native readout/decoder/loss reaches new output layer',
                  block.output_weight.grad is not None and torch.isfinite(block.output_weight.grad).all().item()
                  and block.output_weight.grad.abs().sum().item()>0)
            grads[a]=block.output_weight.grad.detach().clone()
        check(a+' decoder-integration parameters unchanged',all(torch.equal(p,before[k]) for k,p in decoder.named_parameters()))
        del model,criterion,decoder,out,losses,enc
    check('P3-active full decoder starts with identical encoder boxes and native loss values',
          all(torch.equal(references[a],references['baseline']) and all(torch.equal(v,losses_saved[a][k])
              for k,v in losses_saved['baseline'].items()) for a in ARMS))
    check('Fine/fused source produces distinct new-head gradients despite equal zero-residual predictions',
          not torch.equal(grads['fine_residual'],grads['fused_residual']))

if __name__=='__main__':
    started=time.monotonic()
    receipt=dict(status='RUNNING',checks=CHECKS,optimizer_constructions=0,
                 checkpoint_loads=0,parameter_updates_after_construction=0,
                 scientific_efficacy_claim=False)
    path=HERE/('engineering_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    try:
        activate_snapshot(); helpers(); models(); decoder_integration()
        receipt['status']='PASS'
    except Exception:
        receipt.update(status='FAIL',traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-started,
                       full_synthetic_detector_forwards=FULL_FORWARDS,
                       full_synthetic_decoder_forwards=DECODER_FORWARDS,details=DETAILS,
                       prior_failure_retained='engineering_20261008_194257.json; test_residual_v1_failure.py',
                       prior_failure_diagnosis='initial_gradient_diagnosis.json: original fixture selects 300 P5, zero P3')
        with path.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
        print(json.dumps(receipt,ensure_ascii=False,indent=2))
