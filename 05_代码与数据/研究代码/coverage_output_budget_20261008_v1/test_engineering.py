"""CPU engineering checks only. No optimizer, model fitting, or parameter updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, time, traceback, random, itertools
from pathlib import Path
from fractions import Fraction
import torch
from matching import coverage_assignment
from losses import budget_boundary, admission_losses, output_budget_loss

HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)
CHECKS = []

def check(name, condition):
    assert condition, name
    CHECKS.append(name)

def assignment_objective(pairs, priorities, edge_iou, token_rank, gt_rank, ng):
    return (len(pairs), sum((Fraction(float(priorities[i])) for i,j in pairs), Fraction()),
            sum((Fraction(float(edge_iou[i,j])) for i,j in pairs), Fraction()),
            -sum(token_rank[i]*(ng+1)+gt_rank[j] for i,j in pairs))

def test_assignment():
    rng = random.Random(810083)
    for trial in range(80):
        n, g, k = rng.randrange(1,6), rng.randrange(1,5), rng.randrange(1,5)
        scores = [rng.choice([-3., 0., 1e-10, .5, 2.]) for _ in range(n)]
        edges = [(i,j,rng.choice([.5,.75,1.])) for i in range(n) for j in range(g) if rng.random()<.55]
        chosen = coverage_assignment(scores, edges, k)
        tr = {v:i for i,v in enumerate(sorted({i for i,j,v in edges}))}
        gr = {v:i for i,v in enumerate(sorted({j for i,j,v in edges}))}
        weights = {(i,j):v for i,j,v in edges}
        best = assignment_objective([], scores, weights, tr, gr, len(gr))
        for m in range(1,min(n,g,k)+1):
            for qs in itertools.combinations(range(n),m):
                for gs in itertools.permutations(range(g),m):
                    p = list(zip(qs,gs))
                    if all(e in weights for e in p):
                        best=max(best,assignment_objective(p,scores,weights,tr,gr,len(gr)))
        assert assignment_objective(chosen,scores,weights,tr,gr,len(gr)) == best
        assert chosen == coverage_assignment(scores,list(reversed(edges)),k)
    check('80 exhaustive graph comparisons, exact objectives and deterministic ties', True)
    check('Cardinality dominates even very negative priority',len(coverage_assignment([-1e20,-1e20],[(0,0,1.),(1,1,1.)],2))==2)
    check('Cardinality cap enforced',len(coverage_assignment([1.,2.],[(0,0,1.),(1,1,1.)],1))==1)

def test_losses():
    s=torch.tensor([10.,9.,7.9,0.],requires_grad=True)
    loss=budget_boundary(s,[0,2],2)
    grad=torch.autograd.grad(loss,s)[0]
    check('Documented surrogate counterexample loss',abs(loss.item()-.6)<1e-6)
    check('Boundary competitor receives gradient',torch.allclose(grad,torch.tensor([0.,.5,-.5,0.])))
    check('Shift invariant gradient sum',grad.sum().item()==0)
    check('Empty positive set is differentiable zero',budget_boundary(s,[],2).requires_grad)
    check('Budget covers full universe',budget_boundary(s,[0,1],4).item()==0)
    check('M equals K uses highest negative boundary',budget_boundary(torch.tensor([0.,0.,2.]),[0,1],2).item()>2)
    try: budget_boundary(s,[0,0],2)
    except ValueError: check('Duplicate positives rejected',True)
    else: raise AssertionError('Duplicate positives accepted')
    for ids in ([0],[0,1],[0,1,2]):
        z=torch.tensor([4.,3.,2.,1.,0.])
        if budget_boundary(z,ids,3).item()==0:
            assert set(ids)<=set(z.topk(3).indices.tolist())
    check('Zero-hinge positive entries satisfy output budget',True)
    # Correct class is SECOND, still gets supervised if native matching selects it.
    logits=torch.tensor([[[4.,1.],[3.,2.],[0.,0.]]],requires_grad=True)
    boxes=torch.tensor([[[.5,.5,.2,.2],[.2,.2,.1,.1],[.8,.8,.1,.1]]],requires_grad=True)
    targets=[{'boxes':torch.tensor([[.5,.5,.2,.2]]),'labels':torch.tensor([1])}]
    result,meta=output_budget_loss({'pred_logits':logits,'pred_boxes':boxes},targets,[(torch.tensor([0]),torch.tensor([0]))],budget=2)
    gs=torch.autograd.grad(result['loss_output_budget'],(logits,boxes),allow_unused=True)
    check('Non-argmax true class is included',meta[0]['positive_entries']==[1] and gs[0][0,0,1]<0)
    check('Geometry threshold does not backpropagate to boxes',gs[1] is None)
    # Admission gradient routing, on independent synthetic tensors only.
    h=torch.randn(1,4,3,requires_grad=True);w=torch.randn(1,3,requires_grad=True)
    native=torch.randn(1,4,2,requires_grad=True)
    delta=torch.nn.functional.linear(h,w).squeeze(-1)
    priorities=native.amax(-1).detach()+delta
    aux={'priority':priorities,'residual':delta,'classes':torch.zeros(1,4,dtype=torch.long),
         'boxes':boxes.detach()[:,[0,1,2,2]],'valid':torch.ones(1,4,dtype=torch.bool)}
    target0=[dict(targets[0],labels=torch.tensor([0]))]
    losses,_=admission_losses(aux,target0,'cover',budget=2)
    g=torch.autograd.grad(sum(losses.values()),(h,w,native),allow_unused=True)
    check('Admission gradients reach residual head and features, not native score logits',g[0] is not None and g[1] is not None and g[2] is None)
    half=torch.tensor([1.,0.,2.,-1.],dtype=torch.float16,requires_grad=True)
    v=budget_boundary(half,[1],2);gh=torch.autograd.grad(v,half)[0]
    check('Half input loss uses finite FP32 computation',v.dtype==torch.float32 and torch.isfinite(gh).all().item())

def clone_tree(x):
    if torch.is_tensor(x): return x.clone()
    if isinstance(x,dict): return {k:clone_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [clone_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(clone_tree(v) for v in x)
    return x

def test_native_interface():
    from runtime import activate_snapshot,build_components,verify_sources,register_components
    activate_snapshot()
    plans={arm:json.loads((HERE/'configs'/f'{arm}_seed1.json').read_text(encoding='utf-8')) for arm in ('baseline','admission_cover','output_budget')}
    torch.manual_seed(810084);base,crit,_=build_components(plans['baseline']);rng0=torch.get_rng_state().clone()
    # Exercise the same native registry construction used by the training runner.
    register_components()
    from engine.core.workspace import GLOBAL_CONFIG, create
    registry = dict(GLOBAL_CONFIG)
    p = plans['admission_cover']
    registry['BudgetDEIM'] = dict(registry['BudgetDEIM'], recipe=p['model_recipe'], arm=p['arm'])
    registry['BudgetCriterion'] = dict(registry['BudgetCriterion'], recipe=p['criterion_recipe'], arm=p['arm'], **p['budget_loss'])
    torch.manual_seed(810084);candidate=create('BudgetDEIM',registry);ca=create('BudgetCriterion',registry)
    check('Native registry constructs model and criterion without inherited injection fields',candidate.budget_arm == ca.budget_arm == 'admission_cover')
    check('Zero residual construction preserves native RNG consumption',torch.equal(rng0,torch.get_rng_state()))
    check('All common model state tensors match exactly',all(torch.equal(v,candidate.state_dict()[k]) for k,v in base.state_dict().items()))
    from native_adapter import BudgetCriterion
    cb=BudgetCriterion(plans['output_budget']['criterion_recipe'],'output_budget',**plans['output_budget']['budget_loss'])
    parameters_before={k:p.detach().clone() for k,p in candidate.named_parameters()}
    image=torch.rand(1,3,640,640)
    base.eval();candidate.eval()
    with torch.no_grad():
        a=base(image);b=candidate(image)
    check('Zero admission residual preserves full native eval logits and boxes bitwise',all(torch.equal(a[k],b[k]) for k in a))
    check('Inference does not export dense labels or retain a training graph',set(b)==set(a) and candidate.decoder._admission_work is None)
    target=[{'labels':torch.tensor([3]),'boxes':torch.tensor([[.5,.5,.2,.2]])}]
    base.train();candidate.train()
    # Only fresh synthetic models; keep BN running buffers fixed in this interface test.
    for m in list(base.modules())+list(candidate.modules()):
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    with torch.no_grad():
        torch.manual_seed(810085);a=base(image,target)
    torch.manual_seed(810085);b=candidate(image,target)
    with torch.no_grad():
        check('DN remains present and unchanged at zero residual',a['dn_meta']['dn_num_split']==b['dn_meta']['dn_num_split'])
        check('Training final and encoder outputs identical',torch.equal(a['pred_logits'],b['pred_logits']) and torch.equal(a['enc_aux_outputs'][0]['pred_boxes'],b['enc_aux_outputs'][0]['pred_boxes']))
        native_losses=crit(clone_tree(a),target)
        output_losses=cb(clone_tree(a),target)
    cover_losses=ca(clone_tree(b),target)
    check('Admission keeps every native loss value unchanged',all(torch.equal(v,cover_losses[k]) for k,v in native_losses.items()))
    check('Output-budget keeps every native loss value unchanged',all(torch.equal(v,output_losses[k]) for k,v in native_losses.items()))
    check('New losses finite with DN and auxiliary layers',all(torch.isfinite(v).all().item() for v in list(cover_losses.values())+list(output_losses.values())))
    check('Native matcher is focal as in formal baseline',crit.matcher.use_focal_loss)
    sum(cover_losses.values()).backward()
    check('Full synthetic native-plus-admission backward reaches the new head with finite gradient',candidate.decoder.admission_weight.grad is not None and torch.isfinite(candidate.decoder.admission_weight.grad).all().item())
    check('Full synthetic native-plus-admission backward retains finite native box-head gradients',all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in candidate.decoder.enc_bbox_head.parameters()))
    check('No fresh model parameter was updated',all(torch.equal(p,parameters_before[k]) for k,p in candidate.named_parameters()))
    check('Shared sources unchanged after execution',bool(verify_sources()))

if __name__=='__main__':
    begun=time.time();receipt={'status':'RUNNING','checks':CHECKS,'optimizer_constructions':0,'optimizer_updates':0,'trained_checkpoint_loads':0,'scientific_efficacy_claim':False}
    target=HERE/('engineering_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    try:
        test_assignment();test_losses();test_native_interface()
        receipt.update(status='PASS',fresh_synthetic_detector_forwards=4,elapsed_seconds=time.time()-begun)
    except Exception:
        receipt.update(status='FAIL',traceback=traceback.format_exc(),elapsed_seconds=time.time()-begun)
        raise
    finally:
        with target.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
        print(json.dumps(receipt,ensure_ascii=False,indent=2))
