"""Two preregistered DDF candidates. Local CPU independent virtual endpoints only.

No optimizer.step/checkpoint writes. Native frozen model/loss source is imported
unchanged. Corrections replace ONLY ordinary positive DDF adjoints at the actual
non-detached corner residuals; all other gradients remain native.
"""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import datetime,gc,hashlib,itertools,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/source_allocation_frozen_20261020_v1'))
import diagnostic as d
from engine.deim.dfine_utils import bbox2distance,weighting_function
torch,np=d.torch,d.np
OUT=ROOT/'ddf_candidates_frozen'
OLD=WORK/'英文小论文/data/10_来源分配冻结诊断_20261020_v1'
ARMS=['original','half_positive','off_positive','teacherbetter_gate','single_total',
      'multi_layer','matched_norm','relocated_target','GT_two_bin']

def sha(p):return d.native.sha(p)
def save(p,v):d.native.save_json(p,v)
def progress(**kw):d.progress(**kw)

def project_halfspaces(g,c):
    """Exact small active-set QP in float64, per row. c[n,m,d] h[n,d]>=0.

    Normalize row directions and input scale, enumerate all <=3 active sets.
    This avoids sequential PCGrad's order dependence and needs no scipy/runtime.
    """
    original_dtype=g.dtype;g=g.double();c=c.double()
    size=g.norm(dim=-1,keepdim=True);unit=g/size.clamp_min(1e-30)
    c=c/c.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    best=unit.clone();cost=torch.full((len(g),),float('inf'),dtype=torch.float64)
    for count in range(c.shape[1]+1):
        for active in itertools.combinations(range(c.shape[1]),count):
            if not active:
                h=unit;nonnegative=torch.ones(len(g),dtype=torch.bool)
            else:
                a=c[:,active,:];gram=a@a.transpose(1,2)
                lam=-(torch.linalg.pinv(gram,rtol=1e-11)@(a@unit.unsqueeze(-1))).squeeze(-1)
                h=unit+(a.transpose(1,2)@lam.unsqueeze(-1)).squeeze(-1)
                nonnegative=(lam>=-1e-8).all(-1)
            feasible=((c@h.unsqueeze(-1)).squeeze(-1)>=-1e-8).all(-1)&nonnegative
            distance=(h-unit).square().sum(-1)
            better=feasible&(distance<cost)
            best[better]=h[better];cost[better]=distance[better]
    assert torch.isfinite(cost).all(),'QP has no enumerated feasible KKT point'
    assert float((c@best.unsqueeze(-1)).min())>=-2e-8
    return (best*size).to(original_dtype)

def resample_mass(p,positions,bins):
    """Barycentric mass split on sorted nonuniform support, explicit clipping."""
    clipped=positions.clamp(float(bins[0]),float(bins[-1]))
    hi=torch.searchsorted(bins.contiguous(),clipped.contiguous(),right=True).clamp(1,len(bins)-1)
    lo=hi-1;fraction=(clipped-bins[lo])/(bins[hi]-bins[lo])
    out=torch.zeros_like(p)
    out.scatter_add_(-1,lo,p*(1-fraction));out.scatter_add_(-1,hi,p*fraction)
    mass=(p*((positions<bins[0])|(positions>bins[-1]))).sum(-1)
    return out,mass

def inverse_temperature_target(physical,T=5):
    q=physical.pow(1/T)
    return q/q.sum(-1,keepdim=True)

def synthetic_tests():
    g=torch.tensor([[-1.,2.],[1.,2.],[0.,0.]],dtype=torch.float64)
    c=torch.tensor([[[1.,0.],[0.,1.]]]*3,dtype=torch.float64)
    h=project_halfspaces(g,c)
    assert torch.allclose(h,torch.tensor([[0.,2.],[1.,2.],[0.,0.]],dtype=torch.float64),atol=1e-12)
    # Same normal twice, and opposing normals: singular Gram, nonzero nullspace.
    hh=project_halfspaces(g,torch.tensor([[[1.,0.],[-1.,0.],[1.,0.]]]*3,dtype=torch.float64))
    assert float(hh[:,0].abs().max())<1e-12 and torch.equal(hh[:,1],g[:,1])
    # z0=r0,z1=r0+r1: a later gradient can move the first geometry.
    r=torch.tensor([0.,0.],dtype=torch.float64,requires_grad=True)
    z0=r[0];z1=r.sum();gg=torch.autograd.grad(-z1,r,retain_graph=True)[0]
    c0=torch.autograd.grad(z0,r,retain_graph=True)[0];c1=torch.autograd.grad(-z1,r)[0]
    assert float(c0@gg)<0 and float(c1@gg)>0
    assert float(c0@project_halfspaces(gg[None],torch.stack([c0,c1])[None])[0])>=-1e-12
    bins=torch.tensor([-4.,-1.,0.,.3,4.],dtype=torch.float64)
    p=torch.tensor([[0.,.2,.5,.3,0.]],dtype=torch.float64)
    moved,mass=resample_mass(p,bins[None]+.2,bins)
    assert float(mass)==0 and abs(float(moved.sum())-1)<1e-14
    assert abs(float((moved*bins).sum()-(p*bins).sum()-.2))<1e-14
    q=inverse_temperature_target(moved)
    recovered=q.pow(5);recovered/=recovered.sum(-1,keepdim=True)
    assert float((recovered-moved).abs().max())<1e-14
    return dict(QP_identity_zero_singular=True,cumulative_counterexample=True,
                barycentric_mass_mean=True,physical_temperature_inverse=True)

def pos_distill(outputs,targets,union,criterion):
    idx=criterion._get_src_permutation_idx(union)
    gt=torch.cat([t['boxes'][gi] for t,(qi,gi) in zip(targets,union)])
    B,Q=outputs['pred_boxes'].shape[:2];K=len(gt);T=5.
    norm=(4*K*8/B)**.5/((4*K*8/B)**.5+(4*(B*Q-K)*8/B)**.5)/(4*K)
    bins=weighting_function(criterion.reg_max,outputs['up'],outputs['reg_scale']).detach()
    teacher=outputs['pred_corners'][idx].detach().reshape(K,4,33)
    physical=teacher.softmax(-1)
    t_iou=d.box_iou(d.box_cxcywh_to_xyxy(outputs['pred_boxes'][idx].detach()),d.box_cxcywh_to_xyxy(gt))[0].diagonal()
    totals={k:0 for k in ['original','teacherbetter_gate','relocated_target','GT_two_bin']}
    diagnostics=[]
    for layer,o in enumerate(outputs['aux_outputs']):
        student=o['pred_corners'][idx].reshape(K,4,33)
        s_iou=d.box_iou(d.box_cxcywh_to_xyxy(o['pred_boxes'][idx].detach()),d.box_cxcywh_to_xyxy(gt))[0].diagonal().detach()
        weight=s_iou[:,None]*norm*criterion.weight_dict['loss_ddf']
        with torch.no_grad():
            ref=o['ref_points'][idx].detach();scale=float(outputs['reg_scale'].abs())
            xyxy=d.box_cxcywh_to_xyxy(gt)
            desired=torch.stack([(ref[:,0]-xyxy[:,0])/(ref[:,2]/scale+1e-16)-scale/2,
                                 (ref[:,1]-xyxy[:,1])/(ref[:,3]/scale+1e-16)-scale/2,
                                 (xyxy[:,2]-ref[:,0])/(ref[:,2]/scale+1e-16)-scale/2,
                                 (xyxy[:,3]-ref[:,1])/(ref[:,3]/scale+1e-16)-scale/2],-1)
            shift=desired-(physical*bins).sum(-1)
            relocated,clip_mass=resample_mass(physical,bins[None,None,:]+shift[...,None],bins)
            two,_=resample_mass(torch.ones_like(desired[...,None]),desired[...,None],bins) if False else (None,None)
            # Same native FGL two-bin target, including native out-of-support policy.
            labels,wr,wl=bbox2distance(ref,xyxy,criterion.reg_max,outputs['reg_scale'],outputs['up'])
            labels=labels.long().reshape(K,4);two=torch.zeros_like(physical)
            two.scatter_add_(-1,labels[...,None],wl.reshape(K,4,1))
            two.scatter_add_(-1,(labels+1)[...,None],wr.reshape(K,4,1))
            target={'original':(teacher/T).softmax(-1),'teacherbetter_gate':(teacher/T).softmax(-1),
                    'relocated_target':inverse_temperature_target(relocated),
                    'GT_two_bin':inverse_temperature_target(two)}
            errors=((relocated*bins).sum(-1)-desired).abs()
            support_in=(desired>=bins[0])&(desired<=bins[-1])
            # Clipping means this translation does NOT generally preserve the GT mean.
            diagnostics.append(dict(layer=layer,pairs=K,teacherbetter=int((t_iou>s_iou).sum()),
                desired_edges_out_of_support=int((~support_in).sum()),total_edges=4*K,
                relocated_clipped_mass_sum=float(clip_mass.sum()),
                relocated_GT_mean_absolute_error_sum=float(errors.sum()),
                relocated_GT_mean_max_error=float(errors.max()),
                relocated_two_bin_L1_sum=float((relocated-two).abs().sum()),
                inverse_temperature_recovery_max_error=float(((target['relocated_target'].pow(T)/target['relocated_target'].pow(T).sum(-1,keepdim=True))-relocated).abs().max())))
        for name,prob in target.items():
            kl=torch.nn.functional.kl_div((student/T).log_softmax(-1),prob,reduction='none').sum(-1)*T*T
            w=weight*((t_iou>s_iou).to(weight.dtype)[:,None] if name=='teacherbetter_gate' else 1.)
            totals[name]=totals[name]+(kl*w).sum()
        # Independently reconstruct full native DDF (positive + negative) value.
        z=o['pred_corners'].reshape(-1,33);tt=o['teacher_corners'].detach().reshape(-1,33)
        mask=torch.zeros(B,Q,dtype=torch.bool);mask[idx]=True
        w=o['teacher_logits'].sigmoid().max(-1).values.clone();w[idx]=s_iou
        m4=mask[...,None].expand(-1,-1,4).reshape(-1)
        v=(w[...,None].expand(-1,-1,4).reshape(-1).detach()*T*T*
           torch.nn.functional.kl_div((z/T).log_softmax(-1),(tt/T).softmax(-1),reduction='none').sum(-1))
        positive=(m4.sum()*8/B)**.5;negative=((~m4).sum()*8/B)**.5
        full=((v[m4].mean()*positive+v[~m4].mean()*negative)/(positive+negative))*criterion.weight_dict['loss_ddf']
        diagnostics[-1]['full_native_ddf_reconstruction']=float(full.detach())
    return totals,diagnostics,idx

def capture_losses(criterion,outputs,targets):
    criterion.mark(outputs);trace=[];unions=[]
    match,unionfn=criterion.matcher.forward,criterion._get_go_indices
    def fm(*a,**k):
        r=match(*a,**k);trace.append(d.native.cpu_tree(r['indices']));return r
    def fu(*a,**k):
        r=unionfn(*a,**k);unions.append(d.native.cpu_tree(r));return r
    criterion.matcher.forward,criterion._get_go_indices=fm,fu
    try:losses=criterion(outputs,targets,epoch=24)
    finally:criterion.matcher.forward,criterion._get_go_indices=match,unionfn
    assert len(trace)==5 and len(unions)==1
    return losses,unions[0],trace[0],criterion.record['ordinary']['denominator']

def metric(outputs,targets,union,denom,criterion,own):
    row=d.metric(outputs,targets,union,denom,criterion,own)
    idx=criterion._get_src_permutation_idx(union)
    target=torch.cat([t['boxes'][g] for t,(_,g) in zip(targets,union)])
    ref=outputs['ref_points'][idx].detach()
    label,wr,wl=bbox2distance(ref,d.box_cxcywh_to_xyxy(target),criterion.reg_max,outputs['reg_scale'],outputs['up'])
    z=outputs['pred_corners'][idx].reshape(-1,33)
    with torch.no_grad():
        unweighted=criterion.unimodal_distribution_focal_loss(z,label,wr,wl,torch.ones(len(label)),avg_factor=denom)
        criterion._clear_cache()
        fg=d.native.DEIMCriterion.loss_local(criterion,outputs,targets,union,denom)['loss_fgl']
        row['FGL_unweighted_CE']=float(unweighted)
        row['FGL_native_studentIoU_weighted']=float(fg)
        row['stage_geometry']={name:d.metric(o,targets,union,denom,criterion,own)['weighted_geometry']
            for name,o in zip(['aux0','aux1','final'],outputs['aux_outputs']+[outputs])}
    return row

def make_case(model,criterion,selector,named,binding,buffers,row,seed,folder):
    payload=d.payload_for(row);rng=payload['spec']['forward_DN_seed']
    d.restore_buffers(model,buffers);d.reset_rng(rng);selector.mode='native'
    residuals=[];handles=[]
    for head in model.decoder.dec_bbox_head:
        handles.append(head.register_forward_hook(lambda module,args,out:residuals.append(out)))
    try:outputs=model(payload['images'],payload['targets'])
    finally:
        for h in handles:h.remove()
    assert len(residuals)==3
    topk=selector.observed.clone()
    losses,union,own,denom=capture_losses(criterion,outputs,payload['targets'])
    pos,diag,idx=pos_distill(outputs,payload['targets'],union,criterion)
    for i in range(2):
        error=abs(diag[i]['full_native_ddf_reconstruction']-float(losses[f'loss_ddf_aux_{i}']))
        assert error<3e-6,(i,error)
    total=sum(losses.values());parameters=[p for _,p in named]
    raw=residuals[:2]
    # Positive ordinary DDF cannot propagate through the detached final teacher.
    grad_last=torch.autograd.grad(pos['original'],residuals[-1],allow_unused=True,retain_graph=True)[0]
    assert grad_last is None or float(grad_last.abs().max())==0
    def residual_grads(loss):
        result=torch.autograd.grad(loss,raw,allow_unused=True,retain_graph=True)
        return [torch.zeros_like(r) if g is None else g.detach() for g,r in zip(result,raw)]
    ordinary_offset=raw[0].shape[1]-300
    raw_idx=(idx[0],idx[1]+ordinary_offset)
    gs={name:residual_grads(value) for name,value in pos.items()}
    g=torch.cat([x[raw_idx] for x in gs['original']],-1)
    stage_c=[]
    for stage,o in enumerate(outputs['aux_outputs']+[outputs]):
        terms=d.geometric_terms(o['pred_boxes'],payload['targets'],union)
        G=(5*terms[0].sum()+2*terms[1].sum())/denom
        cc=residual_grads(G)
        stage_c.append(torch.cat([x[raw_idx] for x in cc],-1))
        if stage==0:assert float(cc[1].abs().max())==0
    c=torch.stack(stage_c,1)
    h=project_halfspaces(g,c)
    single=project_halfspaces(g,c.sum(1,keepdim=True))
    ratio=h.double().norm(dim=-1)/g.double().norm(dim=-1).clamp_min(1e-30)
    assert float(ratio.max())<=1+1e-6
    # Candidate/controls only change per-query adjoints on actual residual tensors.
    replacements={
        'original':g,'half_positive':g*.5,'off_positive':g*0,
        'single_total':single,'multi_layer':h,'matched_norm':g*ratio.to(g.dtype)[:,None]}
    for name in ['teacherbetter_gate','relocated_target','GT_two_bin']:
        replacements[name]=torch.cat([x[raw_idx] for x in gs[name]],-1)
    endpoints={};updates={};checks={}
    base=metric(outputs,payload['targets'],union,denom,criterion,own)
    for k,arm in enumerate(ARMS):
        correction=0
        parts=(replacements[arm]-g).split(132,dim=-1)
        for r,delta in zip(raw,parts):correction=correction+(r[raw_idx]*delta.detach()).sum()
        progress(seed=seed,batch=row['batch'],stage='full_parameter_gradient',arm=arm)
        grads=torch.autograd.grad(total+correction-correction.detach(),parameters,allow_unused=True,retain_graph=k<len(ARMS)-1)
        endpoints[arm],updates[arm]=d.virtual_endpoint(model,named,binding,grads)
        del grads,correction
    previous=json.loads((OLD/f'seed{seed}/batch{row["batch"]:02d}_capture.json').read_text(encoding='utf8'))
    for key in ['global_gradient_norm_native_FP32','global_clip_factor','parameter_update_L2_norm']:
        checks[key+'_original_vs_previous']=abs(updates['original'][key]-previous['updates']['original'][key])
        assert checks[key+'_original_vs_previous']<1e-10
    checks['base_topk_exact_previous']=topk.tolist()==previous['topk_indices'];assert checks['base_topk_exact_previous']
    # A zero functional step must reconstruct outputs before endpoint conclusions.
    selector.mode='fixed';selector.fixed=topk;d.reset_rng(rng)
    with torch.no_grad():zero=torch.func.functional_call(model,({n:p.detach() for n,p in model.named_parameters()},d.native.cpu_tree(buffers)),(payload['images'],payload['targets']),strict=True)
    zm=metric(zero,payload['targets'],union,denom,criterion,own)
    checks['zero_final_boxes_max_error']=float((zero['pred_boxes']-outputs['pred_boxes'].detach()).abs().max())
    checks['zero_geometry_abs_error']=abs(zm['weighted_geometry']-base['weighted_geometry'])
    checks['zero_unweighted_FGL_abs_error']=abs(zm['FGL_unweighted_CE']-base['FGL_unweighted_CE'])
    assert checks['zero_final_boxes_max_error']<3e-7
    assert checks['zero_geometry_abs_error']<1e-8 and checks['zero_unweighted_FGL_abs_error']<1e-8
    stats=dict(pairs=len(g),residual_coordinate_dimension=g.shape[-1],
        original_positive_gradient_norm=float(g.double().norm()),multi_positive_gradient_norm=float(h.double().norm()),
        multi_minus_single_norm=float((h-single).double().norm()),
        original_opposes_layer_counts=[int(((c[:,m].double()*g.double()).sum(-1)<-1e-12).sum()) for m in range(3)],
        multi_modified_queries=int(((h-g).abs().max(-1).values>1e-12).sum()),
        matched_norm_max_abs_error=float(((g.double()*ratio[:,None]).norm(dim=-1)-h.double().norm(dim=-1)).abs().max()))
    save(folder/f'batch{row["batch"]:02d}_capture.json',dict(seed=seed,batch=row['batch'],diagnostics=diag,
        residual_stats=stats,checks=checks,base=base,updates=updates,topk_indices=topk.tolist(),
        union=d.indices_json(union),native_loss_components={k:float(v.detach()) for k,v in losses.items()}))
    del outputs,losses,total,pos,residuals,raw,gs,g,c,h,single,replacements,zero
    gc.collect();d.restore_buffers(model,buffers)
    assert not any(p.grad is not None for p in model.parameters())
    return dict(payload=payload,rng=rng,topk=topk,union=union,own=own,denom=denom,endpoints=endpoints,base=base)

def evaluate(model,criterion,selector,buffers,source,target,seed,src,dst):
    result=dict(seed=seed,generator_batch=src,evaluation_batch=dst,before=target['base'],arms={})
    for arm in ARMS:
        result['arms'][arm]={}
        for mode in ['fixed','native']:
            d.reset_rng(target['rng']);selector.mode=mode;selector.fixed=target['topk']
            with torch.no_grad():
                outputs=torch.func.functional_call(model,(source['endpoints'][arm],d.native.cpu_tree(buffers)),(target['payload']['images'],target['payload']['targets']),strict=True)
                if mode=='fixed':union,own,denom=target['union'],target['own'],target['denom']
                else:
                    fresh,union,own,denom=capture_losses(criterion,outputs,target['payload']['targets'])
                    del fresh
                result['arms'][arm][mode]=metric(outputs,target['payload']['targets'],union,denom,criterion,own)
            del outputs
        progress(seed=seed,generator=src,evaluation=dst,stage='cross_batch',arm=arm,
                 fixed_G=result['arms'][arm]['fixed']['weighted_geometry'])
    return result

def main():
    resources=d.native.resource_limits();assert resources['available_RAM_GiB']>12
    tests=synthetic_tests()
    proto=json.loads((OLD/'01_冻结协议与资产.json').read_text(encoding='utf8'))
    for source in proto['sources']:assert sha(source['path'])==source['sha256']
    OUT.mkdir(exist_ok=False);d.OUT=OUT
    protocol=dict(utc=d.now(),spec_path=str(ROOT/'45_两项核查前登记.txt'),spec_sha256=sha(ROOT/'45_两项核查前登记.txt'),
        runner_sha256=sha(__file__),native_import_sha256=sha(d.__file__),old_protocol_sha256=sha(OLD/'01_冻结协议与资产.json'),
        audits_consumed=2,arms=ARMS,synthetic_tests=tests,resources=resources,
        model_forwards_expected=80,new_training_steps=0,optimizer_step_calls=0,remote_operations=0,
        holdout_read=False,scope='BASE24online2seeds, development00/01. Virtual all-parameter historical AdamW endpoints, not actual training or AP.',
        evaluation='fixed physical query + fixed original union primary; native query reselection + fresh native matching/union secondary',
        limitations='Two checkpoints, same8 augmented inputs, reused development only. No independence, significance, convergence or AP evidence. Projection guarantee only residual-coordinate direct first-order step; transported target clipping may destroy GT mean/shape.')
    save(OUT/'01_protocol.json',protocol)
    ledger_path=ROOT/'39_新目标与执行预算.json';ledger=json.loads(ledger_path.read_text(encoding='utf8'))
    assert ledger['auditsUsed']==0
    ledger.update(auditsUsed=2,status='two_preregistered_frozen_audits_running',
        candidates=['joint_residual_multilayer_positive_DDF','physical_teacher_shape_relocation'],
        audit_protocol_sha256=sha(OUT/'01_protocol.json'))
    ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf8')
    recipes=json.loads((d.OLD/'baseline_recipes.json').read_text(encoding='utf-8-sig'))
    start=time.monotonic();rows=[]
    for seed in [1,2]:
        folder=OUT/f'seed{seed}';folder.mkdir()
        cp=proto['checkpoints'][str(seed)];assert sha(cp['path'])==cp['sha256']
        raw=torch.load(cp['path'],map_location='cpu',weights_only=False)
        assert raw['last_epoch']==23 and recipes[str(seed)]['clip_max_norm']==.1
        model=d.native.build_model(recipes[str(seed)]);model.load_state_dict(raw['model'],strict=True);model.train()
        criterion=d.native.build_criterion(recipes[str(seed)]);criterion.load_state_dict(raw['criterion'],strict=True)
        named,binding=d.full_optimizer_binding(model,raw,recipes[str(seed)])
        assert len(named)==488 and sum(p.numel() for _,p in named)==10231995
        buffers={n:b.detach().clone() for n,b in model.named_buffers()};before_hash=d.native.model_hash(model)
        binding_hash=hashlib.sha256()
        for n,v in binding.items():
            for key,x in v['state'].items():binding_hash.update((n+key).encode());binding_hash.update(x.numpy().tobytes() if torch.is_tensor(x) else str(x).encode())
        del raw;gc.collect();selector=d.QuerySelection(model.decoder)
        cases={}
        for batch in [0,1]:
            assert time.monotonic()-start<900
            assert d.psutil.virtual_memory().available>8*2**30
            cases[batch]=make_case(model,criterion,selector,named,binding,buffers,proto['batches'][batch],seed,folder)
        for src,dst in [(0,1),(1,0)]:
            assert time.monotonic()-start<900
            result=evaluate(model,criterion,selector,buffers,cases[src],cases[dst],seed,src,dst)
            rows.append(result);save(folder/f'cross_{src:02d}_to_{dst:02d}.json',result)
        d.restore_buffers(model,buffers);assert d.native.model_hash(model)==before_hash
        assert sha(cp['path'])==cp['sha256']
        after_hash=hashlib.sha256()
        for n,v in binding.items():
            for key,x in v['state'].items():after_hash.update((n+key).encode());after_hash.update(x.numpy().tobytes() if torch.is_tensor(x) else str(x).encode())
        assert after_hash.hexdigest()==binding_hash.hexdigest()
        save(folder/'integrity.json',dict(original_checkpoint_SHA_unchanged=True,original_model_and_buffers_unchanged=True,
             original_optimizer_copy_unchanged=True,trainable_tensors=len(named),trainable_scalars=sum(p.numel() for _,p in named)))
        del model,criterion,named,binding,cases,buffers;gc.collect()
    save(OUT/'02_results.json',dict(protocol=protocol,rows=rows,seconds=time.monotonic()-start,training_steps=0))
    ledger=json.loads(ledger_path.read_text(encoding='utf8'));ledger['status']='two_audits_complete_awaiting_expert_review'
    ledger['results_sha256']=sha(OUT/'02_results.json');ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf8')
    progress(stage='complete',seconds=time.monotonic()-start)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/('failure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
