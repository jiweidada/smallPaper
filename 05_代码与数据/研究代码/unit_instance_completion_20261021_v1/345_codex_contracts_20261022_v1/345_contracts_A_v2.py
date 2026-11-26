"""Execute A1..A12 in order. Failure is final for this run; B stays unexecuted."""
import importlib,copy,time,math,hashlib,traceback
C=importlib.import_module('345_context');O=importlib.import_module('345_numpy_oracle')
import torch,numpy as np
T=importlib.import_module('train344');G=T.geom;core=T.core311;gcal=importlib.import_module('gcal');UL=importlib.import_module('uic_losses')
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
COUNT=dict(subject_models=0,implicit_common_init_references=0,old_reference_models=0,cache_loads=0,head_attempts=0,head_completions=0,
    loss_calls=0,autograd_calls=0,explicit_plan_calls=0,NumPy_plan_calls=0,independent_geometry_calls=0,negative_controls=0,
    global_models=0,global_forwards=0,global_image_decodes=0,PNG_writes=0,optimizer_steps=0)
RESULTS=[];REAL={};SETS=[]

def model(arm,seed=1,active=False):
    m=T.make_common_init(arm,seed).eval();COUNT['subject_models']+=1;COUNT['implicit_common_init_references']+=1
    if active:
        # Nonconstant test-fixture parameters only; no optimization or source change.
        gen=torch.Generator().manual_seed(345100+seed)
        with torch.no_grad():
            for p in (m.classifier.weight,m.box_head[-1].weight,m.association.query.weight,m.association.null[-1].weight):
                p.copy_(torch.randn(p.shape,generator=gen)*.03)
            if arm=='S':
                m.count_projection.weight.copy_(torch.randn(m.count_projection.weight.shape,generator=gen)*.03)
                for block in m.blocks:block.p_attention.out_proj.weight.copy_(torch.randn(block.p_attention.out_proj.weight.shape,generator=gen)*.03)
    return m

def forward(m,b):
    COUNT['head_attempts']+=1
    out=m.forward_features(m.trunk(b['rgb']),b['P_boxes'],b['P_cls'],b['P_scores'],b['P_valid'])
    COUNT['head_completions']+=1;return out

def losses(m,b):
    COUNT['loss_calls']+=1;COUNT['head_attempts']+=1
    result=T.compute_losses_340(m.arm,m,b);COUNT['head_completions']+=1;return result

def grad(loss,params,retain_graph=True):
    COUNT['autograd_calls']+=1
    return torch.autograd.grad(loss,tuple(params),allow_unused=True,retain_graph=retain_graph)

def array(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def diff(a,b):
    aa,bb=array(a),array(b);assert aa.shape==bb.shape
    if aa.dtype==np.bool_ or np.issubdtype(aa.dtype,np.integer):return float(np.any(aa!=bb))
    return float(np.max(np.abs(aa.astype(np.float64)-bb.astype(np.float64)),initial=0))
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.detach().contiguous().numpy().tobytes()==b.detach().contiguous().numpy().tobytes()
def strict(a,b):
    d=diff(a,b);assert bits(a,b),('bitwise mismatch',d);return d
def close(a,b,tol):
    d=diff(a,b);assert d<=tol,('numeric mismatch',d,tol);return d
def finite(d):
    for k,v in d.items():
        if torch.is_tensor(v):assert torch.isfinite(v).all(),k
def cap(*args,which=core):COUNT['explicit_plan_calls']+=1;return which.independent_cap(*args)

def synthetic():
    gt=torch.tensor([[2.,40.,6.,46.],[10.,40.,16.,46.],[20.,40.,26.,46.],[30.,40.,36.,46.],[40.,40.,46.,46.],[50.,50.,60.,60.]])
    w=torch.tensor([0.,0.,64.,64.]);gw=torch.ones(6,dtype=torch.bool)
    e={'image_id':-345,'seed':0,'window_xyxy_640':w,'crop128':torch.rand(3,128,128,generator=torch.Generator().manual_seed(345)),
       'fullp_boxes':torch.tensor([[3.,3.,6.,6.],[20.,2.,28.,8.],[32.,12.,42.,24.]]),
       'fullp_labels':torch.tensor([7,8,9]),'fullp_scores':torch.tensor([.8,.6,.3]),'fullp_nonpositive_mask':torch.zeros(3,dtype=torch.bool),
       'm0_cand_owner':torch.full((3,),-1,dtype=torch.long),'gt_boxes_640':gt,'gt_labels':torch.arange(6),
       'G_W':gw,'U_W':gw.clone(),'K_g':G.kg_window(gt,gw,w),'n_G_W':6,'n_U_W':6}
    return T.collate([e])

def captures(m,b):
    taps={};handles=[]
    for i,block in enumerate(m.blocks):
        handles.append(block.register_forward_hook(lambda mod,args,out,k=f'hidden{i}':taps.__setitem__(k,out.detach().clone())))
    for name in ('query','key','null'):
        obj=getattr(m.association,name)
        handles.append(obj.register_forward_hook(lambda mod,args,out,k=name:taps.__setitem__(k,out.detach().clone())))
    try:out=forward(m,b)
    finally:
        for h in handles:h.remove()
    return out,taps

def A1():
    rows=[]
    for arm in 'NSGR':
        for name,b in SETS:
            m=model(arm,2 if name.startswith('seed2') else 1,True)
            ref=gcal.GeometryCalibratedHead() if arm=='G' else core.CompletionHead(arm);COUNT['old_reference_models']+=1
            ref.load_state_dict(m.state_dict(),strict=True);ref.eval()
            with torch.no_grad():a,ha=captures(m,b);r,hr=captures(ref,b)
            fields={k:strict(a[k],r[k]) for k in ('pred_boxes','pred_logits','query_states','mass','density_field')}
            for k in a['plan']:fields['plan/'+k]=strict(a['plan'][k],r['plan'][k])
            if arm=='G':
                for k in a['generation_plan']:fields['generation/'+k]=strict(a['generation_plan'][k],r['generation_plan'][k])
            for i in range(2):fields[f'hidden{i}']=strict(ha[f'hidden{i}'],hr[f'hidden{i}'])
            corrupted=r['query_states'].clone();corrupted.view(-1)[0]+=1.
            assert not bits(a['query_states'],corrupted);COUNT['negative_controls']+=1
            rows.append({'arm':arm,'input':name,'max_abs_diff':max(fields.values()),'all_fields_bit_equal':True,'fields':fields,'negative_corruption_detected':True})
    return {'method':'Current and original frozen heads load exactly the same nonconstant fixture state; direct tensor-byte compare including every plan field and both hidden layers.','rows':rows}

def verify_plan(m,c,k,cl,v,out,tol):
    ref=O.cap(array(m),array(c),array(k),array(cl),array(v));COUNT['NumPy_plan_calls']+=1
    err={name:close(out[name],ref[name],tol) for name in ref}
    close(out['T'].sum(1)+out['z'],v.to(m.dtype),tol)
    assert (out['explained']<=m+tol).all() and (out['T']>=-tol).all() and (out['z']>=-tol).all()
    close(out['z'],out['p0']+(out['p']-out['T']).sum(1),tol)
    assert (out['T'][m.gather(1,torch.where(v,cl,0)[...,None].expand(cl.shape[0],cl.shape[1],m.shape[-1])).transpose(1,2)==0]==0).all()
    bad=out['T'].clone()
    if bad.numel():
        bad.view(-1)[0]+=1.;assert diff(bad,ref['T'])>.1;COUNT['negative_controls']+=1
    return max(err.values())

def A2():
    rows=[]
    for arm in 'NSGR':
        mod=gcal.base if arm=='G' else core
        gen=torch.Generator().manual_seed(3452)
        m=torch.rand(2,10,5,generator=gen,dtype=torch.float64);m[:,0,0]=0;m[1,1]=0
        c=torch.randn(2,5,7,generator=gen,dtype=torch.float64);k=torch.randn(2,7,generator=gen,dtype=torch.float64)
        cl=torch.tensor([[0,0,1,2,3,4,-99],[1,1,2,2,3,-99,-99]]);v=cl>=0
        out=cap(m,c,k,cl,v,which=mod);mx=verify_plan(m,c,k,cl,v,out,2e-12)
        empty=cap(m,c[...,:0],k[:,:0],cl[:,:0],v[:,:0],which=mod);strict(empty['residual'],m)
        z=cap(torch.zeros_like(m),c,k,cl,v,which=mod);assert (z['T']==0).all();strict(z['z'],v.double())
        # Independently split the second same-cost row into two half-mass rows.
        mm=torch.tensor([[[.7,1.2]]],dtype=torch.float64);cc=torch.tensor([[[.3],[-.2]]],dtype=torch.float64)
        kk=torch.tensor([[.1]],dtype=torch.float64);cls=torch.zeros(1,1,dtype=torch.long);valid=torch.ones(1,1,dtype=torch.bool)
        base=cap(mm,cc,kk,cls,valid,which=mod)
        spl=cap(torch.tensor([[[.7,.6,.6]]],dtype=torch.float64),torch.tensor([[[.3],[-.2],[-.2]]],dtype=torch.float64),kk,cls,valid,which=mod)
        for field in ('T','p'):
            close(base[field][:,0],spl[field][:,0],2e-12);close(base[field][:,1],spl[field][:,1:].sum(1),2e-12)
        close(base['z'],spl['z'],2e-12);close(base['residual'][:,:,1],spl['residual'][:,:,1:].sum(2),2e-12)
        rows.append({'arm':arm,'input':'synthetic','NumPy_max_abs_diff':mx,'empty_zero_mass_split_and_conservation':True})
        for name,b in SETS[1:]:
            mdl=model(arm,2 if name.startswith('seed2') else 1,True)
            with torch.no_grad():o,taps=captures(mdl,b)
            _,sb,sc,_=core.candidate_inputs(b['P_boxes'],b['P_cls'],b['P_scores'],b['P_valid'])
            geom=G.geometry_cost(sb);ind=O.geometry(array(sb));COUNT['independent_geometry_calls']+=1
            ge=close(geom,ind,1e-5)
            cost=-torch.einsum('bid,bjd->bij',taps['key'],taps['query'])/math.sqrt(128)+geom
            er=verify_plan(o['mass'].detach(),cost,taps['null'].squeeze(-1),sc,b['P_valid'],o['plan'],1e-5)
            rows.append({'arm':arm,'input':name,'NumPy_max_abs_diff':er,'independent_geometry_max_abs_diff':ge})
    return {'method':'Scalar NumPy mass-reference probabilities, classwise row cap, explicit excess-to-null accumulation; separate x=column,y=row grid enumeration. Internal unused scale at u=0 is not a semantic output compared to the oracle.','rows':rows}

def A3():
    rows=[]
    for arm in 'NSGR':
        mod=gcal.base if arm=='G' else core
        c=torch.tensor([[[60.]]],requires_grad=True);k=torch.zeros(1,1,requires_grad=True)
        out=cap(torch.ones(1,1,1),c,k,torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),which=mod)
        finite(out);gc,gk=grad(out['T'].sum(),(c,k));assert torch.isfinite(gc).all() and torch.isfinite(gk).all()
        u=float(out['u'].item());assert 1e-28<u<1e-24
        expected=math.exp(-60)/(1+math.exp(-60));assert abs(float(gc)+expected*(1-expected))<1e-31
        # Deliberate legacy algebra: 0 * derivative(m/u) is non-finite at tiny u.
        tiny=torch.tensor([1e-26],requires_grad=True);wrong=torch.minimum(torch.ones(1),torch.ones(1)/tiny)
        wrong_grad=grad(wrong.sum(),(tiny,))[0];assert not torch.isfinite(wrong_grad).all();COUNT['negative_controls']+=1
        rows.append({'arm':arm,'u':u,'grad_cost':float(gc),'grad_null':float(gk),'analytic_grad_abs_diff':abs(float(gc)+expected*(1-expected)),'legacy_negative_detected':True})
    return {'method':'Analytic Bernoulli derivative at C=60 and deliberate legacy m/u control; real FP32 inputs additionally checked finite in A2/A6.','rows':rows}

def A4():
    rows=[]
    for arm in 'NSGR':
        mdl=model(arm,active=True)
        for name,b in SETS:
            for mode in ('J0','all_padding'):
                empty=dict(b)
                if mode=='J0':
                    for key in ('P_boxes','P_cls','P_scores','P_valid'):empty[key]=b[key][:,:0]
                else:empty['P_valid']=torch.zeros_like(b['P_valid'])
                # Loss uses the same empty P_W subset, so controlled entries also remove fullP.
                ee=[]
                for original in b['entries']:
                    e=dict(original)
                    for key in ('fullp_boxes','fullp_labels','fullp_scores','fullp_nonpositive_mask','m0_cand_owner'):e[key]=original[key][:0]
                    ee.append(e)
                empty['entries']=ee
                with torch.no_grad():loss,out=losses(mdl,empty)
                finite(loss);finite(out);rows.append({'arm':arm,'input':name,'mode':mode,'finite_losses':{k:float(x) for k,x in loss.items()}})
        # Check overflow with all six GT in denominator, using independent SciPy assignment size.
        with torch.no_grad():ll,oo=losses(mdl,SETS[0][1])
        e=SETS[0][1]['entries'][0];ub=G.win_norm_cxcywh(e['gt_boxes_640'][e['U_W']],e['window_xyxy_640'])
        native=UL.compute_losses({'pred_boxes':oo['pred_boxes'][0],'pred_logits':oo['pred_logits'][0]},ub,e['gt_labels'][e['U_W']],denominator=6.,k=4)
        assert native['num_matched']==4 and native['overflow']==2 and native['denominator']==6.
        det=native['loss_mal']+5*native['loss_bbox']+2*native['loss_giou'];close(ll['loss_det'],det,1e-6)
        wrong=det*1.5;assert diff(det,wrong)>1e-4;COUNT['negative_controls']+=1
        rows.append({'arm':arm,'overflow_GT':6,'matched':4,'unmatched':2,'denominator':6,'independent_min_4_N':min(4,6),'loss_det_abs_diff':diff(ll['loss_det'],det),'wrong_denom4_detected':True})
    return {'method':'Controlled empty P targets and real batches; six-GT synthetic overflow checked against direct native matcher and denominator6, with wrong-denominator4 control.','rows':rows}

def variants(b):
    keys=('P_boxes','P_cls','P_scores','P_valid');valid=b['P_valid'];out=[]
    translated=b['P_boxes'].clone()
    # All coordinate changes use existing geom342 transforms. A shifted 64px
    # reference window implements a width/height-preserving local translation.
    zero=torch.tensor([0.,0.,64.,64.]);shift=torch.tensor([1.,2.,65.,66.])
    translated[valid]=G.win_norm_xyxy(G.win_norm_to_window_pixels(b['P_boxes'][valid],zero),shift)
    a=dict(b);a['P_boxes']=translated;out.append(('boxes_valid_translation',a))
    a=dict(b);a['P_cls']=torch.where(valid,(b['P_cls']+3)%10,b['P_cls']);out.append(('classes_valid_only',a))
    a=dict(b);a['P_scores']=torch.where(valid,1-b['P_scores'],b['P_scores']);out.append(('scores_valid_only',a))
    perm=dict(b)
    for key in keys:perm[key]=b[key].clone()
    for row in range(len(valid)):
        ix=valid[row].nonzero().flatten()
        for key in keys:perm[key][row,ix]=b[key][row,ix.flip(0)]
    out.append(('joint_valid_permutation',perm))
    for kind in ('duplicate','delete_alternate'):
        indices=[]
        for row in range(len(valid)):
            ix=valid[row].nonzero().flatten()
            indices.append(torch.cat((ix,ix)) if kind=='duplicate' else ix[::2])
        maxn=max((len(ix) for ix in indices),default=0);a=dict(b)
        for key in keys:
            shape=(len(valid),maxn)+tuple(b[key].shape[2:]);a[key]=torch.zeros(shape,dtype=b[key].dtype)
            for row,ix in enumerate(indices):a[key][row,:len(ix)]=b[key][row,ix]
        out.append((kind+'_joint_fields',a))
    a=dict(b)
    for key in keys:a[key]=b[key][:,:0]
    out.append(('empty_joint_fields',a))
    a=dict(b);a['P_valid']=torch.zeros_like(valid);out.append(('all_padding_joint_supply',a))
    return out

def A5():
    rows=[];reverse=[]
    for name,b in SETS:
        m=model('N',active=True)
        with torch.no_grad():base=forward(m,b)
        for kind,v in variants(b):
            # Independently validate fixture slots before subject execution.
            tokens,safe,classes,counts=core.candidate_inputs(v['P_boxes'],v['P_cls'],v['P_scores'],v['P_valid'])
            assert torch.isfinite(tokens).all()
            if kind in ('boxes_valid_translation','classes_valid_only','scores_valid_only','joint_valid_permutation'):
                for key in ('P_boxes','P_cls','P_scores','P_valid'):strict(v[key][~b['P_valid']],b[key][~b['P_valid']])
            with torch.no_grad():out=forward(m,v)
            for field in ('pred_boxes','pred_logits','query_states'):strict(base[field],out[field])
            aux_changed=base['plan']['z'].shape!=out['plan']['z'].shape or not bits(base['plan']['z'],out['plan']['z'])
            rows.append({'input':name,'perturbation':kind,'all_output_bytes_identical':True,'max_abs_diff':0.,
                         'valid_candidates_before':int(b['P_valid'].sum()),'valid_candidates_after':int(v['P_valid'].sum()),'aux_z_changed':aux_changed})
        # Deliberately reproduce the rejected old fixture. Add one zero pad to
        # the synthetic all-valid batch so the negative control is applicable.
        pad=dict(b)
        for key in ('P_boxes','P_cls','P_scores','P_valid'):
            shape=(b[key].shape[0],1)+tuple(b[key].shape[2:])
            pad[key]=torch.cat((b[key],torch.zeros(shape,dtype=b[key].dtype)),dim=1)
        wrong=dict(pad);wrong['P_boxes']=pad['P_boxes'].flip(1)
        rb=wrong['P_boxes'][wrong['P_valid']]
        invalid_count=int((~(rb[:,2:]>rb[:,:2]).all(-1)).sum());assert invalid_count>0
        try:
            with torch.no_grad():forward(m,wrong)
        except AssertionError:
            COUNT['negative_controls']+=1;reverse.append({'input':name,'expected_failure':'boxes-only flip with unchanged validity',
                'rejected':True,'active_nonpositive_boxes':invalid_count,'extra_padding_columns_for_test':1})
        else:raise AssertionError('malformed boxes-only flip was accepted')
        if name!='synthetic':
            # Exact old real-batch construction, with no added test padding.
            legacy=dict(b);legacy['P_boxes']=b['P_boxes'].flip(1)
            rb=legacy['P_boxes'][legacy['P_valid']];legacy_invalid=int((~(rb[:,2:]>rb[:,:2]).all(-1)).sum())
            if legacy_invalid:
                try:
                    with torch.no_grad():forward(m,legacy)
                except AssertionError:COUNT['negative_controls']+=1;reverse.append({'input':name,'expected_failure':'exact old real fixture','rejected':True,'active_nonpositive_boxes':legacy_invalid,'extra_padding_columns_for_test':0})
                else:raise AssertionError('old invalid real fixture accepted')
        bad=base['query_states']+1.;assert not bits(base['query_states'],bad);COUNT['negative_controls']+=1
    return {'method':'345b: only valid slots are translated/changed; all four fields move together for valid-slot permutation and count changes. Nonzero N readout, exact output bytes, malformed boxes-only flip is an expected rejected control.',
            'applicability':{'N':'tested','S':'not applicable: intended P generation path','G':'not applicable: intended P generation path','R':'not applicable: intended P generation path'},'rows':rows,'negative_tests':reverse}

def A6():
    rows=[]
    for arm in 'NSGR':
        for name,b in SETS:
            m=model(arm,active=True);ll,out=losses(m,b)
            one={}
            for loss_name in ('loss_det','loss_assoc','loss_mass'):
                gg=grad(ll[loss_name],m.mass_head.parameters());mx=max((float(g.abs().max()) for g in gg if g is not None),default=0.)
                assert all(torch.isfinite(g).all() for g in gg if g is not None)
                if loss_name!='loss_mass':assert mx==0.
                else:assert mx>0.
                one[loss_name]={'gradient_max_abs':mx,'all_None':all(g is None for g in gg)}
            positive=grad(out['mass'].sum(),m.mass_head.parameters());assert max(float(g.abs().max()) for g in positive if g is not None)>0;COUNT['negative_controls']+=1
            assert not out['plan']['mass'].requires_grad and not out['density_field'].requires_grad if arm in ('N','S') else not out['plan']['mass'].requires_grad
            rows.append({'arm':arm,'input':name,'mass_head_gradients':one,'direct_mass_positive_control':True})
    return {'method':'Separate autograd.grad per true L_det/L_assoc/L_mass using nonconstant heads, plus direct-mass positive control; no backward/optimizer accumulation.','rows':rows}

def A7():
    rows=[]
    for name,b in SETS:
        m=model('S',active=True);calls=[]
        h=m.count_projection.register_forward_pre_hook(lambda mod,args:calls.append(args[0].detach().clone()))
        try:
            with torch.no_grad():a=forward(m,b)
            perm=dict(b)
            for key in ('P_boxes','P_cls','P_scores','P_valid'):perm[key]=b[key].flip(1)
            with torch.no_grad():r=forward(m,perm)
            errs={key:close(a[key],r[key],2e-5) for key in ('pred_boxes','pred_logits','query_states')}
            expected=np.zeros((b['rgb'].shape[0],10))
            for i in range(len(expected)):
                for j in range(b['P_valid'].shape[1]):
                    if b['P_valid'][i,j]:expected[i,int(b['P_cls'][i,j])]+=1
            close(a['counts'],expected,0);close(calls[0],np.log1p(expected),1e-6);strict(calls[0],calls[1])
            changed=variants(b)[0][1];changed['P_scores']=torch.where(b['P_valid'],1-b['P_scores'],b['P_scores'])
            with torch.no_grad():forward(m,changed)
            strict(calls[0],calls[-1])
            invalid=dict(b);invalid['P_valid']=torch.zeros_like(b['P_valid']);entered=[]
            hh=[bl.p_attention.register_forward_pre_hook(lambda mod,args:entered.append(True)) for bl in m.blocks]
            with torch.no_grad():z=forward(m,invalid)
            for hook in hh:hook.remove()
            assert not entered;finite(z)
            assert diff(calls[0],calls[-1])>0;COUNT['negative_controls']+=1
        finally:h.remove()
        rows.append({'input':name,'permutation_max_differences':errs,'count_scalar_NumPy_exact':True,'count_ignores_coordinates_and_score':True,'empty_row_cross_attention_calls':len(entered),'count_corruption_detected':True})
    return {'method':'Nonzero S token/count paths, NumPy scalar class counts and count-projection input hooks; active-row hook rejects any all-masked attention.','applicability':{'N':'not applicable: no P cross-attention/count projection','S':'tested','G':'not applicable: no P cross-attention/count projection','R':'not applicable: no P cross-attention/count projection'},'rows':rows}

def A8():
    rows=[]
    for name,b in SETS:
        seed=2 if name.startswith('seed2') else 1;g=model('G',seed);r=model('R',seed)
        with torch.no_grad():go,gh=captures(g,b);ro,rh=captures(r,b)
        fields={k:strict(go[k],ro[k]) for k in ('pred_boxes','pred_logits','query_states','density_field')}
        for i in range(2):fields[f'decoder_hidden_{i}']=strict(gh[f'hidden{i}'],rh[f'hidden{i}'])
        strict(go['generation_plan']['residual'],ro['plan']['residual']);assert (g.class_null==0).all()
        changed=gh['hidden0'].clone();changed.view(-1)[0]+=1.;assert not bits(changed,rh['hidden0']);COUNT['negative_controls']+=1
        rows.append({'input':name,'max_abs_diff':max(fields.values()),'all_hidden_and_outputs_bit_equal':True,'fields':fields,'hidden_corruption_detected':True})
    return {'method':'Hooks on both decoder block outputs, exact tensor bytes under actual common initialization; hidden corruption control independent from zero output heads.','applicability':{'N':'not applicable: G/R common-init comparison','S':'not applicable: G/R common-init comparison','G':'tested','R':'tested'},'rows':rows}

def A9():
    rows=[]
    for name,b in SETS:
        m=model('G',active=True)
        with torch.no_grad():before=forward(m,b)
        with torch.no_grad():m.association.query.weight.add_(.15);m.association.null[-1].bias.add_(.7)
        with torch.no_grad():after=forward(m,b)
        for k in ('pred_boxes','pred_logits','query_states','density_field'):strict(before[k],after[k])
        for k in before['generation_plan']:strict(before['generation_plan'][k],after['generation_plan'][k])
        auxdiff=diff(before['plan']['p'],after['plan']['p']);assert auxdiff>1e-8;COUNT['negative_controls']+=1
        ll,_=losses(m,b);routes={}
        for k in ('loss_det','loss_mass','loss_assoc'):
            gr=grad(ll[k],(m.class_null,))[0]
            if k=='loss_det':assert gr is not None and torch.isfinite(gr).all() and gr.abs().max()>0
            else:assert gr is None
            routes[k]=None if gr is None else gr.detach().tolist()
        rows.append({'input':name,'generation_and_predictions_bit_equal':True,'aux_max_abs_change':auxdiff,'class_null_gradient_routes':routes})
    return {'method':'Controlled changes only to auxiliary query/null weights; full generation plan/output byte comparisons and three separate true-loss autograd calls.','applicability':{'N':'not applicable: G-only branch isolation','S':'not applicable: G-only branch isolation','G':'tested','R':'not applicable: aux is intentionally generation input'},'rows':rows}

def A10():
    rows=[]
    for name,b in SETS:
        r=model('R',active=True);n=model('N');n.load_state_dict(r.state_dict(),strict=True)
        with torch.no_grad():ro=forward(r,b);no=forward(n,b)
        strict(ro['density_field'],ro['plan']['residual']);assert diff(ro['density_field'],ro['mass'])>1e-8
        def replace(mod,args,out):
            a,p=out;q=dict(p);q['residual']=q['mass'];return a,q
        hook=r.association.register_forward_hook(replace)
        try:
            with torch.no_grad():forced=forward(r,b)
        finally:hook.remove()
        for k in ('pred_boxes','pred_logits','query_states','density_field'):strict(forced[k],no[k])
        qdiff=diff(ro['query_states'],no['query_states']);assert qdiff>1e-7;COUNT['negative_controls']+=1
        rows.append({'input':name,'R_field_is_residual_bytes_equal':True,'R_with_mass_equals_N_bytes':True,'R_residual_vs_N_hidden_max_abs_diff':qdiff})
    return {'method':'Association-output hook changes only the supplied residual tensor in a temporary test forward; same weights/RGB/candidates, nonzero readout; N is independently constructed with strict copied state.','applicability':{'N':'reference tested','S':'not applicable: extra token path','G':'not applicable: separate geometry generation plan','R':'tested'},'rows':rows}

def PNG(e,out,arm):
    from PIL import Image,ImageDraw
    crop=Image.fromarray((e['crop128'].permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')).resize((512,512),Image.NEAREST)
    a=Image.new('RGB',(1040,584),'white');a.paste(crop,(8,56));d=ImageDraw.Draw(a)
    keep=G.intersecting_fullp_candidates(e['fullp_boxes'],e['window_xyxy_640'],e['fullp_nonpositive_mask'])
    boxes=G.win_norm_to_window_pixels(G.win_norm_xyxy(e['fullp_boxes'][keep],e['window_xyxy_640']),e['window_xyxy_640'],(512,512)).numpy()
    for edge in boxes:d.rectangle(tuple(np.clip(edge,0,511)+[8,56,8,56]),outline='cyan',width=1)
    xy=G.grid_xy().numpy()*512
    for x,y in xy:d.ellipse((x+7,y+55,x+9,y+57),fill='white')
    field=out['density_field'][0].sum(0).detach().numpy().reshape(16,16);scale=max(float(field.max()),1e-12)
    heat=np.zeros((16,16,3),dtype=np.uint8);heat[:,:,0]=(field/scale*255).round().astype('uint8')
    a.paste(Image.fromarray(heat).resize((512,512),Image.NEAREST),(524,56))
    for (x,y),v in zip(xy,field.flatten()):
        if v>0:d.ellipse((x+6,y+54,x+10,y+58),outline='red')
    d.text((8,8),f'345 seed={e["seed"]} image={e["image_id"]} arm={arm}; initial common weights, generation residual; not trained',fill='black')
    d.text((8,28),'Real crop / unclipped P_W (display clipped) / frozen grid + residual>0 red markers',fill='black')
    d.text((524,28),'Generation residual sum over classes (red heat, scaled within this PNG)',fill='black')
    path=C.HERE/f'345_A11_seed{e["seed"]}_{arm}_residual.png'
    with open(path,'xb') as f:a.save(f,format='PNG')
    COUNT['PNG_writes']+=1;return str(path)

def A11():
    rows=[];paths=[]
    for name,b in SETS:
        for arm in 'NSGR':
            m=model(arm)
            with torch.no_grad():out=forward(m,b)
            for i,e in enumerate(b['entries']):
                real=b['P_boxes'][i][b['P_valid'][i]]
                if not len(real):continue
                correct=G.geometry_cost(real);ref=O.geometry(array(real[None]));COUNT['independent_geometry_calls']+=1;close(correct,ref,1e-5)
                for kind,wrong in [('cxcywh',G.xyxy_to_cxcywh(real)),('window_pixels',G.win_norm_to_window_pixels(real,e['window_xyxy_640']))]:
                    try:cost=G.geometry_cost(wrong)
                    except AssertionError:rejected=True;delta=None
                    else:delta=diff(cost,correct);rejected=delta>1e-4
                    assert rejected,(kind,name,arm,e['image_id']);COUNT['negative_controls']+=1
                    rows.append({'arm':arm,'input':name,'image_id':e['image_id'],'wrong_format':kind,'rejected_or_distinct':True,'max_abs_diff_if_accepted':delta})
    for seed in (1,2):
        e=REAL[seed][0];b=T.collate([e])
        for arm in ('R','G'):
            m=model(arm,seed)
            with torch.no_grad():out=forward(m,b)
            paths.append(PNG(e,out,arm))
    return {'method':'Shared geometry under test versus independent column/row enumeration; deliberate cxcywh and pixel inputs on all real/synthetic valid candidates; four PNGs as explicitly requested in345.','rows':rows,'png_paths':paths}

def A12():
    S=importlib.import_module('stage1_cache_v3');old=C.read(C.SPEC/'327_训练实施输入索引.json');cp=next(x for x in old['checkpoints'] if x['seed']==1)
    assert C.sha(cp['path'])==cp['expected_SHA256_from_288'];m=S.load_real_model(cp['path']);COUNT['global_models']+=1
    before={k:v.detach().clone() for k,v in m.state_dict().items()};assert len(before)==794
    from PIL import Image
    CX=importlib.import_module('344_context');e=REAL[1][0];src=CX.TRAIN/'images'/e['file_name'];assert C.sha(src)==e['source_image_sha256']
    with Image.open(src) as f:pil=f.convert('RGB')
    COUNT['global_image_decodes']+=1
    with torch.no_grad():boxes,logits=S.global_forward_640(m,pil)
    COUNT['global_forwards']+=1;assert boxes.shape==(300,4) and logits.shape==(300,10)
    assert all(bits(v,m.state_dict()[k]) for k,v in before.items())
    assert all(not p.requires_grad and p.grad is None and p.device.type=='cpu' for p in m.parameters())
    assert not boxes.requires_grad and not logits.requires_grad
    corrupt=next(iter(before.values())).clone();corrupt.view(-1)[0]+=1.;assert not bits(corrupt,next(iter(before.values())));COUNT['negative_controls']+=1
    return {'method':'One actual raw global model and one real-image bounded forward; independent cloned794 state tensors and byte comparison, CPU/frozen/grad None checks, corrupted-state control.',
        'seed':1,'batch':1,'state_tensor_count':794,'state_bit_equal':True,'all_grad_None':True,'all_requires_grad_false':True,
        'checkpoint':cp['path'],'checkpoint_sha256':C.sha(cp['path']),
        'applicability':{'N':'shared frozen detector contract','S':'shared frozen detector contract','G':'shared frozen detector contract','R':'shared frozen detector contract'},
        'seed2_scope':'not constructed;345 expressly requests one global construction/one bounded batch, both seed head tests use their own SHA-bound caches.'}

def main():
    start=time.perf_counter();current=0
    auth=C.SPEC/'345b_补充裁定_A5测试构造修正.txt'
    C.write('345_A5修正预定口径_v2.json',{'authorization_path':str(auth),'authorization_sha256':C.sha(auth),
        'legal_changes':'valid-only geometric translation/class/score; all four fields jointly permuted, duplicated, deleted or emptied; padding values stay padding.',
        'negative_control':'boxes-only flip must be rejected; report exact old real-batch count separately from extra-pad synthetic control.',
        'A1_A4':'inherit verified originals after source SHA checks, no rerun; corrected A5 plus A7 fixture only.',
        'thresholds':'unchanged from345_A_预定验收口径.json','已执行':{'protocol_writes':1,'model_forwards':0,'optimizer_steps':0},
        '未执行':['A5 through A12','FLOPs','B','真实训练'],'未核':['resumed outcomes'],'source_sha256':C.sources()})
    try:
        oldfailure=C.read(C.HERE/'345_result_A实现失败_停止.json')
        for path,h in oldfailure['source_sha256'].items():assert C.sha(path)==h,path
        RESULTS.extend(C.read(C.HERE/f'345_result_A{i:02d}.json') for i in range(1,5))
        ids=importlib.import_module('344_context').development_ids()[:32]
        SETS.append(('synthetic',synthetic()))
        for seed in (1,2):
            entries=[T.load_entry(seed,i) for i in ids];COUNT['cache_loads']+=32;REAL[seed]=entries
            for j in range(2):SETS.append((f'seed{seed}_real16_batch{j}',T.collate(entries[j*16:j*16+16])))
        for current,test in enumerate((A5,A6,A7,A8,A9,A10,A11,A12),5):
            before=dict(COUNT);t0=time.perf_counter();details=test()
            row={'contract':current,'status':'已验证','independent_or_negative_method':details.pop('method'),
                'details':details,'call_counts':{k:COUNT[k]-before[k] for k in COUNT},'wall_seconds':time.perf_counter()-t0,'source_sha256':C.sources(),
                'authorization_sha256':C.sha(auth)}
            RESULTS.append(row)
            C.write(f'345_result_A{current:02d}_v2.json',{**row,'已执行':row['call_counts'],'未执行':['B','真实训练','GPU'],'未核':['主控独立复核']})
            print(f'A{current:02d} v2 PASS head={COUNT["head_completions"]} grad={COUNT["autograd_calls"]} time={time.perf_counter()-start:.1f}s',flush=True)
        C.write('345_result_A合同总表_v2.json',{'status':'12项合同已验证；FLOPs和B未执行','contracts':RESULTS,
            'scope':'A1..A4 original pass; A5..A12 resumed under345b','已执行_本次':COUNT,'已执行':COUNT,
            'wall_seconds':time.perf_counter()-start,'未执行':['FLOPs补足','B正式入口','真实训练','GPU','SSH','队列'],
            '未核':['4PNG本轮目视','主控独立复核'],'source_sha256':C.sources(),'authorization_sha256':C.sha(auth)})
    except Exception as ex:
        failure={'contract':current,'status':'实现失败','exception':repr(ex),'traceback':traceback.format_exc()}
        C.write('345_result_A实现失败_停止_v2.json',{'status':'实现失败；按345b停下，B未执行','passed_contracts':RESULTS,'failed':failure,
            'remaining_contracts':[{'contract':i,'status':'未验证','reason':'未执行；前项失败后停止'} for i in range(max(current+1,5),13)],
            '已执行':COUNT,'wall_seconds':time.perf_counter()-start,'未执行':['B正式入口实现与验收','剩余A合同','FLOPs补足','真实训练','GPU','SSH','队列'],
            '未核':['失败原因的主控裁定'],'source_sha256':C.sources(),'authorization_sha256':C.sha(auth)});print(failure['traceback'],flush=True);raise SystemExit(1)

if __name__=='__main__':main()
