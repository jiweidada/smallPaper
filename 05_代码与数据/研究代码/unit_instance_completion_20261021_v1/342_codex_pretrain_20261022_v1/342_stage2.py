"""342 stage-2 CPU diagnostics + bounded 20-step smoke; stops at controller gate."""
import importlib, sys, json, time, copy, random, traceback, ast
C = importlib.import_module('342_context')
import torch
import numpy as np
import train342 as T
import train340 as OLD
import train341 as V4
import geom342 as G
import core
import gcal
I = importlib.import_module('342_independent')
S1 = importlib.import_module('342_stage1')
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
COUNTS = {'cache_loads':0,'train340_loss_calls':0,'train342_loss_calls':0,
          'extra_head_forwards':0,'optimizer_steps':0,'global_model_loads':0,'global_forwards':0,
          'checkpoint_saves':0,'checkpoint_loads':0,'native_criterion_calls':0}

def compute(module, arm, model, batch):
    key = 'train340_loss_calls' if module is OLD else 'train342_loss_calls'
    COUNTS[key] += 1
    return module.compute_losses_340(arm,model,batch)

def opt_for(model):
    opt = T.build_optimizer(model)
    def stepped(*args): COUNTS['optimizer_steps'] += 1
    opt.register_step_post_hook(stepped)
    return opt

def stats(t):
    t = t.detach().reshape(-1).double()
    if not t.numel(): return {'count':0}
    return {'count':t.numel(),'min':float(t.min()),'max':float(t.max()),'mean':float(t.mean()),
            'quantiles_p0_p25_p50_p75_p95_p100':torch.quantile(t,torch.tensor([0.,.25,.5,.75,.95,1.],dtype=torch.float64)).tolist()}

def difference(a,b):
    diff = (a.detach()-b.detach()).abs()
    return {'shape':list(a.shape),'max_abs':float(diff.max()) if diff.numel() else 0.,
            'mean_abs':float(diff.mean()) if diff.numel() else 0.,'bit_equal':torch.equal(a,b)}

def state(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items()}

def same_state(a,b):
    return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)

def equal_tree(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    return a==b

def rng_seed():
    random.seed(2026102101)
    np.random.seed(2026102101)
    torch.manual_seed(2026102101)

def independent_kg(box,w):
    # Separable linear hat basis on 18 virtual centers (-1..16), folded at
    # boundaries; different from the primary floor + four-neighbor scatter.
    x0,y0,x1,y1 = map(float,box)
    left,top,right,bottom = map(float,w)
    x,y = (x0+x1)/2,(y0+y1)/2
    ax,ay = np.zeros(16),np.zeros(16)
    for idx in range(-1,17):
        wx = max(0.,1.-abs((x-left)/(right-left)*16-(idx+.5)))
        wy = max(0.,1.-abs((y-top)/(bottom-top)*16-(idx+.5)))
        ax[min(15,max(0,idx))] += wx
        ay[min(15,max(0,idx))] += wy
    return np.outer(ay,ax).reshape(-1)

def diagnostics(entries,batches):
    report = {'conclusion':'未核','windows':[],'same_initial_batch_comparison':{},'adapter_parity':{}}
    for e in entries:
        keep = G.intersecting_candidates(e['fullp_boxes'],e['window_xyxy_640'])
        bb = G.win_norm_xyxy(e['fullp_boxes'][keep],e['window_xyxy_640'])
        geo = G.geometry_cost(bb)[0]
        ref = I.geometry_raw(e['fullp_boxes'][keep].numpy(),e['window_xyxy_640'].numpy())
        err = float(np.abs(geo.numpy()-ref).max()) if ref.size else 0.
        assert err < 1e-5
        # Genuine 1920x1080 etc original image shapes are in the cache; no
        # original files/annotations outside development are opened here.
        kgerr=0.
        for j in e['G_W'].nonzero().flatten().tolist():
            kg = independent_kg(e['gt_boxes_640'][j],e['window_xyxy_640'])
            kgerr=max(kgerr,float(np.abs(kg-e['K_g'][j].numpy()).max()))
        assert kgerr < 1e-5
        report['windows'].append({'image_id':e['image_id'],'orig_wh':e['orig_wh'],
            'P_box_range':stats(bb),'correct_geometry':stats(geo),'geometry_oracle_max_abs_diff':err,
            'K_g_independent_max_abs_diff':kgerr})
    before_batch=OLD.collate(entries[:16])
    batch=batches[0]
    assert torch.equal(before_batch['P_valid'],batch['P_valid'])
    valid=batch['P_valid']
    report['old_candidate_range']=stats(before_batch['P_boxes'][valid])
    report['new_candidate_range']=stats(batch['P_boxes'][valid])
    report['padding_ignored']=True
    assert float(before_batch['P_boxes'][valid].abs().max()) > 32
    # Pixels are not necessarily restricted to 0..64: unclipped candidates
    # can have negative/outside-window edges in both spaces.
    report['conclusion']='确有尺度错误'
    for arm in 'NSGR':
        old=OLD.make_common_init(arm,1)
        new=T.make_common_init(arm,1)
        assert same_state(state(old),state(new)), 'initial states must match'
        captures=[]
        # Capture the actual legacy geometry from function return locals. This
        # executes the preserved implementation, without writing or patching it.
        def trace(frame,event,arg):
            if event=='return' and frame.f_code.co_name=='forward' and 'geometry' in frame.f_locals:
                captures.append(frame.f_locals['geometry'].detach().clone())
            return trace
        sys.settrace(trace)
        try:
            with torch.no_grad(): lo,oo=compute(OLD,arm,old,before_batch)
        finally:
            sys.settrace(None)
        assert len(captures)==1
        with torch.no_grad(): ln,on=compute(T,arm,new,batch)
        oldgeo=captures[0]
        newgeo=new.association.last_geometry
        assert not torch.equal(oldgeo,newgeo)
        per={k:difference(oo['plan'][k],on['plan'][k]) for k in ('T','z','residual')}
        report['same_initial_batch_comparison'][arm]={
            'state_bit_equal_before_forward':True,'old_geometry':stats(oldgeo[:,:,valid[0]]) if False else stats(oldgeo),
            'new_geometry':stats(newgeo),'geometry_difference':difference(oldgeo,newgeo),
            'plan_aux_difference':per,'loss_assoc_before':float(lo['loss_assoc']),
            'loss_assoc_after':float(ln['loss_assoc']),
            'loss_assoc_abs_difference':abs(float(lo['loss_assoc'])-float(ln['loss_assoc']))}
        if arm=='G':
            report['same_initial_batch_comparison'][arm]['generation_plan_difference']={
                k:difference(oo['generation_plan'][k],on['generation_plan'][k]) for k in ('T','z','residual')}
        # Holding corrected inputs fixed isolates the adapter from the input fix.
        with torch.no_grad():
            COUNTS['extra_head_forwards']+=2
            ref=old.forward_features(old.trunk(batch['rgb']),batch['P_boxes'],batch['P_cls'],batch['P_scores'],valid)
            adapted=new.forward_features(new.trunk(batch['rgb']),batch['P_boxes'],batch['P_cls'],batch['P_scores'],valid)
        parity={key:difference(ref[key],adapted[key]) for key in ('pred_boxes','pred_logits','query_states','tokens15','density_field')}
        parity.update({'plan_'+key:difference(ref['plan'][key],adapted['plan'][key]) for key in ('T','z','residual')})
        assert all(x['max_abs']<1e-5 for x in parity.values())
        report['adapter_parity'][arm]=parity
        assert torch.equal(adapted['tokens15'][...,:4][valid],batch['P_boxes'][valid])
    return report

def native_loss_contract(batch):
    from engine.deim.deim_criterion import DEIMCriterion
    from engine.deim.matcher import HungarianMatcher
    from uic_losses import compute_losses
    matcher=HungarianMatcher({'cost_class':2.,'cost_bbox':5.,'cost_giou':2.},use_focal_loss=True,alpha=.25,gamma=2.)
    crit=DEIMCriterion(matcher=matcher,weight_dict={'loss_mal':1,'loss_bbox':5,'loss_giou':2},
        losses=['mal','boxes'],alpha=.75,gamma=1.5,num_classes=10,mal_alpha=None)
    model=T.make_common_init('R',1)
    with torch.no_grad(): losses,out=compute(T,'R',model,batch)
    targets=[]; indices=[]
    denominator=float(max(1,sum(int(e['U_W'].sum()) for e in batch['entries'])))
    masses=[]
    for i,e in enumerate(batch['entries']):
        boxes=G.win_norm_cxcywh(e['gt_boxes_640'][e['U_W']],e['window_xyxy_640'])
        labels=e['gt_labels'][e['U_W']]
        term=compute_losses({'pred_boxes':out['pred_boxes'][i],'pred_logits':out['pred_logits'][i]},
                            boxes,labels,denominator=denominator)
        pairs=term['pairs']
        indices.append((torch.tensor([p[0] for p in pairs],dtype=torch.long),torch.tensor([p[1] for p in pairs],dtype=torch.long)))
        targets.append({'boxes':boxes,'labels':labels})
        # Independent mass objective evaluated in NumPy, positive entries only
        # for the logarithm while retaining +m for every grid/class.
        mstar=np.zeros((10,256),dtype=np.float64)
        for j in e['G_W'].nonzero().flatten().tolist():
            mstar[int(e['gt_labels'][j])]+=e['K_g'][j].numpy()
        m=out['mass'][i].numpy().astype(np.float64)
        termm=m-mstar
        positive=mstar>0
        termm[positive]+=mstar[positive]*np.log((mstar[positive]+1e-8)/(m[positive]+1e-8))
        masses.append(termm.sum()/max(1,int(e['G_W'].sum())))
    with torch.no_grad():
        native_mal=crit.loss_labels_mal(out,targets,indices,denominator)['loss_mal']
        native_box=crit.loss_boxes(out,targets,indices,denominator)
    COUNTS['native_criterion_calls']+=2
    native=native_mal+5*native_box['loss_bbox']+2*native_box['loss_giou']
    assert abs(float(losses['loss_det']-native))<1e-5
    assert abs(float(losses['loss_mass'])-np.mean(masses))<1e-5
    return {'batch_all_complete_U_W':int(denominator),
        'empty_U_W_images':sum(not bool(e['U_W'].any()) for e in batch['entries']),
        'loss_det_train342':float(losses['loss_det']),'loss_det_native':float(native),
        'native_abs_diff':abs(float(losses['loss_det']-native)),
        'loss_mass_train342':float(losses['loss_mass']),'loss_mass_numpy':float(np.mean(masses)),
        'mass_abs_diff':abs(float(losses['loss_mass'])-float(np.mean(masses))),
        'basis':'296 §5公式；batch分母口径沿342明确的主控解读；未调权重、标签或阈值'}

def frozen_detector(index):
    import stage1_cache_v3 as S
    cp=index['checkpoints'][0]
    assert C.sha(cp['path'])==cp['expected_SHA256_from_288']
    COUNTS['global_model_loads']+=1
    model=S.load_real_model(cp['path'])
    before=state(model)
    assert all(not p.requires_grad for p in model.parameters())
    with torch.no_grad():
        COUNTS['global_forwards']+=1
        model(torch.zeros(1,3,640,640))
    assert same_state(before,state(model))
    assert all(p.grad is None for p in model.parameters())
    return model,before,{'checkpoint_path':cp['path'],'checkpoint_sha256':C.sha(cp['path']),
                        'state_tensor_count':len(before),'parameters':sum(p.numel() for p in model.parameters()),
                        'synthetic_cpu_forward_state_unchanged':True,'all_requires_grad_false':True}

def smoke(batches,global_model,global_before):
    result={}
    for arm in 'NSGR':
        rng_seed()
        model=T.make_common_init(arm,1)
        model.train()
        initial=state(model)
        opt=opt_for(model)
        log=[]
        t0=time.perf_counter()
        for step in range(20):
            b=batches[step%len(batches)]
            lr=T.lr_at(step,1218)
            for gi,g in enumerate(opt.param_groups):
                g['lr']=lr*10 if arm=='G' and gi==2 else lr
            opt.zero_grad(set_to_none=True)
            losses,out=compute(T,arm,model,b)
            assert all(bool(torch.isfinite(v)) for v in losses.values())
            losses['loss_total'].backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert all(bool(torch.isfinite(g).all()) for g in grads)
            row={'step':step,'losses':{k:float(v.detach()) for k,v in losses.items()},
                 'lr_groups':[g['lr'] for g in opt.param_groups],
                 'all_gradients_finite':True,
                 'mass_total_mean':float(out['mass'].sum((1,2)).mean().detach()),
                 'saturated_row_fraction':float((out['plan']['u']>out['plan']['mass']).float().mean())}
            if arm=='G':
                row['kappa_before']=model.class_null.detach().tolist()
                row['kappa_gradient_before_clip']=model.class_null.grad.detach().tolist() if model.class_null.grad is not None else None
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            row['gradient_norm_before_clip']=float(norm)
            if arm=='G': row['kappa_gradient_after_clip']=model.class_null.grad.detach().tolist()
            opt.step()
            if arm=='G':
                row['kappa_after']=model.class_null.detach().tolist()
                row['kappa_update']=(np.array(row['kappa_after'])-np.array(row['kappa_before'])).tolist()
            assert all(p.grad is None for p in global_model.parameters())
            assert all(id(p) not in {id(q) for g in opt.param_groups for q in g['params']} for p in global_model.parameters())
            log.append(row)
        unchanged=same_state(global_before,state(global_model))
        assert unchanged
        changed=[k for k,v in model.state_dict().items() if not torch.equal(initial[k],v)]
        result[arm]={'steps':20,'batch_sizes':[len(b['entries']) for b in batches],
                     'cpu_seconds':time.perf_counter()-t0,'loss_finite':True,
                     'changed_state_tensor_count':len(changed),'changed_names':changed,
                     'parameter_count':sum(p.numel() for p in model.parameters()),
                     'global_all_grad_none_each_step':True,'global_state_unchanged_after_20_steps':unchanged,
                     'log':log}
        if arm=='G':
            a=np.array([r['kappa_gradient_before_clip'] for r in log])
            result[arm].update({'class_null_final':model.class_null.detach().tolist(),
                'per_class_nonzero_gradient_steps':(a!=0).sum(0).tolist(),
                'every_step_all_10_gradients_nonzero':bool((a!=0).all()),
                'each_step_any_gradient_nonzero':(a!=0).any(1).tolist(),
                'note':'初始末层为零可导致第一步检测链梯度为零；按实际记录，不据此调配方'})
            # Establish loss routing at the actually updated smoke state.
            l,_=compute(T,arm,model,batches[0])
            routing={}
            for name in ('loss_det','loss_mass','loss_assoc'):
                grad=torch.autograd.grad(l[name],model.class_null,allow_unused=True,retain_graph=True)[0]
                routing[name]=None if grad is None else grad.detach().tolist()
            assert routing['loss_mass'] is None and routing['loss_assoc'] is None
            result[arm]['class_null_loss_routing']=routing
        print(json.dumps({'completed_arm':arm,'steps':20,'seconds':result[arm]['cpu_seconds'],
                          'loss_total_last':log[-1]['losses']['loss_total']},ensure_ascii=False),flush=True)
    return result

def resume(batches):
    res={}
    for arm in 'NSGR':
        rng_seed()
        continuous=T.make_common_init(arm,1); oc=opt_for(continuous)
        COUNTS['train342_loss_calls']+=2
        T.run_steps(continuous,oc,batches,0,2,1218)
        cstate=state(continuous); copt=copy.deepcopy(oc.state_dict()); crng=T.capture_rng()
        rng_seed()
        split=T.make_common_init(arm,1); osplit=opt_for(split)
        COUNTS['train342_loss_calls']+=1
        T.run_steps(split,osplit,batches,0,1,1218)
        path=C.HERE/f'342_CPU_smoke_resume_{arm}.pt'
        COUNTS['checkpoint_saves']+=1
        T.save_checkpoint(path,split,osplit,1,0,2026102101,T.capture_rng())
        restored=T.make_common_init(arm,1); ores=opt_for(restored)
        COUNTS['checkpoint_loads']+=1
        st,ep,seed=T.load_checkpoint(path,restored,ores)
        lr_restored=[g['lr'] for g in ores.param_groups]
        assert lr_restored==[g['lr'] for g in osplit.param_groups]
        COUNTS['train342_loss_calls']+=1
        T.run_steps(restored,ores,batches,st,1,1218)
        param_equal=same_state(cstate,state(restored))
        optimizer_equal=equal_tree(copt,ores.state_dict())
        rng_equal=equal_tree(crng,T.capture_rng())
        assert param_equal and optimizer_equal and rng_equal
        assert (st,ep,seed)==(1,0,2026102101)
        res[arm]={'all_state_tensors_bit_equal':param_equal,'optimizer_bit_equal':optimizer_equal,
                  'rng_equal':rng_equal,'state_tensor_count':len(cstate),'restored_lr_groups':lr_restored,
                  'checkpoint_path':str(path),'checkpoint_sha256':C.sha(path),
                  'cursor':{'step':st,'epoch':ep,'shuffle_seed':seed},
                  'scope':'相同32张顺序、两个batch；continuous2 vs 1+save/load+1；仅CPU冒烟'}
    return res

def downstream_list():
    return [
        {'use':'collate P_W selection','expected':'640-plane xyxy + W','actual':'geom342.intersecting_candidates','status':'已验证'},
        {'use':'collate P_boxes','expected':'unclipped window-normalized xyxy','actual':'geom342.win_norm_xyxy','status':'已验证'},
        {'use':'head.forward_features / candidate_inputs','expected':'same normalized xyxy','actual':'inherited frozen validation + shared geometry adapter','status':'已验证'},
        {'use':'15D token first 4 values / S P-token','expected':'same normalized xyxy, no clip','actual':'tokens15[:4] bit equal P_boxes[valid]','status':'已验证'},
        {'use':'aux association d_out','expected':'[B,256,J] using 311 grid','actual':'342_head.SharedAssociation -> geom342.geometry_cost','status':'已验证'},
        {'use':'G generation d_out','expected':'same normalized xyxy','actual':'342_head.GeometryCalibratedHead342 -> geom342.geometry_cost','status':'已验证'},
        {'use':'U_W target ubox','expected':'window-normalized cxcywh','actual':'geom342.win_norm_cxcywh','status':'已验证'},
        {'use':'association owner subset','expected':'same P_W order','actual':'geom342.intersecting_candidates','status':'已验证'},
        {'use':'K_g','expected':'window-local row-major 16x16, unit mass','actual':'340 cache retained; independently rechecked via separable hat basis','status':'已验证'},
        {'use':'native loss adapter internal IoU/GIoU representation conversion','expected':'native cxcywh / xyxy parity','actual':'inherited uic_losses + frozen native box_ops','status':'已验证','boundary':'native loss code preserved; not a 640/window coordinate-space conversion'},
        {'use':'old train340/train341 / old geom342 / old g_effect','expected':'historical evidence only','actual':'retained, not used for corrected smoke; train340 only used for required before/after comparison','status':'已验证'}]

def main():
    res={'status':'未核','gate':'阶段2后等主控，不进入阶段3','已执行':COUNTS,
         '未执行':['真实训练','GPU','SSH','队列','heldout3236读取','阶段3至5','训练后G有效性'],
         '未核':['主控独立数值复核','主控6张PNG目视复核'], '待主控':['阶段2结果与差异文件确认'],
         'source_sha256':C.source_shas()}
    try:
        assert json.loads((C.HERE/'342_阶段1_result.json').read_text(encoding='utf-8'))['status']=='已验证'
        index,ids=C.inputs()
        entries=[T.load_entry(1,i) for i in ids]
        COUNTS['cache_loads']+=len(entries)
        for iid,e in zip(ids,entries): assert e['image_id']==iid
        batches=[T.collate(entries[:16]),T.collate(entries[16:])]
        res['development_ids']=ids; res['seed']=1
        res['cache_sha256']={str(C.WORK/'340_cache_seed1'/f'{i:06d}.pt'):C.sha(C.WORK/'340_cache_seed1'/f'{i:06d}.pt') for i in ids}
        res['diagnostic']=diagnostics(entries,batches)
        print('diagnosis complete: pixel-vs-normalized scale error confirmed',flush=True)
        res['loss_formula_contracts']=[native_loss_contract(b) for b in batches]
        res['downstream_coordinate_audit']=downstream_list()
        # Every LR is checked against v4, not inferred from one optimizer state.
        curves={arm:[] for arm in 'NSGR'}
        for step in range(1218):
            assert T.lr_at(step,1218)==V4.lr_at(step,1218)
            assert T.lr_at_kappa(step,1218)==V4.lr_at_kappa(step,1218)
        for arm in 'NSGR':
            o=T.build_optimizer(T.make_common_init(arm,1))
            groups=[{'lr':g['lr'],'weight_decay':g['weight_decay'],'parameters':sum(p.numel() for p in g['params'])} for g in o.param_groups]
            assert len(groups)==(3 if arm=='G' else 2)
            if arm=='G': assert groups[-1]['weight_decay']==0 and groups[-1]['parameters']==10
            curves[arm]=groups
        res['optimizer_routes']={'groups':curves,'lr_points_checked':1218,'v4_curve_bit_equal':True}
        gm,gs,gr=frozen_detector(index)
        res['global']=gr
        res['smoke']=smoke(batches,gm,gs)
        res['resume']=resume(batches)
        res['png_paths']=S1.vis(entries,ids,'342_阶段2_候选格点_Kg',include_kg=True)
        COUNTS['PNG生成']=6
        res['formula_changes_beyond_coordinate_repair']=[
            {'issue':'sum(max(1,|U_W|)) includes empty images','correction':'max(1,sum(|U_W|))','basis':'342主控分母口径'},
            {'issue':'batch-normalized L_det divided by batch size a second time','correction':'sum of native contributions normalized once','basis':'296原生分母；本轮直接native loss parity'},
            {'issue':'mass zero-target cells discarded +m; empty G_W forced zero','correction':'retain +m everywhere; only 0log0 term is zero','basis':'296 §5明确公式；NumPy独立对照'},
            {'issue':'resume overwrites all group LRs with first group LR','correction':'retain optimizer state_dict per-group LRs','basis':'317v4 x10路由及完整状态续跑'}]
        snap=json.loads((C.HERE/'342_输入快照.json').read_text(encoding='utf-8'))
        changed=[p for p,h in snap['source_sha256'].items() if C.sha(p)!=h]
        assert changed==[],changed
        res['preexisting_files_unchanged']={'checked':len(snap['source_sha256']),'changed_paths':changed}
        res['status']='已验证（CPU工程项）；待主控'
    except Exception as ex:
        res['status']='实现失败'
        res['exception']=repr(ex); res['traceback']=traceback.format_exc()
        print(res['traceback'],flush=True)
    res['source_sha256']=C.source_shas()
    C.write_json('342_阶段2_result.json',res)
    C.write_json('342_result_训练尺度诊断.json',res.get('diagnostic',{'status':'未执行'}))
    C.write_json('342_result_四臂CPU冒烟.json',{'arms':res.get('smoke',{}),'source_sha256':res['source_sha256'],
        '已执行':COUNTS,'未执行':res['未执行'],'未核':res['未核']})
    C.write_json('342_result_续跑逐位.json',{'arms':res.get('resume',{}),'source_sha256':res['source_sha256'],
        '已执行':{'arms_checked':len(res.get('resume',{}))},'未执行':['真实训练'],'未核':['全1218步续跑']})
    lines=['# 342 阶段2中文汇总','',f"状态：{res['status']}。已停在阶段2闸门，等待主控确认。",'',
        '科学方向判定未执行，真实训练未执行；本轮仅CPU工程验证。','',
        f"版本目录：{C.HERE}",'',
        f"尺度诊断：{res.get('diagnostic',{}).get('conclusion','未执行')}。候选框未clip，因此原像素值可超出0~64，正确归一化值也可超出0~1。",'',
        '修复前后使用相同batch/同公共初态，geometry、T/z/residual、loss_assoc差异见 342_result_训练尺度诊断.json。',
        '活动头的aux/G生成几何统一走geom342；目标框也由共享入口转换。311_v2/320_v2及原训练/缓存代码保留。','',
        '另按296明文修复检测分母/重复batch平均/零目标mass项，按完整状态恢复保留G每组LR；逐项差异见342_train342_与341.diff。',
        '两batch的损失直接对照native criterion与NumPy公式，结果见阶段2 result。','',
        '| 臂 | CPU步数 | loss有限 | 参数张量改变数 | 续跑逐位 |','|---|---:|---|---:|---|']
    for arm,item in res.get('smoke',{}).items():
        lines.append(f"| {arm} | {item['steps']} | {item['loss_finite']} | {item['changed_state_tensor_count']} | {res.get('resume',{}).get(arm,{}).get('all_state_tensors_bit_equal','未执行')} |")
    g=res.get('smoke',{}).get('G',{})
    lines+=['',f"G逐类非零梯度步数：{g.get('per_class_nonzero_gradient_steps','未执行')}（完整每步10维值见JSON，未改配方）。",'',
        '6张候选框/格点/K_g PNG已生成；主控目视核对未执行。',
        '全部调用计数、源SHA、32份缓存SHA、global零grad/state证据见342_阶段2_result.json。','',
        '待主控：确认本阶段结果及附带明文公式修复。阶段3至5未执行，342_开训前审查包.md尚未产出。']
    C.write_text('342_阶段2_中文汇总.md','\n'.join(lines)+'\n')
    print(json.dumps({'stage2_status':res['status'],'counts':COUNTS,'exception':res.get('exception')},ensure_ascii=False),flush=True)
    if res['status']=='实现失败': sys.exit(1)

if __name__=='__main__': main()
