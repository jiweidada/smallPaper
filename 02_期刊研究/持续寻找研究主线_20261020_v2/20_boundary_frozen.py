"""Frozen P2 boundary evidence: cloned new linear heads, independent one-step.

No detector training, no old parameter/optimizer writes, no heldout or remote.
"""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import datetime, gc, json, math, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as n
torch,np=n.torch,n.np
F=n.F
OLD=WORK/'英文小论文/data/09_DN初始化冻结诊断_20261020_v1'
DATA=WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1'
OUT=ROOT/'21_boundary_frozen'
ARMS=['original','difference','mean','full_concat','mean_matched','concat_matched']
GROUPS=['all','pre_IoU0_shortside_lt4','pre_IoU0','pre_IoU_ge05','capacity_leQ','capacity_gtQ']


def save(name,obj): n.save_json(OUT/name,obj)


def log(**kw):
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),**kw}
    print(json.dumps(row,ensure_ascii=False),flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def paired_terms(a,b):
    axy=n.box_cxcywh_to_xyxy(a); bxy=n.box_cxcywh_to_xyxy(b)
    inter=(torch.minimum(axy[:,2:],bxy[:,2:])-torch.maximum(axy[:,:2],bxy[:,:2])).clamp_min(0).prod(-1)
    area=(axy[:,2:]-axy[:,:2]).prod(-1); area_b=(bxy[:,2:]-bxy[:,:2]).prod(-1)
    union=area+area_b-inter
    iou=inter/union
    cover=(torch.maximum(axy[:,2:],bxy[:,2:])-torch.minimum(axy[:,:2],bxy[:,:2])).prod(-1)
    giou=iou-(cover-union)/cover
    return 5*(a-b).abs().sum(-1)+2*(1-giou),iou


def gather_geometry(cache,boxes):
    a=torch.cat([boxes[i,q] for i,(q,g) in enumerate(cache['ordinary']['indices'])])
    b=torch.cat([cache['targets'][i]['boxes'][g].double() for i,(q,g) in enumerate(cache['ordinary']['indices'])])
    return paired_terms(a,b)


def arm_features(sampled):
    inside,outside=sampled[:,:,:,0,:],sampled[:,:,:,1,:]
    return {'difference':((inside-outside)/math.sqrt(2)).flatten(-2),
            'mean':((inside+outside)/math.sqrt(2)).flatten(-2),
            'full_concat':sampled.flatten(-3)}


def sample_boundaries(features,boxes):
    """B,Q,4 sides,2 inward/outward strips,16 bilinear midpoint samples,C."""
    edge=n.box_cxcywh_to_xyxy(boxes.float())
    B,Q=boxes.shape[:2]
    tangent=(torch.arange(8,dtype=torch.float32)+.5)/8
    normal=torch.tensor([.25,.75],dtype=torch.float32)*(2/640)
    grids=[]
    for e in range(4):
        parts=[]
        for k in range(2):
            sign=(1 if e in [0,1] else -1)*(1 if k==0 else -1)
            if e in [0,2]:
                x=edge[...,e,None,None]+sign*normal[None,None,:,None]
                y=edge[...,1,None,None]+(edge[...,3]-edge[...,1])[...,None,None]*tangent
            else:
                x=edge[...,0,None,None]+(edge[...,2]-edge[...,0])[...,None,None]*tangent
                y=edge[...,e,None,None]+sign*normal[None,None,:,None]
            x,y=torch.broadcast_tensors(x,y)
            parts.append(torch.stack([x,y],-1).reshape(B,Q,16,2))
        grids.append(torch.stack(parts,2))
    grid=torch.stack(grids,2) # B,Q,4,2,16,2
    outside=((grid<0)|(grid>1)).any(-1)
    values=F.grid_sample(features,grid.reshape(B,-1,1,2)*2-1,mode='bilinear',padding_mode='border',align_corners=False)
    C=features.shape[1]
    sampled=values.reshape(B,C,Q,4,2,16).mean(-1).permute(0,2,3,4,1).double().contiguous()
    return sampled,{'outside_image_points':int(outside.sum()),'total_sample_points':outside.numel()}


def fresh_update(gradient,group):
    start=torch.zeros_like(gradient)
    binding={'new_boundary_W':{'group':group,'state':{
        'step':torch.tensor(0.),'exp_avg':torch.zeros_like(start),'exp_avg_sq':torch.zeros_like(start),
        **({'max_exp_avg_sq':torch.zeros_like(start)} if group['amsgrad'] else {})}}}
    norm=float(gradient.norm())
    factor=n.clip_factor(norm)
    value=n.adam_virtual([start],[gradient],binding,factor)[0]
    return value,{'gradient_norm':norm,'clip_factor':factor,'W_norm':float(value.norm())}


def synthetic():
    torch.manual_seed(78021)
    fake=torch.ones(2,5,4,2,3,dtype=torch.float64)
    assert bool((arm_features(fake)['difference']==0).all())
    fake=torch.randn_like(fake); feats=arm_features(fake)
    w=torch.randn(4,12,dtype=torch.float64)
    expanded=torch.stack([w.reshape(4,4,3)/math.sqrt(2),-w.reshape(4,4,3)/math.sqrt(2)],-2).flatten(-3)
    assert float((F.linear(feats['difference'],w)-F.linear(feats['full_concat'],expanded)).abs().max())<2e-15
    group={'amsgrad':False,'betas':[.9,.999],'lr':.0004,'weight_decay':.0001,'eps':1e-8,'maximize':False}
    g=torch.tensor([[.2,-.3,0.]],dtype=torch.float64)
    got,detail=fresh_update(g,group)
    clipped=g*detail['clip_factor']
    expected=-group['lr']*clipped/(clipped.abs()+group['eps'])
    assert float((got-expected).abs().max())<1e-16
    # Compare the aligned geometry and derivative to unchanged native matrix-diagonal path.
    a=torch.tensor([[.3,.4,.2,.1],[.2,.7,.04,.08]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[.4,.5,.1,.2],[.8,.7,.04,.08]],dtype=torch.float64)
    direct,_=paired_terms(a,b)
    from engine.deim.box_ops import generalized_box_iou
    exact=5*(a-b).abs().sum(-1)+2*(1-generalized_box_iou(n.box_cxcywh_to_xyxy(a),n.box_cxcywh_to_xyxy(b)).diagonal())
    assert float((direct-exact).abs().max())<1e-14
    ga=torch.autograd.grad(direct.sum(),a,retain_graph=True)[0]
    gb=torch.autograd.grad(exact.sum(),a)[0]
    assert float((ga-gb).abs().max())<1e-14
    return {'constant_diff_zero':True,'concat_contains_difference':True,'fresh_AdamW_matches_closed_first_step':True,
            'aligned_geometry_value_and_gradient_match_native':True}


def group_masks(cache):
    base=cache['native_pre_boxes'].double()
    _,old_iou=gather_geometry(cache,base)
    small=torch.cat([cache['targets'][i]['boxes'][g,2:].min(-1).values.double()*640<4
                     for i,(q,g) in enumerate(cache['ordinary']['indices'])])
    capacity=torch.cat([torch.full((len(q),),len(cache['targets'][i]['labels'])<=base.shape[1],dtype=torch.bool)
                        for i,(q,g) in enumerate(cache['ordinary']['indices'])])
    return {'all':torch.ones_like(old_iou,dtype=torch.bool),
            'pre_IoU0_shortside_lt4':(old_iou==0)&small,'pre_IoU0':old_iou==0,
            'pre_IoU_ge05':old_iou>=.5,'capacity_leQ':capacity,'capacity_gtQ':~capacity}


def metrics(cache,features,u,w):
    boxes=u.sigmoid() if w is None else (u+F.linear(features,w)).sigmoid()
    value,iou=gather_geometry(cache,boxes)
    groups={}
    for key,mask in group_masks(cache).items():
        count=int(mask.sum());den=float(cache['ordinary']['denominator'])
        groups[key]={'pairs':count,'denominator':den,'raw_geometric_sum':float(value[mask].sum()) if count else None,
                     'loss_contribution':float(value[mask].sum()/den) if count else None,
                     'mean_IoU':float(iou[mask].mean()) if count else None}
    return groups


def main():
    started=time.time(); OUT.mkdir(exist_ok=False)
    resource=n.resource_limits()
    save('00_implementation.json',{'protocol_sha256':n.sha(ROOT/'19_边界分支冻结核查事前协议.txt'),
                                  'script_sha256':n.sha(__file__),'resource':resource,'synthetic':synthetic(),
                                  'new_detector_forwards':0,'new_backbone_P2_forwards_planned':4,
                                  'new_detector_backwards':0,'model_optimizer_steps':0})
    old_protocol=json.loads((OLD/'01_锁定资产与诊断口径.json').read_text(encoding='utf8'))
    recipes=json.loads((DATA/'baseline_recipes.json').read_text(encoding='utf-8-sig'))
    all_rows,sources=[],[]
    for seed in [1,2]:
        spec=old_protocol['checkpoints'][str(seed)]; cp=Path(spec['path'])
        assert n.sha(cp)==spec['sha256']
        raw=torch.load(cp,map_location='cpu',weights_only=False); assert raw['last_epoch']==23
        model=n.build_model(recipes[str(seed)]);model.load_state_dict(raw['model'],strict=True);model.train()
        for p in model.parameters():p.requires_grad_(False)
        before=n.model_hash(model);buffers={k:v.clone() for k,v in model.named_buffers()}
        del raw;gc.collect()
        caches={};X={};U={};group=None
        receipt=json.loads((OLD/f'seed{seed}/capture_complete.json').read_text(encoding='utf8'))
        cache_shas={item['file']:item['sha256'] for item in receipt['capture_files']}
        for record in old_protocol['batches']:
            batch=record['batch']; path=OLD/f'seed{seed}/batch{batch:02d}_frozen_head.pt'
            assert n.sha(path)==cache_shas[path.name]
            cache=torch.load(path,map_location='cpu',weights_only=False)
            payload_path=DATA/'shared_inputs/inputs'/record['file']
            assert n.sha(payload_path)==record['sha256']
            payload=torch.load(payload_path,map_location='cpu',weights_only=False)
            assert all(torch.equal(a['boxes'],b['boxes']) and torch.equal(a['labels'],b['labels']) for a,b in zip(cache['targets'],payload['targets']))
            with torch.no_grad():
                for key,val in model.named_buffers():val.copy_(buffers[key])
                p2=model.backbone.stages[0](model.backbone.stem(payload['images']))
                assert p2.shape[-2:]==(160,160),p2.shape
                sampled,sampling=sample_boundaries(p2,cache['native_pre_boxes'])
                feats=arm_features(sampled)
                u=torch.logit(cache['native_pre_boxes'].double())
                assert torch.isfinite(u).all()
                zero=torch.zeros(4,feats['difference'].shape[-1],dtype=torch.float64)
                initial_error=float(((u+F.linear(feats['difference'],zero)).sigmoid()-cache['native_pre_boxes'].double()).abs().max())
                assert initial_error<3e-16
                base_terms,_=gather_geometry(cache,cache['native_pre_boxes'].double())
                native_loss_error=abs(float(base_terms.sum()/cache['ordinary']['denominator'])-cache['native_ordinary_pre_loss'])
                assert native_loss_error<3e-5,native_loss_error
                for key,val in model.named_buffers():val.copy_(buffers[key])
            caches[batch]=cache;X[batch]=feats;U[batch]=u
            this_group=next(iter(cache['optimizer_bindings'].values()))['group']
            assert group is None or group==this_group
            group=this_group
            sources.append({'seed':seed,'batch':batch,'checkpoint':str(cp),'checkpoint_sha256':spec['sha256'],
                            'cached_native_pre':str(path),'cached_native_pre_sha256':n.sha(path),
                            'payload':str(payload_path),'payload_sha256':n.sha(payload_path),
                            'P2_shape':list(p2.shape),'initial_box_error':initial_error,
                            'FP64_geometry_vs_saved_native_FP32_error':native_loss_error,**sampling})
            log(seed=seed,batch=batch,stage='P2_features_complete',P2_shape=list(p2.shape),seconds=time.time()-started)
            del payload,p2,sampled;gc.collect()
        assert n.model_hash(model)==before
        assert n.sha(cp)==spec['sha256']
        del model;gc.collect()
        for generator,evaluation in [(0,1),(1,0)]:
            weights={};details={}
            for arm in ['difference','mean','full_concat']:
                x=X[generator][arm]
                w=torch.zeros(4,x.shape[-1],dtype=torch.float64,requires_grad=True)
                boxes=(U[generator]+F.linear(x,w)).sigmoid()
                terms,_=gather_geometry(caches[generator],boxes)
                loss=terms.sum()/caches[generator]['ordinary']['denominator']
                g=torch.autograd.grad(loss,w)[0]
                weights[arm],details[arm]=fresh_update(g,group)
                move=F.linear(x,weights[arm]);details[arm]['generator_logit_RMS']=float(move.square().mean().sqrt())
                details[arm]['parameters']=w.numel()
            for new,old in [('mean_matched','mean'),('concat_matched','full_concat')]:
                old_rms=details[old]['generator_logit_RMS'];target_rms=details['difference']['generator_logit_RMS']
                scale=target_rms/old_rms if old_rms else 0.
                weights[new]=weights[old]*scale
                details[new]={'generator_only_logit_scale':scale,'parameters':weights[new].numel(),
                              'generator_logit_RMS':float(F.linear(X[generator][old],weights[new]).square().mean().sqrt())}
            result={}
            with torch.no_grad():
                for arm in ARMS:
                    feature_arm='mean' if arm=='mean_matched' else 'full_concat' if arm=='concat_matched' else arm
                    feat=None if arm=='original' else X[evaluation][feature_arm]
                    result[arm]=metrics(caches[evaluation],feat,U[evaluation],weights.get(arm))
            all_rows.append({'seed':seed,'generator':generator,'evaluation':evaluation,'fresh_group':group,'updates':details,'metrics':result})
            log(seed=seed,generator=generator,evaluation=evaluation,stage='cross_evaluation_complete',seconds=time.time()-started)
        del caches,X,U;gc.collect()
    pooled={}
    for seed in [1,2]:
        pooled[seed]={}
        selected=[r for r in all_rows if r['seed']==seed]
        for arm in ARMS:
            pooled[seed][arm]={}
            for key in GROUPS:
                cells=[r['metrics'][arm][key] for r in selected]
                count=sum(r['pairs'] for r in cells);den=sum(r['denominator'] for r in cells)
                total=sum(r['raw_geometric_sum'] or 0 for r in cells)
                pooled[seed][arm][key]={'pairs':count,'denominator':den,'raw_geometric_sum':total if count else None,
                                      'loss_contribution':total/den if count else None}
    decision={}
    for seed in [1,2]:
        p=pooled[seed];key='pre_IoU0_shortside_lt4'
        candidate=p['difference'][key]['loss_contribution']
        controls={a:p[a][key]['loss_contribution'] for a in ARMS if a!='difference'}
        contrasts={a:candidate-v if candidate is not None and v is not None else None for a,v in controls.items()}
        ordinary_delta=p['difference']['all']['loss_contribution']-p['original']['all']['loss_contribution']
        passed=all(v is not None and v<0 for v in contrasts.values()) and ordinary_delta<=0
        decision[seed]={'primary_differences':contrasts,'all_ordinary_difference_minus_original':ordinary_delta,'passed':passed}
    result={'state':'complete','scope':'New W only, single fresh-moment virtual step; backbone features fixed; native ordinary pre matches/normalization. Not full detector or historical optimizer endpoint.',
            'protocol_sha256':n.sha(ROOT/'19_边界分支冻结核查事前协议.txt'), 'rows':all_rows,'pooled':pooled,'decision':decision,
            'retain_feature_premise_for_later_novelty_review':all(r['passed'] for r in decision.values()),
            'sources':sources,'seconds':time.time()-started,'model_parameters_and_buffers_unchanged':True,
            'new_backbone_P2_forwards':4,'new_detector_forwards':0,'branch_backwards':12,
            'model_optimizer_steps':0,'heldout_reads':False,'remote_operations':False}
    save('01_results.json',result)
    save('02_manifest.json',{p.name:n.sha(p) for p in OUT.iterdir() if p.is_file()})
    log(stage='complete',decision=decision,retain_feature_premise=result['retain_feature_premise_for_later_novelty_review'])


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'retry_identical_script':False})
        raise
