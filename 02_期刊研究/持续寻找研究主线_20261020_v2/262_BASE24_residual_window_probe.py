"""261: fixed-cache residual-window resource probe. Root reviews/runs.
No detector/checkpoint/new forward/loss/autograd/update/heldout.
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='2'
sys.dont_write_bytecode=True
import argparse, ast, gc, hashlib, importlib.util, pathlib, time, traceback, types
ROOT=pathlib.Path(__file__).resolve().parent
PROTOCOL=ROOT/'261_C_局部补全选窗_先验协议.txt'
REUSE=ROOT/'237_BASE24_saved_GO_repair_probe.py'
REUSE_SHA='20a98d07a57def68399fa71338e05388b6cf7f359117a44d89ceaaabf7ff348d'
if hashlib.sha256(REUSE.read_bytes()).hexdigest()!=REUSE_SHA: raise RuntimeError('237 changed')
spec=importlib.util.spec_from_file_location('_probe262_reuse237',REUSE)
V=importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
P=V.P; H=P.H; np,torch=P.np,P.torch
need,sha,save=P.require,P.sha,P.save
POST=P.A.NATIVE/'postprocessor.py'
COUNTS=dict(prediction_cache_loads=0,native_matcher_calls=0,native_postprocess_calls=0,synthetic_native_postprocess_calls=0,
            graph_maxmatching_calls=0,model_builds=0,checkpoint_loads=0,model_forwards=0,
            criterion_forwards=0,autograd_calls=0,model_backwards=0,model_vjps=0,optimizer_steps=0,heldout_reads=0)
ARMS=('A_density','B_residual','C_oracle'); THRESHOLDS=(.5,.75)

def load_post_api():
    # Isolate unmodified source functions; do not instantiate an nn.Module/model.
    import torchvision
    tree=ast.parse(POST.read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PostProcessor')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults={a.arg:ast.literal_eval(d) for a,d in zip(init.args.args[-len(init.args.defaults):],init.args.defaults)}
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scale_boxes_to_original_image')
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    ns={'torch':torch,'torchvision':torchvision,'F':torch.nn.functional}
    exec(compile(ast.Module(body=[helper,forward],type_ignores=[]),str(POST),'exec'),ns)
    return defaults,ns['forward'],torchvision

def post_config(recipe,defaults):
    need(recipe['postprocessor']=='PostProcessor' and recipe['num_classes']==10 and recipe['use_focal_loss'] is True
         and recipe['remap_mscoco_category'] is False and recipe['evaluator']['max_dets']==100,'native post recipe')
    local=recipe['PostProcessor']; top=local.get('num_top_queries',recipe.get('num_top_queries',defaults['num_top_queries']))
    threshold=local.get('score_threshold',recipe.get('score_threshold',defaults['score_threshold']))
    need(top==300 and threshold==.001,'fixed native top300/threshold')
    return types.SimpleNamespace(num_top_queries=top,num_classes=10,use_focal_loss=True,
                                 score_threshold=threshold,remap_mscoco_category=False,deploy_mode=False)

def post_selection(output,hw,cfg,forward):
    logits,boxes=output['pred_logits'],output['pred_boxes']; h,w=hw
    sizes=torch.tensor([[w,h]],dtype=boxes.dtype).repeat(len(boxes),1)
    native=forward(cfg,output,sizes,resize_pad=None)
    scores,flat=logits.sigmoid().flatten(1).topk(cfg.num_top_queries,dim=-1)
    # Same no-padding cxcywh conversion as torchvision; no width clamp or clipping.
    xy=torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),dim=-1)*sizes.repeat(1,2)[:,None,:]
    rows=[]
    for i,expected in enumerate(native):
        keep=scores[i]>cfg.score_threshold; selected=flat[i,keep]; q=selected//cfg.num_classes; c=selected%cfg.num_classes
        need(torch.equal(expected['labels'],c) and torch.equal(expected['scores'],scores[i,keep])
             and torch.equal(expected['boxes'],xy[i,q]),'native postprocessor exact parity')
        rows.append(dict(flat=selected.numpy().copy(),q=q.numpy().copy(),labels=c.numpy().copy(),
                         scores=scores[i,keep].numpy().copy(),boxes=expected['boxes'],
                         natural_top300=flat[i].tolist(),threshold_kept=keep.tolist()))
    return rows

def perclass100(labels,scores):
    order=np.argsort(-scores,kind='mergesort'); counts={}; keep=[]
    for i in order:
        c=int(labels[i]); counts[c]=counts.get(c,0)+1
        if counts[c]<=100: keep.append(int(i))
    return np.asarray(keep,dtype=np.int64)

def pixels(boxes,hw,tv):
    h,w=hw
    return tv.ops.box_convert(boxes,in_fmt='cxcywh',out_fmt='xyxy')*boxes.new_tensor([w,h,w,h])

def pair_iou(a,b,box):
    va=torch.isfinite(a).all(-1)&(a[:,2:]>a[:,:2]).all(-1)
    vb=torch.isfinite(b).all(-1)&(b[:,2:]>b[:,:2]).all(-1)
    result=np.zeros((len(a),len(b)),dtype=np.float32)
    if bool(va.any()) and bool(vb.any()):
        value=box.box_iou(a[va],b[vb])[0]
        need(bool(torch.isfinite(value).all()),'finite FP32 IoU')
        result[np.ix_(np.flatnonzero(va.numpy()),np.flatnonzero(vb.numpy()))]=value.numpy()
    return result

def match(edges):
    COUNTS['graph_maxmatching_calls']+=1
    pairs=H.match_capacity_one(edges)
    need(len(pairs)==P.A.matching_count(edges),'Kuhn/SciPy maximum cardinality')
    return pairs

def greedy(iou,labels,gtlabels,threshold):
    used=set(); pairs=[]
    for q in range(len(labels)):  # P100 already score ordered; stable native tie order.
        best=threshold; owner=None
        for g in range(len(gtlabels)):
            if g not in used and labels[q]==gtlabels[g] and iou[q,g]>=best:
                owner=g; best=float(iou[q,g])  # COCO non-ignore tie goes to later GT.
        if owner is not None: used.add(owner); pairs.append((q,owner))
    return pairs

def make_grid(hw):
    h,w=hw
    return np.asarray([(x,y,min(x+64,w),min(y+64,h)) for y in range(0,h,64) for x in range(0,w,64)],dtype=np.float64)

def cell_ids(centers,hw):
    h,w=hw; nx=(w+63)//64; ny=(h+63)//64
    valid=np.isfinite(centers).all(-1)&(centers>=0).all(-1)&(centers[:,0]<=w)&(centers[:,1]<=h)
    result=np.full(len(centers),-1,dtype=np.int64)
    ids=np.floor(centers[valid]/64).astype(np.int64)
    result[valid]=np.minimum(ids[:,1],ny-1)*nx+np.minimum(ids[:,0],nx-1)
    return result

def winner(scores):
    return int(np.argmax(scores)) if len(scores) and float(np.max(scores))>0 else None

def choose_without_GT(output,index,post,hw,box,tv):
    grid=make_grid(hw); n=len(grid); h,w=hw
    centers=(output['pred_boxes'][index,:,:2]*torch.tensor([w,h])).numpy().astype(np.float64)
    ids=cell_ids(centers,hw); density=np.bincount(ids[ids>=0],minlength=n)
    stages=[('pre',output['pre_outputs']),('aux_0',output['aux_outputs'][0]),
            ('aux_1',output['aux_outputs'][1]),('enc_0',output['enc_aux_outputs'][0])]
    residual=np.zeros(n,dtype=np.float64); pool=[]
    for name,stage in stages:
        bb,zz=stage['pred_boxes'][index],stage['pred_logits'][index]
        need(bb.shape==(300,4) and zz.shape==(300,10) and bb.dtype==zz.dtype==torch.float32
             and not bb.requires_grad and not zz.requires_grad and bool(torch.isfinite(bb).all())
             and bool(torch.isfinite(zz).all()),'fixed selected300 stage pool')
        pp=pixels(bb,hw,tv); cells=cell_ids((bb[:,:2]*bb.new_tensor([w,h])).numpy().astype(np.float64),hw)
        legal=(pp[:,2:]>pp[:,:2]).all(-1).numpy(); usable=legal&(cells>=0)
        overlaps=pair_iou(pp,post['boxes'],box)
        best=overlaps.max(-1) if overlaps.shape[1] else np.zeros(300,dtype=np.float32)
        conf=zz.sigmoid().max(-1).values.numpy().astype(np.float64)
        weights=conf*(1-np.clip(best.astype(np.float64),0,1))
        residual+=np.bincount(cells[usable],weights=weights[usable],minlength=n)
        pool.append(dict(stage=name,records=300,usable=int(usable.sum()),invalid_box=int((~legal).sum()),
                         outside_center=int((cells<0).sum()),weight_sum=float(weights[usable].sum())))
    return grid,dict(A_density=winner(density),B_residual=winner(residual)),dict(
        density=density.tolist(),residual=residual.tolist(),pool=pool,pool_records=1200,
        final_centers_outside=int((ids<0).sum()),pool_deduplicated=False,GT_used=False)

def visibility(window,gtxy,gtcells,cell):
    intersection=np.maximum(np.minimum(gtxy[:,2:],window[2:])-np.maximum(gtxy[:,:2],window[:2]),0).prod(-1)
    fraction=intersection/(gtxy[:,2:]-gtxy[:,:2]).prod(-1)
    full=(gtxy[:,:2]>=window[:2]).all(-1)&(gtxy[:,2:]<=window[2:]).all(-1)
    return full,gtcells==cell,fraction

def baseline(overlaps,plabel,gtlabel,tiny):
    reports={}; graphs={}; tinyids=np.flatnonzero(tiny)
    for t in THRESHOLDS:
        key=str(t); edge=(overlaps>=t)&(plabel[:,None]==gtlabel[None,:]); pairs=match(edge)
        degree=edge.sum(0); covered={g for q,g in pairs}; missing=[g for g in range(len(gtlabel)) if g not in covered]
        tp=match(edge[:,tiny]); gp=greedy(overlaps,plabel,gtlabel,t)
        graphs[key]=edge
        reports[key]=dict(max_all=len(pairs),max_tiny_independent=len(tp),pairs=pairs,
             tiny_pairs=[(q,int(tinyids[g])) for q,g in tp],unmatched_GT=missing,degree=degree.tolist(),
             degree0_GT=np.flatnonzero(degree==0).tolist(),unmatched_with_edges=[g for g in missing if degree[g]>0],
             greedy100_all=len(gp),greedy100_tiny_among_all=sum(bool(tiny[g]) for q,g in gp),greedy_pairs=gp)
    return reports,graphs

def ideal_gain(graph,eligible,tiny,base):
    augmented=np.concatenate((graph,np.tile(eligible,(4,1))),axis=0)
    need(np.array_equal(augmented[:len(graph)],graph),'all original candidate edges retained')
    allpairs=match(augmented); tp=match(augmented[:,tiny]); delta=len(allpairs)-base['max_all']; td=len(tp)-base['max_tiny_independent']
    need(0<=delta<=4 and 0<=td<=4,'four-instance monotone capacity')
    return dict(all_after=len(allpairs),all_gain=delta,tiny_independent_after=len(tp),tiny_independent_gain=td,
                ideal_assigned_GT=[g for q,g in allpairs if q>=len(graph)],original_edges_retained=True)

def image_probe(index,output,target,post,hw,box,tv):
    grid,chosen,evidence=choose_without_GT(output,index,post,hw,box,tv)  # No target input.
    gt=target['boxes']; labels=target['labels'].numpy(); h,w=hw
    gtxy=pixels(gt,hw,tv); gtnp=gtxy.numpy().astype(np.float64)
    centers=(gt[:,:2]*gt.new_tensor([w,h])).numpy().astype(np.float64); gcells=cell_ids(centers,hw)
    tiny=(gt.numpy().astype(np.float64)[:,2:]*np.asarray([w,h])).min(-1)<4
    keep=perclass100(post['labels'],post['scores']); overlaps=pair_iou(post['boxes'][keep],gtxy,box)
    base,graphs=baseline(overlaps,post['labels'][keep],labels,tiny)
    missing=np.zeros(len(gt),dtype=bool); missing[base['0.5']['unmatched_GT']]=True
    degree0=np.asarray(base['0.5']['degree'])==0
    oracle=np.asarray([int((visibility(win,gtnp,gcells,j)[0]&missing).sum()) for j,win in enumerate(grid)])
    chosen['C_oracle']=winner(oracle); evidence['oracle_missing_full']=oracle.tolist()
    windows={}
    for arm in ARMS:
        cell=chosen[arm]
        if cell is None: windows[arm]=dict(cell=None,NA=True,reason='no_positive_selection_evidence'); continue
        full,center,frac=visibility(grid[cell],gtnp,gcells,cell); table={}
        for name,mask in (('complete',full),('center',center),('intersects',frac>0)):
            table[name]=dict(GT=int(mask.sum()),actual_sub4=int((mask&tiny).sum()),
                 natural_unmatched=int((mask&missing).sum()),natural_unmatched_tiny=int((mask&missing&tiny).sum()),
                 degree0=int((mask&degree0).sum()),unmatched_with_edges=int((mask&missing&~degree0).sum()))
        gains={name:{str(t):ideal_gain(graphs[str(t)],mask,tiny,base[str(t)]) for t in THRESHOLDS}
               for name,mask in (('complete',full),('center',center))}
        visible=np.flatnonzero((frac>0)|center|full)
        windows[arm]=dict(cell=cell,NA=False,xyxy_pixels=grid[cell].tolist(),
            selection_score=float(evidence[{'A_density':'density','B_residual':'residual','C_oracle':'oracle_missing_full'}[arm]][cell]),
            visibility=table,ideal4=gains,GT_rows=[dict(GT=int(g),label=int(labels[g]),actual_sub4=bool(tiny[g]),
                complete=bool(full[g]),center=bool(center[g]),area_fraction=float(frac[g]),
                natural_unmatched=bool(missing[g]),degree0=bool(degree0[g])) for g in visible])
    return dict(image_index=index,GT=len(gt),actual_sub4=int(tiny.sum()),shape_hw=list(hw),grid_xyxy=grid.tolist(),
        post=dict(P_count=len(post['flat']),P_unique_queries=len(set(post['q'].tolist())),P100_count=len(keep),
                  P_flat=post['flat'].tolist(),P_scores=post['scores'].tolist(),P_boxes_xyxy=post['boxes'].tolist(),
                  natural_top300=post['natural_top300'],threshold_kept=post['threshold_kept'],P100_indices=keep.tolist()),
        baseline=base,selection=evidence,windows=windows,source_scope='cached selected ordinary outputs only')

def synthetic(forward,tv):
    cfg=types.SimpleNamespace(num_top_queries=300,num_classes=10,use_focal_loss=True,score_threshold=.001,
                              remap_mscoco_category=False,deploy_mode=False)
    zz=torch.full((1,300,10),-20.); zz[0,0,0]=12; zz[0,0,1]=11
    bb=torch.tensor([.5,.5,.1,.1]).repeat(1,300,1)
    p=post_selection(dict(pred_boxes=bb,pred_logits=zz),(544,544),cfg,forward)[0]
    COUNTS['synthetic_native_postprocess_calls']+=1
    need(p['q'].tolist()==[0,0] and p['labels'].tolist()==[0,1],'native flatten multiple classes per query')
    cfg.score_threshold=.5; zz.fill_(-20); zz[0,0,0]=0
    need(len(post_selection(dict(pred_boxes=bb,pred_logits=zz),(544,544),cfg,forward)[0]['q'])==0,'strict score threshold')
    COUNTS['synthetic_native_postprocess_calls']+=1
    keep=perclass100(np.array([0]*101+[1]*3),np.ones(104))
    need(keep.tolist()==list(range(100))+list(range(101,104)),'stable per-class100')
    need(len(match(np.array([[1],[1]],dtype=bool)))==1 and len(match(np.eye(2,dtype=bool)))==2,'candidate capacity one')
    need(len(match(np.zeros((0,2),dtype=bool)))==0 and len(match(np.zeros((2,0),dtype=bool)))==0,'empty graph')
    iou=np.array([[.8,.7],[.6,0]],dtype=np.float32)
    need(len(greedy(iou,np.zeros(2),np.zeros(2),.5))==1 and len(match(iou>=.5))==2,'greedy is not maximum')
    edge=np.array([[1,1]],dtype=bool); base={'max_all':1,'max_tiny_independent':1}
    z=ideal_gain(edge,np.array([1,0],dtype=bool),np.ones(2,dtype=bool),base)
    need(z['all_gain']==1,'inside covered GT releases candidate to outside via augmentation')
    grid=make_grid((544,544)); ids=cell_ids(np.array([[64.,0],[544,544],[-1,0]]),(544,544))
    need(len(grid)==81 and grid[-1].tolist()==[512,512,544,544] and ids.tolist()==[1,80,-1],'64 grid boundary/remainder')
    full,center,frac=visibility(grid[0],np.array([[63,10,65,12],[64,20,66,22],[0,0,2,2]],dtype=float),np.array([1,1,0]),0)
    need(full.tolist()==[False,False,True] and center.tolist()==[False,False,True] and frac.tolist()==[.5,0.,1.],'visibility distinctions')
    need(winner(np.zeros(3)) is None and winner(np.array([1.,1.,0]))==0,'NA and y/x tie')
    return dict(flatten_multiclass=True,strict_threshold=True,perclass100=True,competing_candidates=True,
                external_augmentation=True,greedy_not_maximum=True,grid_remainder=True,visibility=True,NA=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--synthetic-only',action='store_true')
    parser.add_argument('--out',type=pathlib.Path,default=ROOT/'263_BASE24_residual_window_probe'); args=parser.parse_args()
    started=time.monotonic(); resources=H.resource_limits()
    need(resources['priority']=='below_normal' and torch.version.cuda is None,'CPU resource contract')
    for key,path in (('postprocessor',POST),('evaluator',P.A.EVAL_SOURCE),('box_ops',P.A.NATIVE/'box_ops.py'),('dfine_utils',P.A.NATIVE/'dfine_utils.py')):
        need(sha(path)==P.A.EXPECTED[key],'native source before synthetic: '+key)
    native=H.load_native_math(); box=sys.modules[native.__package__+'.box_ops']; defaults,forward,tv=load_post_api()
    tests=dict(window=synthetic(forward,tv),GO_compatibility=V.compatibility_self_checks())
    if args.synthetic_only: print(H.json.dumps(dict(tests=tests,counts=COUNTS),ensure_ascii=False)); return
    out=args.out.resolve(); need(out.parent==ROOT and out.name.startswith('263_') and not out.exists(),'new263 only'); out.mkdir()
    files,expected=V.manifest(); files.update(reuse237=files['script'],protocol236=files['protocol'],script=pathlib.Path(__file__),
        protocol=PROTOCOL,postprocessor=POST,evaluator_source=P.A.EVAL_SOURCE)
    expected.update(reuse237=REUSE_SHA,postprocessor=P.A.EXPECTED['postprocessor'],evaluator_source=P.A.EXPECTED['evaluator'])
    stage='sources'; before={}; batches=[]; artifacts={}
    try:
        before={k:sha(p) for k,p in files.items()}; need(all(before[k]==v for k,v in expected.items()),'fixed source/input hashes')
        save(out/'00_manifest_and_synthetic.json',dict(paths={k:str(p.resolve()) for k,p in files.items()},sha256=before,tests=tests,resources=resources))
        recipes=P.read_json(files['recipe']); shapes=P.input_shapes(); audit=V.load_union_audit()
        for letter,seed in (('A',1),('B',2)):
            predrows,analysisrows,provenance=P.metadata(letter,seed); recipe=recipes[str(seed)]; cfg=post_config(recipe,defaults)
            compute,union=P.native_readers(box,recipe)
            for batch in range(8):
                P.M.guard(); stage=f'{letter}_batch{batch:02d}'; hw=tuple(shapes[batch]['shape'][2:])
                need(predrows[batch]['input_sha256']==shapes[batch]['sha256'],'actual input shape seal')
                pp=P.DATA/letter/'frozen_predictions/development'/predrows[batch]['relative']
                ap=P.DATA/letter/'development'/analysisrows[batch]['relative']
                need(sha(pp)==predrows[batch]['sha256'] and sha(ap)==analysisrows[batch]['sha256'],'batch hashes')
                prior=P.read_json(ap); value=torch.load(pp,map_location='cpu',weights_only=False); COUNTS['prediction_cache_loads']+=1
                output,targets,original,costs,denom=V.validate_batch(value,prior,batch,seed,predrows[batch],recipe,compute,union,audit)
                COUNTS['native_matcher_calls']+=1
                posts=post_selection(output,hw,cfg,forward); COUNTS['native_postprocess_calls']+=1
                images=[image_probe(i,output,t,posts[i],hw,box,tv) for i,t in enumerate(targets)]
                report=dict(letter=letter,seed=seed,batch=batch,shape=shapes[batch]['shape'],input_sha256=predrows[batch]['input_sha256'],
                    prediction_sha256=predrows[batch]['sha256'],source_ids=predrows[batch]['source_ids'],provenance=provenance,
                    denominator_and_identity_audit=denom,native_postprocess_exact=True,postprocess=dict(top=300,score_threshold=cfg.score_threshold,perclass_cap=100),images=images)
                filename=stage+'.json'; save(out/filename,report); artifacts[filename]=sha(out/filename); batches.append(report)
                del value,output,targets,original,costs,posts; gc.collect(); print(stage,'complete',flush=True)
        need(COUNTS['prediction_cache_loads']==COUNTS['native_matcher_calls']==COUNTS['native_postprocess_calls']==16,'exact cache budget')
        summaries={}
        for letter in ('A','B'):
            images=[im for b in batches if b['letter']==letter for im in b['images']]; arms={}
            for arm in ARMS:
                windows=[im['windows'][arm] for im in images]; valid=[w for w in windows if not w['NA']]
                arms[arm]=dict(NA=sum(w['NA'] for w in windows),selected=len(valid),
                    natural_missing_complete=sum(w['visibility']['complete']['natural_unmatched'] for w in valid),
                    natural_missing_complete_tiny=sum(w['visibility']['complete']['natural_unmatched_tiny'] for w in valid),
                    ideal4={v:{str(t):{key:sum(w['ideal4'][v][str(t)][key] for w in valid) for key in ('all_gain','tiny_independent_gain')}
                               for t in THRESHOLDS} for v in ('complete','center')})
            main_gain=arms['B_residual']['ideal4']['complete']['0.5']['tiny_independent_gain']
            summaries[letter]=dict(inputs=len(images),GT=sum(im['GT'] for im in images),actual_sub4=sum(im['actual_sub4'] for im in images),
                arms=arms,primary_B_complete_tiny05_gain=main_gain,necessary_screen=main_gain>=1)
        passed=[summaries[a]['necessary_screen'] for a in ('A','B')]
        unchanged={k:sha(p)==before[k] for k,p in files.items()}; need(all(unchanged.values()),'sources changed')
        need(all(COUNTS[k]==0 for k in ('model_builds','checkpoint_loads','model_forwards','criterion_forwards','autograd_calls','model_backwards','model_vjps','optimizer_steps','heldout_reads')),'zero model/gradient budget')
        save(out/'01_results.json',dict(state='complete',summaries=summaries,necessary_screen_both=all(passed),
            screen_status='both' if all(passed) else ('mixed' if any(passed) else 'neither'),counts=COUNTS,resources=resources,
            sources_unchanged=unchanged,artifact_sha256=artifacts,seconds=time.monotonic()-started,
            conditional_input_necessary_proven=False,actual_completion_head_tested=False,training_authorized=False,AP=False,
            deployment_score_refusion_tested=False,arbitrary_four_query_model_upper_bound=False))
        save(out/'02_output_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    except BaseException:
        save(out/'FAILED.json',dict(state='error_no_method_conclusion',stage=stage,counts=COUNTS,traceback=traceback.format_exc(),seconds=time.monotonic()-started)); raise

if __name__=='__main__': main()
