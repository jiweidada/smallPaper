"""274: fixed263 windows, real cached pre/aux candidates, and full300 Geo0.
Written only for root review/execution. No detector/checkpoint/new forward/grad.
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='2'
sys.dont_write_bytecode=True
import argparse, gc, hashlib, importlib.util, pathlib, time, traceback, types
ROOT=pathlib.Path(__file__).resolve().parent
PROTOCOL=ROOT/'274_固定窗真实候选回收与Geo0分解_事前协议.txt'
REUSE=ROOT/'262_BASE24_residual_window_probe.py'
REUSE_SHA='ba695c5627e1120ed4bf2f7bd9bef274c94b39ccf8ba1d183e185c3e277ee065'
if hashlib.sha256(REUSE.read_bytes()).hexdigest()!=REUSE_SHA: raise RuntimeError('262 changed')
spec=importlib.util.spec_from_file_location('_probe275_reuse262',REUSE)
R=importlib.util.module_from_spec(spec); spec.loader.exec_module(R)  # Definitions only; never R.main.
V,P,H=R.V,R.P,R.H; np,torch=R.np,R.torch; need,sha,save=R.need,R.sha,R.save
COUNTS=R.COUNTS
COUNTS['candidate_fusion_postprocess_calls']=0
OLD=ROOT/'263_BASE24_residual_window_probe'
OLD_HASHES={
 '00_manifest_and_synthetic.json':'fc6cd224c14c4f9f9e35cb195b9f41240d723cb15a0039e82417a64151c0f7c8',
 '01_results.json':'56b7bb86d3f9927bd501df48a2df9be7aab82a0157942b5da47c2513797530b3',
 '02_output_sha256.json':'8e200dee1ffe99616aa0207e5b80222f2370a9904c529c37a7c5f42f9535d934'}
WINDOWS=('A_density','B_residual')
RULES=('take4','take4thenfilter','take8filterthenkeep4')
STAGES=('pre','aux_0','aux_1')
GROUPS=('all','tiny','Geo0','tiny_Geo0')

def listids(mask): return np.flatnonzero(mask).tolist()

def stage_winners(weights,legal):
    need(weights.shape==legal.shape and weights.shape[0]==3,'three-stage winner shape')
    safe=np.where(legal,weights,-np.inf)
    indices=np.argmax(safe,axis=0)  # First stage wins an exact tie.
    return indices,legal.any(0)

def rank_window(records,cell):
    return sorted((r for r in records if r['cell']==cell),key=lambda r:(-r['residual'],r['stage_index'],r['q']))[:8]

def rule_indices(top):
    return dict(take4=list(range(min(4,len(top)))),
        take4thenfilter=[j for j in range(min(4,len(top))) if top[j]['keep']],
        take8filterthenkeep4=[j for j in range(len(top)) if top[j]['keep']][:4])

def candidate_pool(output,index,post,hw,box,tv):
    h,w=hw; stages=[output['pre_outputs'],*output['aux_outputs']]
    need(len(stages)==3,'only pre/aux0/aux1, excluding encoder')
    payload=[]; weights=[]; valid=[]
    finalxy=R.pixels(output['pred_boxes'][index],hw,tv)
    for s,stage in enumerate(stages):
        bb,zz=stage['pred_boxes'][index],stage['pred_logits'][index]
        need(bb.shape==(300,4) and zz.shape==(300,10) and bb.dtype==zz.dtype==torch.float32
             and not bb.requires_grad and not zz.requires_grad and bool(torch.isfinite(bb).all())
             and bool(torch.isfinite(zz).all()),'original selected300 frozen stage')
        xy=R.pixels(bb,hw,tv); legal=(xy[:,2:]>xy[:,:2]).all(-1).numpy()
        center=(bb[:,:2]*bb.new_tensor([w,h])).numpy().astype(np.float64)
        overlaps=R.pair_iou(xy,post['boxes'],box)
        best=overlaps.max(-1) if overlaps.shape[1] else np.zeros(300,dtype=np.float32)
        conf=zz.sigmoid().max(-1).values.numpy().astype(np.float64)
        weight=conf*(1-np.clip(best.astype(np.float64),0,1))
        weights.append(weight); valid.append(legal)
        payload.append(dict(boxes=bb,logits=zz,xy=xy,cells=R.cell_ids(center,hw),conf=conf,
                            bestP=best,bestFinal=R.pair_iou(xy,finalxy,box).max(-1)))
    weights,valid=np.stack(weights),np.stack(valid); selected,exists=stage_winners(weights,valid)
    result=[]
    for q in np.flatnonzero(exists):
        s=int(selected[q]); p=payload[s]
        result.append(dict(q=int(q),stage_index=s,stage=STAGES[s],cell=int(p['cells'][q]),
            residual=float(weights[s,q]),max_class_score=float(p['conf'][q]),maxIoU_P=float(p['bestP'][q]),
            maxIoU_final300=float(p['bestFinal'][q]),keep=bool(p['bestFinal'][q]<.5),
            box=p['boxes'][q],logits=p['logits'][q],xy=p['xy'][q]))
    return result,dict(source_records=900,valid_stage_records=int(valid.sum()),query_winners=len(result),
        no_valid_stage_queries=int((~exists).sum()),outside_center_winners=sum(r['cell']<0 for r in result),
        zero_residual_winners=sum(r['residual']==0 for r in result),stage_tie_order=list(STAGES),GT_used=False)

def group_masks(tiny,geo0):
    return dict(all=np.ones(len(tiny),dtype=bool),tiny=tiny,Geo0=geo0,tiny_Geo0=tiny&geo0)

def evaluate(post,gtxy,labels,tiny,geo0,box,baseline_ids=None):
    keep=R.perclass100(post['labels'],post['scores']); overlaps=R.pair_iou(post['boxes'][keep],gtxy,box)
    plabels=post['labels'][keep]; gp=R.greedy(overlaps,plabels,labels,.5)
    covered={int(g) for _,g in gp}; old=covered if baseline_ids is None else set(baseline_ids)
    edge=(overlaps>=.5)&(plabels[:,None]==labels[None,:])
    groups={}
    for name,mask in group_masks(tiny,geo0).items():
        ids=set(listids(mask)); new=covered&ids; previous=old&ids
        groups[name]=dict(TP=len(new),gained_GT=sorted(new-previous),lost_GT=sorted(previous-new),
            retained_GT=sorted(new&previous),gained=len(new-previous),lost=len(previous-new),retained=len(new&previous))
    bornP=post['q']>=300; born100=post['q'][keep]>=300
    return dict(P_count=len(post['flat']),P100_count=len(keep),TP=len(gp),FP=len(keep)-len(gp),
        covered_GT=sorted(covered),greedy_pairs=gp,groups=groups,
        capacity=dict(all=len(R.match(edge)),tiny_independent=len(R.match(edge[:,tiny]))),
        new_candidates_in_P=int(bornP.sum()),new_queries_in_P=len(set(post['q'][bornP].tolist())),
        new_candidates_in_P100=int(born100.sum()),new_queries_in_P100=len(set(post['q'][keep][born100].tolist())),
        P_flat=post['flat'].tolist(),P100_indices=keep.tolist())

def assert_old_parity(old,post,base,full_base,keep):
    saved=old['post']
    need(saved['P_flat']==post['flat'].tolist() and saved['P_scores']==post['scores'].tolist()
         and saved['P_boxes_xyxy']==post['boxes'].tolist() and saved['P100_indices']==keep.tolist()
         and saved['natural_top300']==post['natural_top300'] and saved['threshold_kept']==post['threshold_kept'],
         '263 exact original postprocess replay')
    for t in ('.5',):  # Both native262 thresholds were computed; primary comparison below is .5.
        key=str(float(t)); previous=old['baseline'][key]; fresh=full_base[key]
        for field in ('max_all','max_tiny_independent','unmatched_GT','degree','degree0_GT',
                      'unmatched_with_edges','greedy100_all','greedy100_tiny_among_all'):
            need(previous[field]==fresh[field],'263 baseline parity '+field)
        for field in ('pairs','tiny_pairs','greedy_pairs'):
            need(previous[field]==[list(p) for p in fresh[field]],'263 original pair parity '+field)
    need(base['TP']==old['baseline']['0.5']['greedy100_all']
         and base['capacity']['all']==old['baseline']['0.5']['max_all'],'263 evaluator parity')

def wrong_conditions(post,gtxy,labels,tiny,geo0,full,window,box):
    xy=post['boxes'].numpy(); overlaps=R.pair_iou(post['boxes'],gtxy,box)
    legal=np.isfinite(xy).all(-1)&(xy[:,2:]>xy[:,:2]).all(-1)
    intersection=np.maximum(np.minimum(xy[:,2:],window[2:])-np.maximum(xy[:,:2],window[:2]),0).prod(-1)
    explained=((overlaps>=.5)&(post['labels'][:,None]==labels[None,:])).any(-1)
    best=overlaps.max(-1) if len(labels) else np.zeros(len(xy),dtype=np.float32)
    selected=legal&(intersection>0)&~explained; local=full&geo0
    masks=dict(wrong_class=selected&(best>=.5),partial_geometry=selected&(best>0)&(best<.5),
               no_annotated_overlap=selected&(best==0))
    reports={}
    for name,mask in dict(all_unexplained=selected,**masks).items():
        overlapGT=(overlaps[mask]>0).any(0)&local
        reports[name]=dict(instances=int(mask.sum()),unique_queries=len(set(post['q'][mask].tolist())),
            P_indices=listids(mask),coexists_full_Geo0=bool(mask.any() and local.any()),
            instances_coexisting_full_Geo0=int(mask.sum()) if local.any() else 0,
            overlap_Geo0_GT=listids(overlapGT),overlap_tiny_Geo0_GT=listids(overlapGT&tiny))
    return dict(categories=reports,invalid_P_boxes_total=int((~legal).sum()),
        full_Geo0=int(local.sum()),full_tiny_Geo0=int((local&tiny).sum()),
        definition='No same-class IoU>=.5 edge to any whole-image GT; not greedy-unmatched; no causal claim')

def image_probe(index,output,target,post,old,hw,cfg,forward,box,tv):
    h,w=hw; gt=target['boxes']; labels=target['labels'].numpy(); gtxy=R.pixels(gt,hw,tv)
    gtnp=gtxy.numpy().astype(np.float64)
    tiny=(gt.numpy().astype(np.float64)[:,2:]*np.asarray([w,h])).min(-1)<4
    gcells=R.cell_ids((gt[:,:2]*gt.new_tensor([w,h])).numpy().astype(np.float64),hw)
    finalxy=R.pixels(output['pred_boxes'][index],hw,tv)
    finaliou=R.pair_iou(finalxy,gtxy,box); geo0=finaliou.max(0)<.5
    keep=R.perclass100(post['labels'],post['scores'])
    full_base,_=R.baseline(R.pair_iou(post['boxes'][keep],gtxy,box),post['labels'][keep],labels,tiny)
    base=evaluate(post,gtxy,labels,tiny,geo0,box)
    need(old['image_index']==index and old['GT']==len(gt) and old['actual_sub4']==int(tiny.sum()),'263 target/cohort identity')
    assert_old_parity(old,post,base,full_base,keep)
    need(base['groups']['Geo0']['TP']==0,'geometric Geo0 cannot be originally trueclass covered')
    pool,pool_audit=candidate_pool(output,index,post,hw,box,tv); grid=R.make_grid(hw)
    need(grid.tolist()==old['grid_xyxy'],'263 exact64 grid')
    windows={}
    for name in WINDOWS:
        fixed=old['windows'][name]
        if fixed['NA']:
            windows[name]=dict(NA=True,cell=None,reason=fixed['reason']); continue
        cell=int(fixed['cell']); window=np.asarray(fixed['xyxy_pixels'],dtype=np.float64)
        need(np.array_equal(window,grid[cell]),'original selected window coordinates')
        full,center,fraction=R.visibility(window,gtnp,gcells,cell); local=full&geo0
        top=rank_window(pool,cell); chosen=rule_indices(top); arms={}; records=[]
        for j,r in enumerate(top):
            overlap=R.pair_iou(r['xy'][None],gtxy,box)[0]; hits=(overlap>=.5)&geo0
            scores=r['logits'].sigmoid().numpy()
            record={k:v for k,v in r.items() if k not in ('box','logits','xy')}
            record.update(rank=j,box_normalized_cxcywh=r['box'].tolist(),box_pixels_xyxy=r['xy'].tolist(),
                logits=r['logits'].tolist(),geometric_Geo0_hit_GT=listids(hits),
                geometric_tiny_Geo0_hit_GT=listids(hits&tiny),
                Geo0_hit_trueclass_score=[dict(GT=int(g),label=int(labels[g]),score=float(scores[labels[g]]),
                    IoU=float(overlap[g])) for g in np.flatnonzero(hits)])
            records.append(record)
        for rule,indices in chosen.items():
            boxes=[output['pred_boxes'][index]]+[top[j]['box'][None] for j in indices]
            logits=[output['pred_logits'][index]]+[top[j]['logits'][None] for j in indices]
            augmented=dict(pred_boxes=torch.cat(boxes,0)[None],pred_logits=torch.cat(logits,0)[None])
            fresh=R.post_selection(augmented,hw,cfg,forward)[0]; COUNTS['candidate_fusion_postprocess_calls']+=1
            report=evaluate(fresh,gtxy,labels,tiny,geo0,box,base['covered_GT'])
            if not indices:
                need(report['covered_GT']==base['covered_GT'] and report['P_flat']==base['P_flat'],
                     'empty candidate arm is exact original')
            report.update(appended=len(indices),appended_rank=indices,
                          appended_source=[dict(new_query=300+k,stage=top[j]['stage'],q=top[j]['q']) for k,j in enumerate(indices)])
            arms[rule]=report
        rejected={}
        for n in (4,8):
            rr=[r for r in records[:n] if not r['keep']]
            hits=sorted({g for r in rr for g in r['geometric_Geo0_hit_GT']})
            rejected[str(n)]=dict(candidates=len(rr),geometric_Geo0_hit_candidates=sum(bool(r['geometric_Geo0_hit_GT']) for r in rr),
                geometric_Geo0_GT=hits,geometric_tiny_Geo0_GT=[g for g in hits if tiny[g]])
        deleted=sorted((set(arms['take4']['covered_GT'])-set(arms['take4thenfilter']['covered_GT']))&set(listids(geo0)))
        windows[name]=dict(NA=False,cell=cell,xyxy_pixels=window.tolist(),old263_selection_score=fixed['selection_score'],
            fullvisible_Geo0_GT=listids(local),fullvisible_tiny_Geo0_GT=listids(local&tiny),
            ideal4_Geo0=min(4,int(local.sum())),ideal4_tiny_Geo0=min(4,int((local&tiny).sum())),
            center_visible_Geo0_GT=listids(center&geo0),ranked_top8=records,arms=arms,
            filter_rejected=rejected,actual_take4_Geo0_lost_after_filter_GT=deleted,
            actual_take4_tiny_Geo0_lost_after_filter_GT=[g for g in deleted if tiny[g]],
            wrong_conditions=wrong_conditions(post,gtxy,labels,tiny,geo0,full,window,box))
    return dict(image_index=index,GT=len(gt),actual_sub4=int(tiny.sum()),Geo0_GT=listids(geo0),
        tiny_Geo0_GT=listids(geo0&tiny),P100_trueclass_degree0_GT=full_base['0.5']['degree0_GT'],
        P100_max_unmatched_with_edges_GT=full_base['0.5']['unmatched_with_edges'],
        P100_max_unmatched_GT=full_base['0.5']['unmatched_GT'],baseline=base,pool_audit=pool_audit,windows=windows,
        original263_post_and_baseline_exact=True,GT_source='real augmented training pipeline development cache')

def synthetic(forward,tv,box):
    old=R.synthetic(forward,tv)
    weights=np.array([[1.,0.,2.],[1.,0.,3.],[.5,0.,1.]])
    selected,exists=stage_winners(weights,np.ones_like(weights,dtype=bool))
    need(selected.tolist()==[0,0,1] and exists.all(),'stage tie and zero contribution winner')
    cells=np.array([[1,0,0],[0,0,0],[0,0,0]])
    records=[dict(q=q,stage_index=int(selected[q]),cell=int(cells[selected[q],q]),
                  residual=float(weights[selected[q],q]),keep=False) for q in range(2)]
    need([r['q'] for r in rank_window(records,0)]==[1],'choose stage first, not alternative inside stage; zero retained')
    tied=[dict(q=q,stage_index=s,cell=0,residual=1.) for s,q in ((1,0),(0,2),(0,1))]
    need([(r['stage_index'],r['q']) for r in rank_window(tied,0)]==[(0,1),(0,2),(1,0)],'rank stage then query tie')
    top=[dict(q=j,stage_index=0,cell=0,residual=8-j,keep=j in (1,4,5,6,7)) for j in range(8)]
    ids=rule_indices(top)
    need(ids==dict(take4=[0,1,2,3],take4thenfilter=[1],take8filterthenkeep4=[1,4,5,6]),'no refill vs refill')
    cfg=types.SimpleNamespace(num_top_queries=300,num_classes=10,use_focal_loss=True,score_threshold=.001,
                             remap_mscoco_category=False,deploy_mode=False)
    zz=torch.full((1,301,10),-20.); zz[0,300]=torch.arange(10,dtype=torch.float32)+2
    bb=torch.tensor([.5,.5,.1,.1]).repeat(1,301,1)
    post=R.post_selection(dict(pred_boxes=bb,pred_logits=zz),(640,640),cfg,forward)[0]
    COUNTS['synthetic_native_postprocess_calls']+=1
    need(post['q'].tolist()==[300]*10 and set(post['labels'].tolist())==set(range(10)),
         'appended query retains all ten logits and native flatten budget')
    gt=torch.tensor([[0.,0.,1.,1.]]); final=torch.tensor([[.6,0.,1.6,1.]]); candidate=torch.tensor([[.3,0.,1.3,1.]])
    need(R.pair_iou(final,gt,box)[0,0]<.5 and R.pair_iou(candidate,gt,box)[0,0]>=.5
         and R.pair_iou(candidate,final,box)[0,0]>=.5,'IoU filtering may delete a Geo0 hit')
    allfalse=np.zeros((3,1),dtype=bool)
    need(not stage_winners(np.zeros((3,1)),allfalse)[1][0],'invalid stages excluded')
    return dict(reused262=old,stage_ties=True,stage_before_window=True,zero_contribution=True,
                rule_order=True,full_logits_append=True,IoU_nontransitivity=True,invalid_stages=True)

def summarize(batches):
    result={}
    for letter in ('A','B'):
        images=[im for b in batches if b['letter']==letter for im in b['images']]; windows={}
        for name in WINDOWS:
            valid=[im['windows'][name] for im in images if not im['windows'][name]['NA']]; arms={}
            for rule in RULES:
                reports=[w['arms'][rule] for w in valid]
                arms[rule]={k:sum(r[k] for r in reports) for k in ('appended','P_count','P100_count','TP','FP',
                    'new_candidates_in_P','new_queries_in_P','new_candidates_in_P100','new_queries_in_P100')}
                arms[rule]['groups']={g:{k:sum(r['groups'][g][k] for r in reports) for k in ('TP','gained','lost','retained')} for g in GROUPS}
                arms[rule]['capacity']={k:sum(r['capacity'][k] for r in reports) for k in ('all','tiny_independent')}
            windows[name]=dict(selected=len(valid),NA=len(images)-len(valid),
                fullvisible_Geo0=sum(len(w['fullvisible_Geo0_GT']) for w in valid),
                fullvisible_tiny_Geo0=sum(len(w['fullvisible_tiny_Geo0_GT']) for w in valid),
                sum_min4_fullvisible_Geo0=sum(w['ideal4_Geo0'] for w in valid),
                sum_min4_fullvisible_tiny_Geo0=sum(w['ideal4_tiny_Geo0'] for w in valid),arms=arms,
                filter_rejected={n:{key:sum(len(w['filter_rejected'][n][key]) if isinstance(w['filter_rejected'][n][key],list)
                    else w['filter_rejected'][n][key] for w in valid) for key in ('candidates','geometric_Geo0_hit_candidates',
                    'geometric_Geo0_GT','geometric_tiny_Geo0_GT')} for n in ('4','8')},
                actual_take4_Geo0_lost_after_filter=sum(len(w['actual_take4_Geo0_lost_after_filter_GT']) for w in valid),
                actual_take4_tiny_Geo0_lost_after_filter=sum(len(w['actual_take4_tiny_Geo0_lost_after_filter_GT']) for w in valid),
                wrong_conditions={cat:{key:sum(len(w['wrong_conditions']['categories'][cat][key])
                    if isinstance(w['wrong_conditions']['categories'][cat][key],list)
                    else w['wrong_conditions']['categories'][cat][key] for w in valid)
                    for key in ('instances','unique_queries','coexists_full_Geo0','instances_coexisting_full_Geo0',
                                'overlap_Geo0_GT','overlap_tiny_Geo0_GT')}
                    for cat in ('all_unexplained','wrong_class','partial_geometry','no_annotated_overlap')})
        result[letter]=dict(images=len(images),GT=sum(im['GT'] for im in images),actual_sub4=sum(im['actual_sub4'] for im in images),
            Geo0=sum(len(im['Geo0_GT']) for im in images),tiny_Geo0=sum(len(im['tiny_Geo0_GT']) for im in images),
            baseline={k:sum(im['baseline'][k] for im in images) for k in ('TP','FP','P_count','P100_count')},windows=windows)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--synthetic-only',action='store_true')
    parser.add_argument('--out',type=pathlib.Path,default=ROOT/'276_BASE24_residual_candidate_replay'); args=parser.parse_args()
    started=time.monotonic(); resources=H.resource_limits()
    need(resources['priority']=='below_normal' and torch.version.cuda is None,'CPU resource contract')
    for key,path in (('postprocessor',R.POST),('evaluator',P.A.EVAL_SOURCE),('box_ops',P.A.NATIVE/'box_ops.py'),('dfine_utils',P.A.NATIVE/'dfine_utils.py')):
        need(sha(path)==P.A.EXPECTED[key],'native source before synthetic: '+key)
    native=H.load_native_math(); box=sys.modules[native.__package__+'.box_ops']; defaults,forward,tv=R.load_post_api()
    tests=dict(replay=synthetic(forward,tv,box),GO_compatibility=V.compatibility_self_checks())
    if args.synthetic_only:
        print(H.json.dumps(dict(tests=tests,counts=COUNTS),ensure_ascii=False)); return
    out=args.out.resolve(); need(out.parent==ROOT and out.name.startswith('276_') and not out.exists(),'new276 only'); out.mkdir()
    stage='source_hashes'; before={}; batches=[]; artifacts={}
    try:
        files,expected=V.manifest()
        files.update(reuse237=files['script'],protocol236=files['protocol'],reuse262=REUSE,
            script=pathlib.Path(__file__),protocol=PROTOCOL,postprocessor=R.POST,evaluator_source=P.A.EVAL_SOURCE)
        expected.update(reuse237=R.REUSE_SHA,reuse262=REUSE_SHA,postprocessor=P.A.EXPECTED['postprocessor'],evaluator_source=P.A.EXPECTED['evaluator'])
        for name,digest in OLD_HASHES.items():
            need(sha(OLD/name)==digest,'263 frozen '+name); files['263_'+name]=OLD/name; expected['263_'+name]=digest
        old_outputs=P.read_json(OLD/'02_output_sha256.json')
        old_results=P.read_json(OLD/'01_results.json'); need(old_results['state']=='complete','263 complete')
        for letter in ('A','B'):
            for batch in range(8):
                name=f'{letter}_batch{batch:02d}.json'; key='263_'+name
                need(old_outputs[name]==old_results['artifact_sha256'][name],'263 artifact manifest consistency')
                files[key]=OLD/name; expected[key]=old_outputs[name]
        before={k:sha(p) for k,p in files.items()}; need(all(before[k]==v for k,v in expected.items()),'fixed sources/inputs')
        save(out/'00_manifest_and_synthetic.json',dict(paths={k:str(p.resolve()) for k,p in files.items()},sha256=before,tests=tests,resources=resources))
        recipes=P.read_json(files['recipe']); shapes=P.input_shapes(); audit=V.load_union_audit()
        for letter,seed in (('A',1),('B',2)):
            predrows,analysisrows,provenance=P.metadata(letter,seed); recipe=recipes[str(seed)]; cfg=R.post_config(recipe,defaults)
            compute,union=P.native_readers(box,recipe)
            for batch in range(8):
                P.M.guard(); stage=f'{letter}_batch{batch:02d}'; hw=tuple(shapes[batch]['shape'][2:])
                need(predrows[batch]['input_sha256']==shapes[batch]['sha256'],'actual input shape seal')
                pp=P.DATA/letter/'frozen_predictions/development'/predrows[batch]['relative']
                ap=P.DATA/letter/'development'/analysisrows[batch]['relative']
                need(sha(pp)==predrows[batch]['sha256'] and sha(ap)==analysisrows[batch]['sha256'],'batch original hashes')
                prior=P.read_json(ap); value=torch.load(pp,map_location='cpu',weights_only=False); COUNTS['prediction_cache_loads']+=1
                output,targets,original,costs,denom=V.validate_batch(value,prior,batch,seed,predrows[batch],recipe,compute,union,audit)
                COUNTS['native_matcher_calls']+=1
                posts=R.post_selection(output,hw,cfg,forward); COUNTS['native_postprocess_calls']+=1
                old=P.read_json(OLD/(stage+'.json'))
                need(old['prediction_sha256']==predrows[batch]['sha256'] and old['input_sha256']==predrows[batch]['input_sha256']
                     and old['shape']==shapes[batch]['shape'],'263 exact prediction/input/shape')
                images=[image_probe(i,output,t,posts[i],old['images'][i],hw,cfg,forward,box,tv) for i,t in enumerate(targets)]
                report=dict(letter=letter,seed=seed,batch=batch,shape=shapes[batch]['shape'],input_sha256=predrows[batch]['input_sha256'],
                    prediction_sha256=predrows[batch]['sha256'],source_ids=predrows[batch]['source_ids'],provenance=provenance,
                    denominator_and_identity_audit=denom,old263_windows_reselected=False,images=images)
                filename=stage+'.json'; save(out/filename,report); artifacts[filename]=sha(out/filename); batches.append(report)
                del value,output,targets,original,costs,posts; gc.collect(); print(stage,'complete',flush=True)
        need(COUNTS['prediction_cache_loads']==COUNTS['native_matcher_calls']==COUNTS['native_postprocess_calls']==16,'exact16 cache budget')
        validwindows=sum(not im['windows'][w]['NA'] for b in batches for im in b['images'] for w in WINDOWS)
        need(COUNTS['candidate_fusion_postprocess_calls']==3*validwindows<=384,'three fusion arms per fixed nonNA window')
        unchanged={k:sha(p)==before[k] for k,p in files.items()}; need(all(unchanged.values()),'sources unchanged')
        need(all(COUNTS[k]==0 for k in ('model_builds','checkpoint_loads','model_forwards','criterion_forwards','autograd_calls',
             'model_backwards','model_vjps','optimizer_steps','heldout_reads')),'zero model/gradient budget')
        save(out/'01_results.json',dict(state='complete',summaries=summarize(batches),counts=COUNTS,resources=resources,
            sources_unchanged=unchanged,artifact_sha256=artifacts,seconds=time.monotonic()-started,
            old263_necessary_screen_both=old_results['necessary_screen_both'],old263_gate_changed=False,
            new_method_gate=None,AP=False,learned_completion_head_tested=False,conditional_input_needed_proven=False,
            actual_cached_candidate_refusion_tested=True,training_authorized=False,
            scope='Fixed windows; actual pre/aux candidate replay; full300 geometry decomposition; no causal learning inference'))
        save(out/'02_output_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    except BaseException:
        save(out/'FAILED.json',dict(state='error_no_method_conclusion',stage=stage,counts=COUNTS,
            traceback=traceback.format_exc(),seconds=time.monotonic()-started)); raise

if __name__=='__main__': main()
