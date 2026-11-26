"""Trace native FP overlap patterns along within-run queries; never align queries across runs."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
STAGE_LABELS=('not_occupied_same_class_FP','selected_regressed_encoder_reference','first_layer_pre','final_only','score_or_stage_boundary_uncertain')

def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def first_support(three_stage_IoU,threshold,epsilon):
    for index,v in enumerate(three_stage_IoU):
        if np.any(v>threshold+epsilon):return index+1
        if np.any(np.abs(v-threshold)<=epsilon):return 4
    return 4

def analyze(seed,tag):
    spec=read(HERE/'protocol.json');assert sha(__file__)==spec['analysis_sha256']
    assert seed==2 and tag in ('backbone_silu','p2_skip')
    for p,h in spec['dependencies_sha256'].items():assert sha(ROOT/p)==h,p
    upstream=module('native_FP_trace_upstream',ROOT/'英文小论文/code/native_fp_geometry12_20261011_v2/analyze.py')
    prior=ROOT/f'outputs/native_fp_geometry12_20261011_v2/seed{seed}_{tag}'
    for n,h in read(prior/'manifest.json')['files'].items():assert sha(prior/n)==h,n
    r=read(prior/'report.json');assert r['analysis_sha256']==sha(Path(upstream.__file__))
    assert r['native_TP_counts_exact'] and r['native_FP_counts_exact'] and r['all_native_TP_geometry_supported_by_at_least_one_admissible_query']
    with np.load(prior/'native_FP_categories.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    tools=module('query_trace_cached_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
    protocol,geometry,_,_,readers=tools.prepared()
    base,folder,summary,flags,checks,proof=upstream.source(seed,tag,tools,protocol,readers)
    assert proof==r['proof'] and summary['identity']==r['identity']
    annotation=base['annotation'];grouped={int(i['id']):[] for i in annotation['images']}
    for a in annotation['annotations']:grouped[int(a['image_id'])].append(a)
    with gzip.open(folder/'endpoint_epoch12/matching_all_area.json.gz','rt',encoding='utf8') as f:matching=json.load(f)
    native={(int(x['image_id']),int(x['category_id'])):x for x in matching};assert len(native)==len(matching)
    loc={}
    arrays={}
    for cid in range(10):
        images=old[f'class{cid}_image_id'];dt=old[f'class{cid}_native_DT_local_order']
        loc[cid]={(int(i),int(j)):k for k,(i,j) in enumerate(zip(images,dt))};assert len(loc[cid])==len(images)
        count=len(images)
        arrays[f'class{cid}_first_FP_target_support_stage']=np.zeros((10,count),np.int8)
        arrays[f'class{cid}_FP_and_winner_both_initial_same_GT']=np.full((10,count),-1,np.int8)
        arrays[f'class{cid}_FP_same_query_other_category_native_TP']=np.full((10,count),-1,np.int8)
        arrays[f'class{cid}_known_query_index']=np.full(count,-1,np.int64)
        arrays[f'class{cid}_emitted_categories_same_known_query']=np.full(count,-1,np.int16)
    epsilon=spec['IoU_boundary_epsilon'];tol=spec['score_abs_tolerance'];multi_label_images=[]
    for iid,entries in sorted(grouped.items()):
        with np.load(folder/f'stages_epoch12/image_{iid:06d}.npz',allow_pickle=False) as z:
            stages=[z[k].copy() for k in geometry.STAGES];logits=z['final_logits'].astype(np.float64);hw=z['original_hw'].copy()
        gt=np.array([a['bbox'] for a in entries],np.float64).reshape(-1,4);ids=np.array([a['id'] for a in entries],np.int64);cats=np.array([a['category_id'] for a in entries],np.int64)
        matrices=[]
        for boxes in stages:
            valid=(boxes[:,2:]>0).all(1);mat=np.zeros((len(entries),300))
            mat[:,valid]=geometry.iou_matrix(gt,geometry.pixel_xywh(boxes[valid],hw));matrices.append(mat)
        prob=(1/(1+np.exp(-logits))).astype(np.float32).astype(np.float64)
        cells={};q_labels={};tp_by_query=[{} for _ in range(10)];TP_query_by_GT=[{} for _ in range(10)]
        for cid in range(10):
            row=native.get((iid,cid))
            if row is None:continue
            scores=np.array(row['dtScores']);matches=np.asarray(row['dtMatches'],np.int64).reshape(10,-1)
            queries=[np.flatnonzero(np.abs(prob[:,cid]-score)<=tol) for score in scores]
            assert all(len(q)>0 for q in queries)
            cells[cid]=(row,matches,queries)
            for di,q in enumerate(queries):
                k=loc[cid][(iid,di)]
                assert len(q)==old[f'class{cid}_possible_query_count'][k]
                if len(q)!=1:continue
                qi=int(q[0]);arrays[f'class{cid}_known_query_index'][k]=qi
                q_labels.setdefault(qi,set()).add(cid)
                for ti in range(10):
                    gid=int(matches[ti,di])
                    if gid:
                        tp_by_query[ti].setdefault(qi,set()).add((cid,gid))
                        assert gid not in TP_query_by_GT[ti]
                        TP_query_by_GT[ti][gid]=qi
        multi_label_images.append(dict(image_id=iid,known_unique_queries=len(q_labels),known_queries_emitting_multiple_categories=sum(len(c)>1 for c in q_labels.values()),
            entries_with_unique_score_to_query=sum(len(q)==1 for _,_,queries in cells.values() for q in queries),
            query_identity_uncertain_entries=sum(len(q)>1 for _,_,queries in cells.values() for q in queries)))
        for cid,(row,matches,queries) in cells.items():
            for di,q in enumerate(queries):
                k=loc[cid][(iid,di)];code=old[f'class{cid}_codes'][:,k]
                if len(q)!=1:
                    arrays[f'class{cid}_first_FP_target_support_stage'][code==2,k]=4
                    continue
                qi=int(q[0]);arrays[f'class{cid}_emitted_categories_same_known_query'][k]=len(q_labels[qi])
                for ti,t in enumerate(summary['protocol']['iou_thresholds']):
                    if matches[ti,di]:continue
                    arrays[f'class{cid}_FP_same_query_other_category_native_TP'][ti,k]=int(any(c!=cid for c,g in tp_by_query[ti].get(qi,set())))
                    if code[ti]!=2:continue
                    eligible=(cats==cid)&(matrices[2][:,qi]>t+epsilon)
                    target_ids=ids[eligible]
                    assert len(target_ids)>0
                    arrays[f'class{cid}_first_FP_target_support_stage'][ti,k]=first_support([m[eligible,qi] for m in matrices],t,epsilon)
                    paired=[];unknown=False
                    for gi in np.flatnonzero(eligible):
                        gid=int(ids[gi])
                        if gid not in TP_query_by_GT[ti]:unknown=True;continue
                        winner=TP_query_by_GT[ti][gid];assert winner!=qi,'Same query cannot repeat its same-category class entry'
                        paired.append(bool(matrices[0][gi,qi]>t+epsilon and matrices[0][gi,winner]>t+epsilon))
                    arrays[f'class{cid}_FP_and_winner_both_initial_same_GT'][ti,k]=1 if any(paired) else -1 if unknown else 0
    classes=[]
    for cid in range(10):
        codes=old[f'class{cid}_codes'];first=arrays[f'class{cid}_first_FP_target_support_stage'];other=arrays[f'class{cid}_FP_same_query_other_category_native_TP'];pair=arrays[f'class{cid}_FP_and_winner_both_initial_same_GT']
        by_iou=[]
        for ti,t in enumerate(summary['protocol']['iou_thresholds']):
            prefix=[]
            for n in sorted(set(upstream.PREFIXES)|{codes.shape[1]}):
                if n>codes.shape[1]:continue
                c=codes[ti,:n];st=first[ti,:n];dups=c==2;cross=c==4
                counts={name:int(((st==index)&dups).sum()) for index,name in enumerate(STAGE_LABELS) if index>0}
                assert sum(counts.values())==int(dups.sum())
                prefix.append(dict(rank=n,occupied_same_class_FP=int(dups.sum()),first_FP_same_occupied_GT_support=counts,
                    both_FP_and_winner_query_initial_same_GT_supported=int(((pair[ti,:n]==1)&dups).sum()),
                    both_initial_pair_status_uncertain=int(((pair[ti,:n]<0)&dups).sum()),
                    other_class_overlap_FP=int(cross.sum()),
                    other_class_overlap_FP_also_other_category_TP_same_query=int(((other[ti,:n]==1)&cross).sum()),
                    other_class_overlap_FP_same_query_status_uncertain=int(((other[ti,:n]<0)&cross).sum()),
                    native_FP=int((c!=0).sum()),all_FP_with_other_category_TP_same_query=int(((other[ti,:n]==1)&(c!=0)).sum())))
            by_iou.append(dict(IoU=float(t),prefixes=prefix))
        classes.append(dict(class_id=cid,name=r['classes'][cid]['name'],by_IoU=by_iou))
    dest=ROOT/f'outputs/native_fp_query_trace12_20261011_v2/seed{seed}_{tag}';dest.parent.mkdir(exist_ok=True);dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'within_run_query_FP_trace.npz',**arrays)
    result=dict(status='NATIVE_FP_WITHIN_RUN_QUERY_AND_SAME_OCCUPIED_GT_STAGE_TRACE_DESCRIPTION',seed=seed,tag=tag,identity=r['identity'],
        classes=classes,multi_label_query_observations=multi_label_images,
        native_FP_geometry_manifest_sha256=sha(prior/'manifest.json'),analysis_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),
        three_boundaries_preserved_native_query_order=True,cross_run_query_identity_assumed=False,
        new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_native_matching_calls=0,new_optimizer_updates=0,new_training_epochs=0,
        root_proven=False,novelty_proven=False,
        scope='Per-query traces within each run only. Trace first observed geometric overlap with the same final occupied GT, not the first causal error. Selected300 initial references are already regressed; selection versus regression remains unresolved. Multiple close-score queries remain uncertain; initial FP/winner pairs use only uniquely identified queries. Same-query different-category TP is an output witness, not proof that disabling multilabel selection improves AP. Native all-area matching only, no exact APs-score-prefix explanation.')
    write(dest/'report.json',result)
    write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    assert 'torch' not in sys.modules and not any(x.startswith('pycocoeval') for x in sys.modules)
    print(json.dumps(dict(status=result['status'],seed=seed,tag=tag,people50_prefix1000=next(x for x in classes[1]['by_IoU'][0]['prefixes'] if x['rank']==1000))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--tag',choices=('backbone_silu','p2_skip'),required=True)
    args=p.parse_args();analyze(args.seed,args.tag)
