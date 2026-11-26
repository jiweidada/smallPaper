"""Conservative description of native all-area false positives using saved final boxes and scores.

Never modifies native matches or assumes score rank identifies a unique query under ties.
"""
from pathlib import Path
import argparse, gzip, hashlib, importlib.util, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
LABELS=('native_TP','invalid_or_zero_size_box',
        'same_class_eligible_GT_all_already_native_matched_earlier',
        'same_class_overlap_but_native_match_geometry_unresolved',
        'other_class_overlap_at_threshold',
        'overlap0_1_to_threshold_any_class',
        'overlap_below0_1_any_class',
        'IoU_boundary_uncertain',
        'score_to_box_ambiguous', 'native_ignored')
PREFIXES=(100,500,1000,2000,5000,10000,20000)

def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def fp_label(ious, categories, gt_ids, predicted_class, occupied, threshold, epsilon, valid=True):
    if not valid:return 1
    own=categories==predicted_class;ov=ious[own]
    if np.any(np.abs(ov-threshold)<=epsilon):return 7
    eligible=gt_ids[own][ov>threshold+epsilon]
    if len(eligible):
        return 2 if all(int(g) in occupied for g in eligible) else 3
    other=ious[~own];best_other=float(other.max()) if len(other) else 0.
    if best_other>threshold+epsilon:return 4
    if abs(best_other-threshold)<=epsilon:return 7
    best=float(ious.max()) if len(ious) else 0.
    if abs(best-.1)<=epsilon:return 7
    return 5 if best>.1+epsilon else 6

def source(seed,tag,tools,protocol,readers):
    base=tools.cached_baseline(seed,protocol,readers['full'])
    if tag=='BASE12':
        packet=base['packet'];folder=packet/'baseline_eval';proof=base['source_report_sha256']
        rank_folder=ROOT/protocol['baseline_caches'][str(seed)]['rank']
    else:
        assert seed==2 and tag in ('backbone_silu','p2_skip')
        vpath=ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed{seed}_{tag}_v1/primary_verification_v1.json'
        v=read(vpath);assert v['status']=='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
        assert v['all_collected_files_verified'] and v['actual_stage_images']==548 and v['files_verified']==617
        assert v['verification_script_sha256']==sha(ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2/verify_primary.py')
        assert v['identity']['seed']==seed and v['identity']['arm']==tag
        packet=Path(v['downloaded_packet']);assert sha(packet/'primary_manifest.json')==v['primary_manifest_sha256']
        folder=packet/'run';proof=sha(vpath)
        rank_folder=ROOT/f'outputs/user_backbone_primary_analysis12_20261011_v2/seed{seed}_{tag}/candidate_ranking'
    for n,h in read(rank_folder/'manifest.json')['files'].items():assert sha(rank_folder/n)==h
    rank=read(rank_folder/'report.json');checks=read(rank_folder/'image_score_checks.json')
    summary,flags,_=readers['control'].native_endpoint(folder/'endpoint_epoch12',base['annotation'])
    assert rank['identity']==summary['identity'] and not rank['score_mismatch_images'] and rank['native_TP_proxy_conflicts']==0
    readers['control'].stages(folder,summary['identity'],base['annotation'])
    for k in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):
        assert summary['identity'][k]==base['summary']['identity'][k]
    return base,folder,summary,flags,checks,dict(source_verification_sha256=proof,rank_manifest_sha256=sha(rank_folder/'manifest.json'),
        endpoint_manifest_sha256=sha(folder/'endpoint_epoch12/manifest.json'),stage_manifest_sha256=sha(folder/'stages_epoch12/manifest.json'))

def analyze(seed,tag):
    spec=read(HERE/'protocol.json')
    assert seed==2 and tag in ('backbone_silu','p2_skip') and sha(__file__)==spec['analysis_sha256']
    for n,h in spec['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    tools=module('native_fp_cached_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
    protocol,geometry,_,_,readers=tools.prepared()
    base,folder,summary,flags,checks,proof=source(seed,tag,tools,protocol,readers)
    annotation=base['annotation'];images={int(i['id']):i for i in annotation['images']};gt_by_image={i:[] for i in images}
    for a in annotation['annotations']:gt_by_image[int(a['image_id'])].append(a)
    with gzip.open(folder/'endpoint_epoch12/matching_all_area.json.gz','rt',encoding='utf8') as f:matching=json.load(f)
    by_cell={(int(r['image_id']),int(r['category_id'])):r for r in matching};assert len(by_cell)==len(matching)
    stage_manifest=read(folder/'stages_epoch12/manifest.json')
    rows=[];geometry_native_TP_conflicts=[];possible_query_count=[]
    score_tolerance=spec['score_abs_tolerance'];epsilon=spec['IoU_boundary_epsilon']
    for iid,im in sorted(images.items()):
        path=folder/f'stages_epoch12/image_{iid:06d}.npz';assert sha(path)==stage_manifest['files'][path.name]
        with np.load(path,allow_pickle=False) as z:boxes=z['final_boxes'].copy();logits=z['final_logits'].astype(np.float64);hw=z['original_hw'].copy()
        assert boxes.shape==(300,4) and logits.shape==(300,10) and hw.tolist()==[im['height'],im['width']]
        probability=(1/(1+np.exp(-logits))).astype(np.float32).astype(np.float64)
        entries=sorted(gt_by_image[iid],key=lambda a:int(a['id']))
        ids=np.array([a['id'] for a in entries],np.int64);cats=np.array([a['category_id'] for a in entries],np.int64)
        gt=np.array([a['bbox'] for a in entries],np.float64).reshape(-1,4)
        valid=(boxes[:,2:]>0).all(1);ious=np.zeros((len(entries),300))
        ious[:,valid]=geometry.iou_matrix(gt,geometry.pixel_xywh(boxes[valid],hw))
        positions={int(g):k for k,g in enumerate(ids)}
        for cid in range(10):
            native=by_cell.get((iid,cid))
            if native is None:
                assert not (cats==cid).any();continue
            assert native['aRng']==[0,1e10] and native['maxDet']==100
            assert set(native['gtIds'])==set(ids[cats==cid].tolist()) and not any(native['gtIgnore'])
            scores=np.asarray(native['dtScores'],np.float64)
            matches=np.asarray(native['dtMatches'],np.int64).reshape(10,-1);ignore=np.asarray(native['dtIgnore'],bool).reshape(10,-1)
            assert scores.shape==(matches.shape[1],) and not ignore.any()
            assert np.all(np.diff(scores)<=0) and len(scores)<=100
            candidates=[np.flatnonzero(np.abs(probability[:,cid]-s)<=score_tolerance) for s in scores]
            assert all(len(q)>0 for q in candidates),'Saved native score has no conservatively admissible query'
            native_score_error=checks[str(iid)]['native_score_list_max_abs_error'];assert native_score_error<=score_tolerance
            codes=np.empty(matches.shape,np.int8)
            for ti,threshold in enumerate(summary['protocol']['iou_thresholds']):
                occupied=set()
                for di,q in enumerate(candidates):
                    matched=int(matches[ti,di])
                    if matched:
                        codes[ti,di]=0
                        assert matched in positions and cats[positions[matched]]==cid and matched not in occupied
                        if ious[positions[matched],q].max()<threshold-epsilon:
                            geometry_native_TP_conflicts.append(dict(image=iid,category=cid,IoU=threshold,native_DT_local_order=di,matched_GT=matched))
                        occupied.add(matched)
                    else:
                        possible={fp_label(ious[:,qi],cats,ids,cid,occupied,threshold,epsilon,bool(valid[qi])) for qi in q}
                        codes[ti,di]=next(iter(possible)) if len(possible)==1 else 8
            for di,s in enumerate(scores):
                rows.append(dict(image=iid,category=cid,score=float(s),local_order=di,codes=codes[:,di].copy(),
                                 possible_query_count=len(candidates[di])))
    assert not geometry_native_TP_conflicts,'Native matched TP cannot be supported by any admissible saved box'
    classes=[];array_records={}
    for cid in range(10):
        rr=sorted((r for r in rows if r['category']==cid),key=lambda r:(r['image'],r['local_order']))
        scores=np.array([r['score'] for r in rr]);order=np.argsort(-scores,kind='stable')
        codes=np.stack([r['codes'] for r in rr],axis=1)[:,order]
        counts=[len(x['dtScores']) for x in matching if int(x['category_id'])==cid];assert codes.shape==(10,sum(counts))
        by_iou=[]
        for ti,t in enumerate(summary['protocol']['iou_thresholds']):
            entries=[]
            for n in sorted(set(PREFIXES)|{codes.shape[1]}):
                if n>codes.shape[1]:continue
                values=np.bincount(codes[ti,:n],minlength=len(LABELS));assert values.sum()==n
                entries.append(dict(rank=n,counts=dict(zip(LABELS,values.tolist())),TP=int(values[0]),FP=int(values[1:9].sum())))
            by_iou.append(dict(IoU=float(t),prefixes=entries))
        classes.append(dict(class_id=cid,name=summary['categories_percent'][cid]['name'],predictions=codes.shape[1],by_IoU=by_iou))
        array_records[f'class{cid}_codes']=codes
        array_records[f'class{cid}_native_scores']=scores[order]
        array_records[f'class{cid}_image_id']=np.array([r['image'] for r in rr],np.int64)[order]
        array_records[f'class{cid}_native_DT_local_order']=np.array([r['local_order'] for r in rr],np.int64)[order]
        array_records[f'class{cid}_possible_query_count']=np.array([r['possible_query_count'] for r in rr],np.int64)[order]
    observed_TP=np.array([sum(c['by_IoU'][ti]['prefixes'][-1]['TP'] for c in classes) for ti in range(10)])
    observed_FP=np.array([sum(c['by_IoU'][ti]['prefixes'][-1]['FP'] for c in classes) for ti in range(10)])
    assert observed_TP.tolist()==summary['TP_by_IoU']==flags['hits'].sum(axis=0).tolist()
    assert observed_FP.tolist()==summary['FP_by_IoU']
    dest=ROOT/f'outputs/native_fp_geometry12_20261011_v2/seed{seed}_{tag}';dest.parent.mkdir(exist_ok=True);dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'native_FP_categories.npz',**array_records)
    result=dict(status='NATIVE_FP_COUNTS_EXACT_GEOMETRY_LABELS_CONSERVATIVE_OUTPUT_DESCRIPTION',seed=seed,tag=tag,
        identity=summary['identity'],classes=classes,labels=list(LABELS),images=548,
        native_TP_counts_exact=True,native_FP_counts_exact=True,native_stable_global_and_local_score_ties_preserved=True,
        score_to_box_unique_identity_assumed=False,all_possible_close_score_queries_considered=True,
        all_native_TP_geometry_supported_by_at_least_one_admissible_query=True,
        detections=len(rows),detections_with_multiple_admissible_queries=sum(r['possible_query_count']>1 for r in rows),
        max_admissible_query_count=max(r['possible_query_count'] for r in rows),
        native_TP=observed_TP.tolist(),native_FP=observed_FP.tolist(),proof=proof,
        analysis_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),
        new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_native_matching_calls=0,
        new_optimizer_updates=0,new_training_epochs=0,root_proven=False,novelty_proven=False,
        boundaries='All-area native matches and native score order only. FP categories are conservative overlap descriptions, not exact cause labels or APs explanations. A near-score query set includes all300 queries, including possible cut-off ties; category is assigned only if every admissible box yields that category. Category2 describes all eligible same-class GT already occupied in preceding native matches, not a proven NMS/duplicate module defect. No cross-run query identity, raw native small-area prefix matching, missed annotations, background truth or training-assignment cause is established.')
    write(dest/'report.json',result)
    write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    people=classes[1]['by_IoU'][0]['prefixes']
    print(json.dumps(dict(status=result['status'],seed=seed,tag=tag,detections=len(rows),multiple_possible_queries=result['detections_with_multiple_admissible_queries'],people50_prefix1000=next(x for x in people if x['rank']==1000))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--tag',choices=('backbone_silu','p2_skip'),required=True)
    args=p.parse_args();analyze(args.seed,args.tag)
