"""Within-query relative-class evidence conditioned on saved geometric support."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def relative_margin(logits,categories):
    target=logits[:,categories].T.copy();other=np.repeat(logits[None,:,:],len(categories),axis=0)
    other[np.arange(len(categories)),:,categories]=-np.inf
    return target-other.max(axis=2)
def describe(mask,a,ti,epsilon):
    support=a['support_count'][:,ti]>0;yes=a['max_relative_class_margin'][:,ti]>epsilon
    wrong=a['max_relative_class_margin'][:,ti]<-epsilon;native=a['native_TP'][:,ti]
    uncertain=support&~yes&~wrong
    counts=dict(GT=int(mask.sum()),no_final_geometric_support=int((mask&~support).sum()),supported_with_true_class_top1_exists=int((mask&support&yes).sum()),
        supported_all_queries_true_class_lower_than_other_class=int((mask&support&wrong).sum()),supported_relative_class_boundary_uncertain=int((mask&uncertain).sum()),
        native_TP=int((mask&native).sum()),native_TP_despite_all_supported_queries_wrong_top1=int((mask&native&wrong&support).sum()),
        supported_correct_top1_but_no_native_TP=int((mask&support&yes&~native).sum()))
    assert counts['no_final_geometric_support']+counts['supported_with_true_class_top1_exists']+counts['supported_all_queries_true_class_lower_than_other_class']+counts['supported_relative_class_boundary_uncertain']==counts['GT']
    selected=mask&support
    counts['max_target_logit_query_same_query_pre_to_final_margin']=dict(count=int(selected.sum()),
        pre_median=float(np.median(a['pre_margin_at_final_target_logit_query'][selected,ti])) if selected.any() else None,
        final_median=float(np.median(a['final_margin_at_final_target_logit_query'][selected,ti])) if selected.any() else None,
        delta_median=float(np.median((a['final_margin_at_final_target_logit_query']-a['pre_margin_at_final_target_logit_query'])[selected,ti])) if selected.any() else None)
    return counts
def main():
    spec=read(HERE/'protocol.json');assert spec['analysis_sha256']==sha(__file__)
    for n,h in spec['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    tools=module('relative_class_saved_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
    p,geo,_,_,readers=tools.prepared();base=tools.cached_baseline(2,p,readers['full']);annotation=base['annotation'];ids=base['values']['annotation_id'];cats=base['values']['category_id'];position={int(g):i for i,g in enumerate(ids)}
    grouped={int(im['id']):[] for im in annotation['images']}
    for a in annotation['annotations']:grouped[int(a['image_id'])].append(a)
    groups=tools.groups(annotation,base['values']);groups['COCO_small_input_under4']=groups['COCO_small']&groups['input_under4']
    for cid in range(10):groups[f'COCO_small_class{cid}']=groups['COCO_small']&(cats==cid)
    names={int(c['id']):c['name'] for c in annotation['categories']};runs={};details={};epsilon=spec['class_margin_epsilon']
    for tag in spec['run_tags']:
        if tag=='BASE12':
            folder=base['packet']/'baseline_eval';values=base['values'];proof=base['source_report_sha256'];identity=base['summary']['identity']
        else:
            vpath=ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed2_{tag}_v1/primary_verification_v1.json';v=read(vpath)
            assert v['status']=='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS' and v['all_collected_files_verified'] and v['actual_stage_images']==548
            values=npz(ROOT/f'outputs/user_backbone_primary_analysis12_20261011_v2/seed2_{tag}/candidate_stage_per_GT.npz');folder=Path(v['downloaded_packet'])/'run';proof=sha(vpath);identity=v['identity']
            assert identity['arm']==tag and identity['seed']==2
        for n in ('annotation_id','image_id','category_id','original_area'):assert np.array_equal(values[n],base['values'][n]),n
        readers['control'].stages(folder,identity,annotation)
        sm=read(folder/'stages_epoch12/manifest.json');a=dict(annotation_id=ids.copy(),category_id=cats.copy(),native_TP=values['native_TP'][:,[0,5]].copy())
        for n in ('support_count','same_query_final_target_logit_index'):a[n]=np.zeros((len(ids),2),np.int16)
        for n in ('max_relative_class_margin','pre_margin_at_final_target_logit_query','final_margin_at_final_target_logit_query','pre_IoU_at_final_target_logit_query'):a[n]=np.zeros((len(ids),2),np.float64)
        for iid,entries in sorted(grouped.items()):
            f=folder/f'stages_epoch12/image_{iid:06d}.npz';assert sha(f)==sm['files'][f.name]
            z=npz(f);logits=z['final_logits'].astype(np.float64);pre=z['pre_after_first_layer_logits'].astype(np.float64)
            gt=np.array([g['bbox'] for g in entries],np.float64).reshape(-1,4);category=np.array([g['category_id'] for g in entries]);ix=np.array([position[int(g['id'])] for g in entries])
            if not len(entries):continue
            final=z['final_boxes'];before=z['pre_after_first_layer_boxes'];valid=(final[:,2:]>0).all(1);pre_valid=(before[:,2:]>0).all(1)
            iou=np.zeros((len(entries),300));pre_iou=np.zeros_like(iou)
            iou[:,valid]=geo.iou_matrix(gt,geo.pixel_xywh(final[valid],z['original_hw']));pre_iou[:,pre_valid]=geo.iou_matrix(gt,geo.pixel_xywh(before[pre_valid],z['original_hw']))
            assert np.allclose(iou.max(1),values['max_iou'][ix,2],atol=1e-12,rtol=0)
            margin=relative_margin(logits,category);pre_margin=relative_margin(pre,category);target=logits[:,category].T
            for ti,t in enumerate(spec['IoU_thresholds']):
                supported=iou>=t;active=supported.any(1);a['support_count'][ix,ti]=supported.sum(1)
                best_margin=np.where(supported,margin,-np.inf).max(1);a['max_relative_class_margin'][ix[active],ti]=best_margin[active]
                q=np.where(supported,target,-np.inf).argmax(1);row=np.arange(len(entries));active_ix=ix[active]
                a['same_query_final_target_logit_index'][active_ix,ti]=q[active]
                a['pre_margin_at_final_target_logit_query'][active_ix,ti]=pre_margin[row[active],q[active]]
                a['final_margin_at_final_target_logit_query'][active_ix,ti]=margin[row[active],q[active]]
                a['pre_IoU_at_final_target_logit_query'][active_ix,ti]=pre_iou[row[active],q[active]]
        assert not np.any(a['native_TP']&(a['support_count']==0))
        assert np.array_equal(a['support_count']>0,values['max_iou'][:,2,None]>=np.array(spec['IoU_thresholds']))
        for n,x in a.items():assert np.isfinite(x).all(),n
        runs[tag]=a;details[tag]=dict(identity=identity,source_verification_sha256=proof,stage_manifest_sha256=sha(folder/'stages_epoch12/manifest.json'),groups={group:{str(t):describe(mask,a,ti,epsilon) for ti,t in enumerate(spec['IoU_thresholds'])} for group,mask in groups.items()})
    comparisons={}
    for tag in spec['run_tags'][1:]:
        B=runs['BASE12'];C=runs[tag];rows={}
        for group,mask in groups.items():
            rows[group]={}
            for ti,t in enumerate(spec['IoU_thresholds']):
                b=B['support_count'][:,ti]>0;c=C['support_count'][:,ti]>0
                bg=B['max_relative_class_margin'][:,ti]>epsilon;cg=C['max_relative_class_margin'][:,ti]>epsilon
                both=mask&b&c;gain=mask&~B['native_TP'][:,ti]&C['native_TP'][:,ti];loss=mask&B['native_TP'][:,ti]&~C['native_TP'][:,ti]
                rows[group][str(t)]=dict(GT=int(mask.sum()),native_TP_gained=int(gain.sum()),native_TP_lost=int(loss.sum()),both_runs_final_geometrically_supported_GT=int(both.sum()),
                    among_both_support_true_class_top1_exists_BASE=int((both&bg).sum()),among_both_support_true_class_top1_exists_candidate=int((both&cg).sum()),
                    newly_true_class_top1_supported_among_both=int((both&~bg&cg).sum()),lost_true_class_top1_supported_among_both=int((both&bg&~cg).sum()),
                    gained_TP_BASE_no_final_support=int((gain&~b).sum()),gained_TP_BASE_support_no_correct_top1=int((gain&b&~bg).sum()),gained_TP_BASE_support_correct_top1_exists=int((gain&b&bg).sum()),
                    lost_TP_candidate_no_final_support=int((loss&~c).sum()),lost_TP_candidate_support_no_correct_top1=int((loss&c&~cg).sum()),lost_TP_candidate_support_correct_top1_exists=int((loss&c&cg).sum()))
                r=rows[group][str(t)];assert r['gained_TP_BASE_no_final_support']+r['gained_TP_BASE_support_no_correct_top1']+r['gained_TP_BASE_support_correct_top1_exists']==r['native_TP_gained']
                assert r['lost_TP_candidate_no_final_support']+r['lost_TP_candidate_support_no_correct_top1']+r['lost_TP_candidate_support_correct_top1_exists']==r['native_TP_lost']
        comparisons[tag]=rows
    dest=ROOT/'outputs/user_class_margin_geometry12_20261011_v1';dest.mkdir(exist_ok=False)
    for tag,a in runs.items():np.savez_compressed(dest/(tag+'_per_GT.npz'),**a)
    result=dict(status='FOUR_SAVED_NATIVE_RUNS_GEOMETRY_CONDITIONED_RELATIVE_CLASS_EVIDENCE',seed=2,classes=names,run_tags=spec['run_tags'],runs=details,comparisons=comparisons,
        all_GT=38759,small_GT=26586,images_each=548,IoU_thresholds=spec['IoU_thresholds'],class_margin_epsilon=epsilon,
        source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_native_matching_calls=0,new_training_epochs=0,new_optimizer_updates=0,
        root_proven=False,novelty_proven=False,two_seed_efficacy_proven=False,
        scope='Passive relative-class evidence for all geometry-supported queries, joined across runs only byGT. Within-run pre/final margins use the same query chosen by maximum final target-class logit among final geometry supporters, not necessarily the native matched query. Multi-label selection allows true-class TP even when top1 is wrong; top1-only filtering is not justified. Geometry strata are post-treatment outcomes and associations are not causal effects. SmallGT flags are native all-area matches, not small-area matching or APs decomposition. No pre-LQE raw head isolation.')
    write(dest/'report.json',result);write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    print(json.dumps(dict(status=result['status'],P2_small=comparisons['p2_skip']['COCO_small'],attention_small=comparisons['backbone_attention']['COCO_small'])),flush=True)
if __name__=='__main__':main()
