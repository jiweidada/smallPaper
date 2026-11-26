"""Actual fixed-weight P2 controls: geometry, native TP and relative-class cohorts."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    spec=read(HERE/'protocol.json');assert sha(__file__)==spec['reader_sha256']
    for n,h in spec['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    proof=read(ROOT/'outputs/p2_subspace_collection12_20261011_v1/seed2/verification.json');assert proof['status']=='PASS_COMPLETE_P2_RUNTIME_ARTIFACTS_AND_TWO_NATIVE_ENDPOINTS'
    formal=read(ROOT/'outputs/user_seed2_formal_collection12_20261011_v1/seed2_p2_skip/verification.json');assert formal['status']=='PASS_USER_FORMAL_FIXED12_PRIMARY_AND_SAVED_EMA_EXACT'
    tools=module('subspace_geom_saved_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py');p,geo,_,_,readers=tools.prepared();base=tools.cached_baseline(2,p,readers['full']);ann=base['annotation'];ids=base['values']['annotation_id'];cats=base['values']['category_id'];lookup={int(g):i for i,g in enumerate(ids)}
    relative=module('subspace_relative_class',ROOT/'英文小论文/code/user_class_margin_geometry12_20261011_v1/analyze.py')
    folders={'parentP2':Path(formal['downloaded_packet'])/'run'}
    for mode in ('dc16_norm','svd16_norm'):folders[mode]=ROOT/f'outputs/p2_subspace_collection12_20261011_v1/seed2/extracted/seed2_GPU/{mode}'
    grouped={i:[] for i in range(548)}
    for a in ann['annotations']:grouped[a['image_id']].append(a)
    groups=tools.groups(ann,base['values'])
    for cid in range(10):groups[f'COCO_small_class{cid}']=groups['COCO_small']&(cats==cid)
    output=ROOT/'outputs/p2_subspace_geometry_class12_20261011_v1';output.mkdir(exist_ok=False)
    runs={};details={};margin_data={}
    for tag,folder in folders.items():
        summary,flags,_=readers['control'].native_endpoint(folder/'endpoint_epoch12',ann);identity=summary['identity'];readers['control'].stages(folder,identity,ann)
        if tag=='parentP2':assert identity==formal['identity']
        else:assert identity['runtime_intervention']['parent_checkpoint_sha256']==proof['parent_checkpoint_sha256'] and identity['runtime_intervention']['mode']==tag
        manifest=read(folder/'stages_epoch12/manifest.json')
        def image(i):
            f=folder/f'stages_epoch12/image_{i:06d}.npz';assert sha(f)==manifest['files'][f.name];return npz(f)
        values=geo.collect(ann,image,flags)
        for n in ('annotation_id','category_id','image_id','original_area'):assert np.array_equal(values[n],base['values'][n])
        a=dict(annotation_id=ids,category_id=cats,native_TP=values['native_TP'][:,[0,5]])
        for n in ('support_count','same_query_final_target_logit_index'):a[n]=np.zeros((len(ids),2),np.int16)
        for n in ('max_relative_class_margin','pre_margin_at_final_target_logit_query','final_margin_at_final_target_logit_query','pre_IoU_at_final_target_logit_query'):a[n]=np.zeros((len(ids),2),np.float64)
        for iid,entries in grouped.items():
            if not entries:continue
            z=image(iid);gt=np.array([g['bbox'] for g in entries],np.float64);category=np.array([g['category_id'] for g in entries]);ix=np.array([lookup[g['id']] for g in entries])
            logits=z['final_logits'].astype(np.float64);pre=z['pre_after_first_layer_logits'].astype(np.float64);iou=[]
            for n in ('final_boxes','pre_after_first_layer_boxes'):
                boxes=z[n];valid=(boxes[:,2:]>0).all(1);matrix=np.zeros((len(entries),300));matrix[:,valid]=geo.iou_matrix(gt,geo.pixel_xywh(boxes[valid],z['original_hw']));iou.append(matrix)
            assert np.array_equal(iou[0].max(1),values['max_iou'][ix,2]);margin=relative.relative_margin(logits,category);pre_margin=relative.relative_margin(pre,category);target=logits[:,category].T
            for ti,t in enumerate((.5,.75)):
                support=iou[0]>=t;active=support.any(1);a['support_count'][ix,ti]=support.sum(1);best=np.where(support,margin,-np.inf).max(1);a['max_relative_class_margin'][ix[active],ti]=best[active]
                q=np.where(support,target,-np.inf).argmax(1);rows=np.arange(len(entries));ai=ix[active]
                a['same_query_final_target_logit_index'][ai,ti]=q[active];a['pre_margin_at_final_target_logit_query'][ai,ti]=pre_margin[rows[active],q[active]];a['final_margin_at_final_target_logit_query'][ai,ti]=margin[rows[active],q[active]];a['pre_IoU_at_final_target_logit_query'][ai,ti]=iou[1][rows[active],q[active]]
        assert np.array_equal(a['support_count']>0,values['max_iou'][:,2,None]>=np.array([.5,.75]))
        assert not np.any(a['native_TP']&(a['support_count']==0))
        runs[tag]=values;margin_data[tag]=a;np.savez_compressed(output/(tag+'_geometry_per_GT.npz'),**values);np.savez_compressed(output/(tag+'_class_per_GT.npz'),**a)
        details[tag]=dict(identity=identity,overall_percent=summary['overall_percent'],stage_manifest_sha256=sha(folder/'stages_epoch12/manifest.json'),endpoint_manifest_sha256=sha(folder/'endpoint_epoch12/manifest.json'),groups={g:{str(t):relative.describe(mask,a,ti,spec['margin_epsilon']) for ti,t in enumerate((.5,.75))} for g,mask in groups.items()})
    comparisons={}
    for mode in ('dc16_norm','svd16_norm'):
        B,C=runs['parentP2'],runs[mode];bm,cm=margin_data['parentP2'],margin_data[mode];rows={}
        for group,mask in groups.items():
            rows[group]={}
            for ti,(t,k) in enumerate(((.5,0),(.75,5))):
                bg=B['max_iou'][:,2]>=t;cg=C['max_iou'][:,2]>=t;both=mask&bg&cg;gain=mask&~B['native_TP'][:,k]&C['native_TP'][:,k];loss=mask&B['native_TP'][:,k]&~C['native_TP'][:,k]
                bc=bm['max_relative_class_margin'][:,ti]>spec['margin_epsilon'];cc=cm['max_relative_class_margin'][:,ti]>spec['margin_epsilon']
                row=dict(GT=int(mask.sum()),supported_by_boundary={stage:dict(parent=int((mask&(B['max_iou'][:,s]>=t)).sum()),intervention=int((mask&(C['max_iou'][:,s]>=t)).sum()),delta=int((mask&(C['max_iou'][:,s]>=t)).sum())-int((mask&(B['max_iou'][:,s]>=t)).sum())) for s,stage in enumerate(geo.STAGES)},native_TP_gained=int(gain.sum()),native_TP_lost=int(loss.sum()),native_TP_net=int(gain.sum()-loss.sum()),
                    gained_TP_parent_no_final_support=int((gain&~bg).sum()),gained_TP_parent_supported_without_correct_relative_top1=int((gain&bg&~bc).sum()),gained_TP_parent_supported_correct_top1_exists=int((gain&bg&bc).sum()),
                    lost_TP_intervention_no_final_support=int((loss&~cg).sum()),lost_TP_intervention_supported_without_correct_relative_top1=int((loss&cg&~cc).sum()),lost_TP_intervention_supported_correct_top1_exists=int((loss&cg&cc).sum()),
                    both_final_supported_GT=int(both.sum()),correct_relative_top1_among_both_parent=int((both&bc).sum()),correct_relative_top1_among_both_intervention=int((both&cc).sum()))
                assert row['gained_TP_parent_no_final_support']+row['gained_TP_parent_supported_without_correct_relative_top1']+row['gained_TP_parent_supported_correct_top1_exists']==row['native_TP_gained']
                assert row['lost_TP_intervention_no_final_support']+row['lost_TP_intervention_supported_without_correct_relative_top1']+row['lost_TP_intervention_supported_correct_top1_exists']==row['native_TP_lost'];rows[group][str(t)]=row
        comparisons[mode]=rows
    report=dict(status='ACTUAL_P2_RANK16_INTERVENTIONS_GEOMETRY_AND_RELATIVE_CLASS_COHORTS',seed=2,images=548,GT=38759,small_GT=26586,runs=details,comparisons=comparisons,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_native_matching_calls=0,new_training_epochs=0,training_efficacy_proven=False,unique_root_proven=False,novelty_proven=False,
        scope='All actual saved outputs from two rank16 fixed-weight interventions and their same checkpoint parent. GT identity joins only, no cross-run query alignment. Relative class top1 and support are descriptive post-intervention strata; native multi-label TP can occur despite wrong relative top1. Native all-area GT flags stratified by small size are not small-area matching or APs decomposition. Within-query pre/final comparison is within one run only. Equal weight rank/norm does not match feature energy or approximation error.')
    write(output/'report.json',report);write(output/'manifest.json',dict(files={p.name:sha(p) for p in output.iterdir() if p.is_file()}));assert 'torch' not in sys.modules
    print(json.dumps(dict(status=report['status'],small={k:v['COCO_small'] for k,v in comparisons.items()})),flush=True)
if __name__=='__main__':main()
