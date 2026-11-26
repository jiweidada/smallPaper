"""All-GT descriptive comparison of actual primary12 Stem controls; no replay claim."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
read=lambda p:json.loads(Path(p).read_bytes())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def analyze(seed,arm):
    assert seed in (1,2) and arm in ('attention_spatial_mean','early_final_downsample')
    tools=module('remaining_primary_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
    protocol,geometry,onset,rank,readers=tools.prepared()
    jp=ROOT/'英文小论文/code/stem_joint_error_analysis_20261011_v2'
    assert sha(jp/'analyze.py')==read(jp/'protocol.json')['analysis_sha256']
    joint=module('remaining_primary_joint_partition',jp/'analyze.py')
    folder=ROOT/f'outputs/stem_component_primary_collection_20261011_v2/seed{seed}_{arm}_v1'
    v=read(folder/'primary_verification_v1.json')
    assert v['status']=='PASS_COMPONENT_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
    assert v['verification_script_sha256']==sha(ROOT/'英文小论文/code/stem_component_primary_verify12_20261011_v2/verify_primary.py')
    assert v['all_collected_files_verified'] and v['actual_stage_images']==548
    assert not v['saved_EMA_independent_replay_verified'] and not v['formal_terminal_reader_passed']
    packet=Path(v['downloaded_packet']);manifest=read(packet/'primary_manifest.json')
    assert sha(packet/'primary_manifest.json')==v['primary_manifest_sha256']
    assert v['identity']['seed']==seed and v['identity']['arm']==arm
    for n,item in manifest['files'].items():
        assert (packet/n).stat().st_size==item['bytes'] and sha(packet/n)==item['sha256'],n
    base=tools.cached_baseline(seed,protocol,readers['full'])
    summary,flags,_=readers['control'].native_endpoint(packet/'run/endpoint_epoch12',base['annotation'])
    readers['control'].stages(packet/'run',v['identity'],base['annotation'])
    assert summary['identity']==v['identity'] and summary['native_stats']==v['native_stats']
    cache=ROOT/'outputs/stem_primary_joint_analysis12_20261011_v1'
    assert sha(cache/'report.json')=='787c0a77fe0aa4027097b5cc6f5cb571c6c66d4fb536a85e71382bcfabaf56bd'
    for n,h in read(cache/'manifest.json')['files'].items():assert sha(cache/n)==h,n
    assert read(cache/'report.json')['analysis_sha256']==sha(ROOT/'英文小论文/code/stem_primary_joint_analysis12_20261011_v1/analyze.py')
    full_folder=ROOT/f'outputs/stem_primary_collection_20261011_v1/seed{seed}_original_stem_v1'
    full_v=read(full_folder/'primary_verification_v1.json');full_packet=Path(full_v['downloaded_packet'])
    full_summary,full_flags,_=readers['full'].native_endpoint(full_packet/'run/endpoint_epoch12',base['annotation'])
    assert full_summary['identity']==full_v['identity'] and full_summary['native_stats']==full_v['native_stats']
    for k in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):
        assert summary['identity'][k]==full_summary['identity'][k]
    root=ROOT/'outputs/stem_remaining_primary_analysis12_20261011_v1';root.mkdir(exist_ok=True)
    dest=root/f'seed{seed}_{arm}';dest.mkdir(exist_ok=False)

    def image(iid):
        return load_npz(packet/f'run/stages_epoch12/image_{iid:06d}.npz')

    control=geometry.collect(base['annotation'],image,flags)
    rank.analyze(packet,arm,tools.HERE/'ranking_protocol.json',dest/'control_ranking',
        base['packet']/'native/engine/deim/postprocessor.py',base['packet']/'reference/val_annotations.json')
    rr=read(dest/'control_ranking/report.json')
    assert rr['identity']==v['identity'] and not rr['score_mismatch_images'] and rr['native_TP_proxy_conflicts']==0
    cr=load_npz(dest/'control_ranking/per_GT.npz')
    full=load_npz(cache/f'seed{seed}_original_stem/stage_per_GT.npz')
    fr=load_npz(cache/f'seed{seed}_original_stem/ranking/per_GT.npz')
    for name in ('annotation_id','image_id','category_id','original_area'):
        assert np.array_equal(control[name],full[name]) and np.array_equal(control[name],cr[name]) and np.array_equal(control[name],fr[name])
    ff=tools.sorted_flags(full_flags)
    assert np.array_equal(full['annotation_id'],ff['annotation_id']) and np.array_equal(full['native_TP'],ff['hits'])
    masks=tools.groups(base['annotation'],control)
    # Native small area convention is closed at1024. These actual annotations have no boundary-area GT.
    masks['COCO_small']=control['original_area']<=1024
    assert not np.any(control['original_area']==1024)
    assert len(control['annotation_id'])==38759 and int(masks['COCO_small'].sum())==26586
    assert np.array_equal(masks['input_under4'],cr['input_shortside']<4)
    masks['COCO_small_input_under4']=masks['COCO_small']&masks['input_under4']
    categories={int(c['id']):c['name'] for c in base['annotation']['categories']}
    assert len(categories)==10
    for cid in sorted(categories):
        masks['COCO_small_class_'+str(cid)+'_'+categories[cid]]=masks['COCO_small']&(control['category_id']==cid)
        masks['all_class_'+str(cid)+'_'+categories[cid]]=control['category_id']==cid
    outcomes={}
    for threshold,ti in ((.5,0),(.75,5)):
        cl,cp=joint.partition(control,cr,threshold);fl,fp=joint.partition(full,fr,threshold)
        cs=control['max_iou']>=threshold;fs=full['max_iou']>=threshold
        changed=cs!=fs;any_changed=changed.any(1);first=changed.argmax(1)
        outcomes[str(threshold)]={}
        for name,mask in masks.items():
            gained=mask&~control['native_TP'][:,ti]&full['native_TP'][:,ti]
            lost=mask&control['native_TP'][:,ti]&~full['native_TP'][:,ti]
            matrix=np.zeros((len(joint.LABELS),len(joint.LABELS)),dtype=np.int64)
            np.add.at(matrix,(cl[mask],fl[mask]),1)
            assert int(matrix.sum())==int(mask.sum())
            outcomes[str(threshold)][name]=dict(GT=int(mask.sum()),
                control=joint.summarize(mask,cl,cp,cr,threshold),full_enhanced=joint.summarize(mask,fl,fp,fr,threshold),
                control_to_full_joint_cell_counts=matrix.tolist(),native_all_area_TP_gained=int(gained.sum()),
                native_all_area_TP_lost=int(lost.sum()),native_all_area_TP_net=int(gained.sum()-lost.sum()),
                supported_GT_by_boundary={n:dict(control=int((mask&cs[:,i]).sum()),full=int((mask&fs[:,i]).sum()),
                    delta=int((mask&fs[:,i]).sum()-(mask&cs[:,i]).sum())) for i,n in enumerate(geometry.STAGES)},
                gained_TP_first_observed_geometric_support_change={n:int((gained&any_changed&(first==i)).sum()) for i,n in enumerate(geometry.STAGES)},
                gained_TP_unchanged_three_boundary_support=int((gained&~any_changed).sum()),
                gained_TP_control_outcomes={n:int((gained&(cl==i)).sum()) for i,n in enumerate(joint.LABELS)},
                lost_TP_full_outcomes={n:int((lost&(fl==i)).sum()) for i,n in enumerate(joint.LABELS)})
    np.savez_compressed(dest/'control_stage_per_GT.npz',**control)
    source_protocol=read(packet/'code/training_protocol.json')
    report=dict(status='SAME_GT_PRIMARY12_COMPONENT_DESCRIPTION_SINGLE_SEED_REPLAY_PENDING',seed=seed,arm=arm,
        control_scope=source_protocol['controls'][arm],all548_images=True,all38759_GT=True,all10_categories_retained=True,
        full_AP_percent={n:full_summary['native_stats'][i]*100 for n,i in (('AP',0),('APs',3))},
        control_AP_percent={n:summary['native_stats'][i]*100 for n,i in (('AP',0),('APs',3))},
        full_minus_control_pp={n:(full_summary['native_stats'][i]-summary['native_stats'][i])*100 for n,i in (('AP',0),('APs',3))},
        outcomes=outcomes,labels=list(joint.LABELS),score_mismatch_images=rr['score_mismatch_images'],
        ambiguous_images=rr['ambiguous_images'],native_TP_proxy_conflicts=rr['native_TP_proxy_conflicts'],
        native_score_list_max_abs_error=rr['native_score_list_max_abs_error'],
        provenance=dict(control_verification_sha256=sha(folder/'primary_verification_v1.json'),
            control_primary_manifest_sha256=v['primary_manifest_sha256'],control_stage_manifest_sha256=sha(packet/'run/stages_epoch12/manifest.json'),
            full_primary_joint_manifest_sha256=sha(cache/'manifest.json'),prepared_protocol_sha256=sha(tools.PROTOCOL),
            joint_protocol_sha256=sha(jp/'protocol.json')),analysis_source_sha256=sha(__file__),
        saved_EMA_independent_replay_verified=False,formal_terminal_reader_passed=False,
        other_seed_same_fixed12_endpoint_pending=True,causal_root_proven=False,novelty_proven=False,
        new_model_forwards=0,new_evaluator_calls=0,new_native_evaluator_or_Hungarian_matching_calls=0,new_optimizer_updates=0,new_training_epochs=0,
        scope='Exploratory description of actual native primary12 tensors. Same-GT joins are not cross-run query/box identities. First boundary contains only selected300 REGRESSED boxes; no dense8400 regressed boxes, unique TopK-versus-regression diagnosis, historical training assignments or AP decomposition is inferred. Small-cohort TP uses all-area native GT flags, not APs-specific matching. Other seed and independent saved-EMA proof remain required.')
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    write(dest/'report.json',report)
    write(dest/'manifest.json',dict(files={x.relative_to(dest).as_posix():sha(x) for x in dest.rglob('*') if x.is_file()}))
    print(json.dumps(dict(seed=seed,arm=arm,status=report['status'],delta=report['full_minus_control_pp'],
        small={t:{k:report['outcomes'][t]['COCO_small'][k] for k in ('native_all_area_TP_net','supported_GT_by_boundary','gained_TP_control_outcomes')}
            for t in ('0.5','0.75')})))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(1,2),required=True)
    parser.add_argument('--arm',choices=('attention_spatial_mean','early_final_downsample'),required=True)
    a=parser.parse_args();analyze(a.seed,a.arm)
