"""Same-GT full versus RGB-input control, actual primary12 tensors; replay still pending."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/stem_highfreq_primary_analysis12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,allow_nan=False)
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
tools=module('HF_primary_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
p,geometry,onset,rank,readers=tools.prepared()
jpath=ROOT/'英文小论文/code/stem_joint_error_analysis_20261011_v2'
assert sha(jpath/'analyze.py')==read(jpath/'protocol.json')['analysis_sha256']
joint=module('HF_primary_partition',jpath/'analyze.py')
full_cache=ROOT/'outputs/stem_primary_joint_analysis12_20261011_v1'
assert sha(full_cache/'report.json')=='787c0a77fe0aa4027097b5cc6f5cb571c6c66d4fb536a85e71382bcfabaf56bd'
for n,h in read(full_cache/'manifest.json')['files'].items():assert sha(full_cache/n)==h,n
validated={}
for seed in (1,2):
    folder=ROOT/f'outputs/stem_component_primary_collection_20261011_v2/seed{seed}_highfreq_rgb_v1'
    v=read(folder/'primary_verification_v1.json')
    assert v['status']=='PASS_COMPONENT_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
    assert v['verification_script_sha256']==sha(Path(__file__).parent/'verify_primary.py')
    assert v['all_collected_files_verified'] and v['actual_stage_images']==548
    assert not v['saved_EMA_independent_replay_verified'] and not v['formal_terminal_reader_passed']
    packet=Path(v['downloaded_packet']);manifest=read(packet/'primary_manifest.json')
    assert sha(packet/'primary_manifest.json')==v['primary_manifest_sha256']
    for n,item in manifest['files'].items():
        assert (packet/n).stat().st_size==item['bytes'] and sha(packet/n)==item['sha256']
    base=tools.cached_baseline(seed,p,readers['full'])
    summary,flags,_=readers['control'].native_endpoint(packet/'run/endpoint_epoch12',base['annotation'])
    readers['control'].stages(packet/'run',v['identity'],base['annotation'])
    assert summary['identity']==v['identity'] and summary['native_stats']==v['native_stats']
    full_folder=ROOT/f'outputs/stem_primary_collection_20261011_v1/seed{seed}_original_stem_v1'
    full_v=read(full_folder/'primary_verification_v1.json');full_packet=Path(full_v['downloaded_packet'])
    full_summary,full_flags,_=readers['full'].native_endpoint(full_packet/'run/endpoint_epoch12',base['annotation'])
    assert full_summary['identity']==full_v['identity'] and full_summary['native_stats']==full_v['native_stats']
    for key in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):
        assert summary['identity'][key]==full_summary['identity'][key]
    validated[seed]=(folder,v,packet,base,summary,flags,full_summary,full_flags)
OUT.mkdir(exist_ok=False);rows=[];provenance={}
for seed in (1,2):
    folder,v,packet,base,summary,flags,full_summary,full_flags=validated[seed]
    dest=OUT/f'seed{seed}';dest.mkdir()
    def image(iid):
        with np.load(packet/f'run/stages_epoch12/image_{iid:06d}.npz',allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
    control=geometry.collect(base['annotation'],image,flags)
    rank.analyze(packet,'highfreq_rgb',tools.HERE/'ranking_protocol.json',dest/'RGB_ranking',
        base['packet']/'native/engine/deim/postprocessor.py',base['packet']/'reference/val_annotations.json')
    rr=read(dest/'RGB_ranking/report.json')
    assert rr['identity']==v['identity'] and not rr['score_mismatch_images'] and rr['native_TP_proxy_conflicts']==0
    with np.load(dest/'RGB_ranking/per_GT.npz',allow_pickle=False) as z:cr={n:z[n].copy() for n in z.files}
    with np.load(full_cache/f'seed{seed}_original_stem/stage_per_GT.npz',allow_pickle=False) as z:full={n:z[n].copy() for n in z.files}
    with np.load(full_cache/f'seed{seed}_original_stem/ranking/per_GT.npz',allow_pickle=False) as z:fr={n:z[n].copy() for n in z.files}
    for n in ('annotation_id','image_id','category_id','original_area'):
        assert np.array_equal(control[n],full[n]) and np.array_equal(control[n],cr[n]) and np.array_equal(control[n],fr[n])
    ff=tools.sorted_flags(full_flags)
    assert np.array_equal(full['annotation_id'],ff['annotation_id']) and np.array_equal(full['native_TP'],ff['hits'])
    masks=tools.groups(base['annotation'],control)
    assert np.array_equal(masks['input_under4'],cr['input_shortside']<4)
    masks['COCO_small_input_under4']=masks['COCO_small']&masks['input_under4']
    categories={int(c['id']):c['name'] for c in base['annotation']['categories']}
    for cid in sorted(categories):masks['COCO_small_class_'+str(cid)+'_'+categories[cid]]=masks['COCO_small']&(control['category_id']==cid)
    outcomes={}
    for threshold,ti in ((.5,0),(.75,5)):
        cl,cp=joint.partition(control,cr,threshold);fl,fp=joint.partition(full,fr,threshold)
        cs=control['max_iou']>=threshold;fs=full['max_iou']>=threshold
        changed=cs!=fs;any_changed=changed.any(1);first=changed.argmax(1)
        outcomes[str(threshold)]={}
        for name,mask in masks.items():
            gained=mask&~control['native_TP'][:,ti]&full['native_TP'][:,ti]
            lost=mask&control['native_TP'][:,ti]&~full['native_TP'][:,ti]
            matrix=np.zeros((len(joint.LABELS),len(joint.LABELS)),dtype=np.int64);np.add.at(matrix,(cl[mask],fl[mask]),1)
            assert int(matrix.sum())==int(mask.sum())
            outcomes[str(threshold)][name]=dict(GT=int(mask.sum()),
                RGB_control=joint.summarize(mask,cl,cp,cr,threshold),full_highfreq=joint.summarize(mask,fl,fp,fr,threshold),
                RGB_to_full_joint_cell_counts=matrix.tolist(),native_TP_gained=int(gained.sum()),native_TP_lost=int(lost.sum()),
                native_TP_net=int(gained.sum()-lost.sum()),
                supported_GT_by_boundary={n:dict(RGB=int((mask&cs[:,i]).sum()),full=int((mask&fs[:,i]).sum()),
                    delta=int((mask&fs[:,i]).sum()-(mask&cs[:,i]).sum())) for i,n in enumerate(geometry.STAGES)},
                gained_TP_first_observed_geometric_support_change={n:int((gained&any_changed&(first==i)).sum()) for i,n in enumerate(geometry.STAGES)},
                gained_TP_unchanged_three_boundary_support=int((gained&~any_changed).sum()),
                gained_TP_RGB_outcomes={n:int((gained&(cl==i)).sum()) for i,n in enumerate(joint.LABELS)},
                lost_TP_full_outcomes={n:int((lost&(fl==i)).sum()) for i,n in enumerate(joint.LABELS)})
    row=dict(seed=seed,full_AP_percent={n:full_summary['native_stats'][i]*100 for n,i in (('AP',0),('APs',3))},
        RGB_control_AP_percent={n:summary['native_stats'][i]*100 for n,i in (('AP',0),('APs',3))},
        full_minus_RGB_pp={n:(full_summary['native_stats'][i]-summary['native_stats'][i])*100 for n,i in (('AP',0),('APs',3))},
        outcomes=outcomes,score_mismatch_images=rr['score_mismatch_images'],ambiguous_images=rr['ambiguous_images'],
        native_TP_proxy_conflicts=rr['native_TP_proxy_conflicts'],native_score_list_max_abs_error=rr['native_score_list_max_abs_error'])
    write(dest/'description.json',row);np.savez_compressed(dest/'RGB_stage_per_GT.npz',**control);rows.append(row)
    provenance[str(seed)]=dict(control_verification_sha256=sha(folder/'primary_verification_v1.json'),
        control_manifest_sha256=v['primary_manifest_sha256'],control_stage_manifest_sha256=sha(packet/'run/stages_epoch12/manifest.json'),
        full_primary_joint_manifest_sha256=sha(full_cache/'manifest.json'))
report=dict(status='ANALYZED_BOTH_SEEDS_FULL_VS_RGB_PRIMARY12_SAME_GT_REPLAY_PENDING',seeds=rows,labels=list(joint.LABELS),
    provenance=provenance,analysis_source_sha256=sha(__file__),prepared_protocol_sha256=sha(tools.PROTOCOL),
    saved_EMA_independent_replay_verified=False,formal_terminal_reader_passed=False,causal_root_or_novelty_proven=False,
    new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,
    scope='Actual same-GT fixed12 primary tensors from the unmodified enhanced Stem versus its RGB-input residual-branch control, both seeds, all548 images/38759 GT. IoU support is not exclusive query ownership, training assignment or APs decomposition. Initial boundary is selected300 regressed boxes; no dense8400 box pool or unique Stem/frequency/encoder/TopK cause inferred. RGB versus high-pass inputs also change input statistics/low-frequency content. Independent saved-weight replay remains pending.')
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
write(OUT/'report.json',report);write(OUT/'manifest.json',dict(files={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file()}))
print(json.dumps(dict(status=report['status'],seeds=[dict(seed=r['seed'],delta=r['full_minus_RGB_pp'],
    small50_native_TP_net=r['outcomes']['0.5']['COCO_small']['native_TP_net'],
    small50_supported_GT_by_boundary=r['outcomes']['0.5']['COCO_small']['supported_GT_by_boundary'],
    small50_gained=r['outcomes']['0.5']['COCO_small']['native_TP_gained'],
    small50_lost=r['outcomes']['0.5']['COCO_small']['native_TP_lost'],
    small50_gained_first_changed_boundary=r['outcomes']['0.5']['COCO_small']['gained_TP_first_observed_geometric_support_change'],
    under4_50_native_TP_net=r['outcomes']['0.5']['input_under4']['native_TP_net']) for r in rows])))
