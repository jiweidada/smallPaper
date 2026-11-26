"""Analyze four completed fixed12 EMA diagnostic records against verified GT stages."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
from selection_core import image_description,verify_grid_centers,LEVELS,RADII,THRESHOLDS

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def manifest(folder,expected=None):
    m=read(folder/'manifest.json')
    if expected is not None:assert set(m['files'])==set(expected)
    for name,digest in m['files'].items():
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts and ':' not in name
        assert sha(folder/path)==digest
    return m
def median_finite(x):
    x=np.asarray(x);a=x[np.isfinite(x)]
    return float(np.median(a)) if len(a) else None
def aggregate(values,mask):
    out={}
    for ri,radius in enumerate(RADII):
        out[str(radius)]={}
        for li,level in enumerate(LEVELS):
            c=values['selected_neighborhood_sites'][:,ri,li]
            v=values['valid_neighborhood_sites'][:,ri,li]
            selected=c>0
            row=dict(GT=int(mask.sum()),no_valid_nearby_site=int((mask&(v==0)).sum()),
                any_nearby_site_selected=int((mask&selected).sum()),
                no_nearby_site_selected=int((mask&~selected).sum()),
                median_best_valid_score_margin_to_actual_cutoff=median_finite(values['best_max_class_margin_to_cutoff'][mask,ri,li]),
                median_best_valid_max_class_rank_min=median_finite(values['best_max_class_rank_min'][mask&(v>0),ri,li]))
            row['initial_geometry_overlap']={}
            for ti,t in enumerate(THRESHOLDS):
                any_support=values['initial_any_support'][:,ti]
                local=values['selected_local_initial_support_count'][:,ri,li,ti]>0
                row['initial_geometry_overlap'][str(t)]=dict(
                    any_initial_IoU_support=int((mask&any_support).sum()),
                    local_selected_box_has_IoU_support=int((mask&local).sum()),
                    supported_only_by_nonlocal_selected_boxes=int((mask&any_support&~local).sum()),
                    no_local_selected_site_but_initial_IoU_support=int((mask&~selected&any_support).sum()),
                    local_selected_site_but_no_initial_IoU_support=int((mask&selected&~any_support).sum()))
                assert not (local&~any_support).any()
            out[str(radius)][level]=row
    return out
def group_masks(values,annotation):
    anns={int(a['id']):a for a in annotation['annotations']};ims={int(i['id']):i for i in annotation['images']}
    short=np.array([min(640*anns[int(g)]['bbox'][2]/ims[int(anns[int(g)]['image_id'])]['width'],
                        640*anns[int(g)]['bbox'][3]/ims[int(anns[int(g)]['image_id'])]['height']) for g in values['annotation_id']])
    return {'all':np.ones(len(short),bool),'COCO_small':values['original_area']<1024,
            'input_under4':short<4,'input_4_to_under8':(short>=4)&(short<8),
            'input_8_to_under16':(short>=8)&(short<16)}
def analyze(spec_path,output):
    p=read(HERE/'protocol.json')
    for n,h in p['files'].items():assert sha(HERE/n)==h,n
    for n,h in p['dependencies'].items():assert sha(ROOT/n)==h,n
    spec=read(spec_path);jobs=spec['jobs'];required={(a,s) for a in ('original_stem','native_stem_scratch') for s in (1,2)}
    assert len(jobs)==4 and {(j['arm'],j['seed']) for j in jobs}==required
    paired=Path(spec['paired_stage_analysis']);assert sha(paired/'manifest.json')==spec['paired_stage_manifest_sha256']
    manifest(paired);paired_report=read(paired/'report.json')
    assert paired_report['status']=='SAVED_FIXED12_STAGE_AND_SELECTION_DESCRIPTIONS'
    assert paired_report['protocol_sha256']==sha(ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/protocol.json')
    annotation_path=Path(spec['annotation_path']);annotation=read(annotation_path)
    assert {int(i['id']) for i in annotation['images']}==set(range(548))
    geometry=module('selection_saved_geometry',ROOT/'英文小论文/code/scale_protection12_monitor_20261010_v1/geometry_core.py')
    output=Path(output);assert not output.exists()
    prepared=[]
    for job in jobs:
        folder=Path(job['diagnostic']);identity=paired_report['descriptions'][str((job['arm'],job['seed']))]['identity']
        terminal_report=Path(job['terminal_report'])
        assert sha(terminal_report)==paired_report['descriptions'][str((job['arm'],job['seed']))]['report_sha256']
        terminal=read(terminal_report)
        expected_status=p['terminal_status'][job['arm']]
        assert terminal['status']==expected_status and terminal['identity']==identity
        assert terminal['fixed_endpoint']==dict(epochs=12,zero_based_epoch=11,weights='EMA')
        assert terminal['all548_stage_arrays_reloaded_exact'] and terminal['first100_inputs_and_pre_forward_RNG_exact']
        assert sha(annotation_path)==identity['val_annotation_sha256']
        assert sha(folder/'verification.json')==job['verification_sha256']
        assert sha(folder/'selection_epoch12/manifest.json')==job['selection_manifest_sha256']
        verification=read(folder/'verification.json')
        assert verification['status']=='PASS_SELECTION_DIAGNOSTIC_EXACT_SAVED_EMA_REPLAY'
        assert verification['identity']==identity and verification['seed']==job['seed'] and verification['arm']==job['arm']
        assert verification['diagnostic_protocol_sha256']==p['diagnostic_protocol_sha256']
        assert verification['verification_script_sha256']==p['diagnostic_replay_sha256']
        assert verification['stats_max_abs_fraction_difference']==verification['stats_tolerance']==0
        assert verification['native_GT_hits_exact'] and verification['all548_stage_arrays_exact']
        assert verification['candidate_model_forwards']==verification['actual_selector_calls']==verification['existing_selected_bbox_head_calls']==137
        assert verification['selection_images']==548 and verification['additional_bbox_head_or_topk_calls']==0
        assert verification['baseline_model_forwards']==verification['optimizer_updates']==verification['epochs_added']==0
        done=read(folder/'completion.json');assert done['exit_code']==0 and not done['timed_out']
        selection=folder/'selection_epoch12';sm=manifest(selection,{'summary.json'}|{f'image_{i:06d}.npz' for i in range(548)})
        assert sm['trace_code_sha256']==p['selection_stage_trace_sha256']
        summary=read(selection/'summary.json')
        assert summary['identity']==identity and summary['images']==548 and summary['image_ids']==list(range(548))
        assert summary['epoch_zero_based']==11 and summary['evaluated_weights']=='EMA'
        assert summary['dense_locations']==8400 and summary['selected_queries']==300
        assert summary['additional_head_calls']==summary['additional_topk_calls']==0
        stage=folder/'stages_epoch12';manifest(stage,{'summary.json'}|{f'image_{i:06d}.npz' for i in range(548)})
        assert read(stage/'summary.json')['identity']==identity
        values=npz(paired/(job['arm']+'_seed'+str(job['seed']))/'per_GT.npz')
        assert set(values['annotation_id'].tolist())=={int(a['id']) for a in annotation['annotations']}
        prepared.append((job,folder,values))
    # Reject missing/nonterminal/unverified evidence before creating output.
    output.mkdir()
    descriptions={};arm_values={}
    by_image={i:[] for i in range(548)}
    anns={int(a['id']):a for a in annotation['annotations']}
    for a in annotation['annotations']:by_image[int(a['image_id'])].append(a)
    for job,folder,values in prepared:
        loc={int(g):i for i,g in enumerate(values['annotation_id'])};joined={k:values[k].copy() for k in
            ('annotation_id','category_id','image_id','original_area','native_TP','max_iou')}
        audits=[];observed_ids=set()
        for iid in range(548):
            selection=npz(folder/f'selection_epoch12/image_{iid:06d}.npz')
            stage=npz(folder/f'stages_epoch12/image_{iid:06d}.npz')
            assert int(selection['image_id'])==iid and np.array_equal(selection['original_hw'],stage['original_hw'])
            expected_image=next(im for im in annotation['images'] if int(im['id'])==iid)
            assert selection['original_hw'].tolist()==[expected_image['height'],expected_image['width']]
            targets=sorted(by_image[iid],key=lambda a:int(a['id']))
            gt=np.array([a['bbox'] for a in targets],float).reshape(-1,4)
            boxes=stage['encoder_initial_boxes'];valid=(boxes[:,2:]>0).all(1)
            iou=np.zeros((len(targets),300),float)
            iou[:,valid]=geometry.iou_matrix(gt,geometry.pixel_xywh(boxes[valid],selection['original_hw']))
            result,audit=image_description(selection,targets,iou);audit['image_id']=iid;audits.append(audit)
            audit['finite_observed_anchor_center_max_abs_grid_error']=verify_grid_centers(selection)
            positions=np.array([loc[int(a['id'])] for a in targets],dtype=np.int64)
            assert np.array_equal(values['image_id'][positions],np.full(len(targets),iid))
            assert np.array_equal(values['category_id'][positions],np.array([a['category_id'] for a in targets],dtype=np.int64))
            if len(targets):assert np.array_equal(iou.max(1),values['max_iou'][positions,0])
            for key,array in result.items():
                if key not in joined:joined[key]=np.empty((len(loc),*array.shape[1:]),dtype=array.dtype)
                joined[key][positions]=array
            observed_ids.update(int(a['id']) for a in targets)
        assert observed_ids==set(loc)
        label=job['arm']+'_seed'+str(job['seed']);dest=output/label;dest.mkdir()
        np.savez_compressed(dest/'per_GT.npz',**joined)
        masks=group_masks(joined,annotation)
        groups={g:dict(all=aggregate(joined,m),native_miss_IoU50=aggregate(joined,m&~joined['native_TP'][:,0]),
                       native_miss_IoU75=aggregate(joined,m&~joined['native_TP'][:,5])) for g,m in masks.items()}
        description=dict(groups=groups,image_audits=audits,verification_sha256=job['verification_sha256'],
            selection_manifest_sha256=job['selection_manifest_sha256'],actual_GT=len(loc),all548_images=True,
            normalized_radius_windows_are_spatial_proxies=True)
        write(dest/'description.json',description);descriptions[label]=description;arm_values[(job['arm'],job['seed'])]=joined
    pairs={}
    for seed in (1,2):
        a,b=arm_values[('native_stem_scratch',seed)],arm_values[('original_stem',seed)]
        for key in ('annotation_id','category_id','image_id','original_area'):assert np.array_equal(a[key],b[key])
        pairs[str(seed)]={}
        for group,mask in group_masks(a,annotation).items():
            table={}
            for ri,radius in enumerate(RADII):
                av=a['selected_neighborhood_sites'][:,ri,3]>0;bv=b['selected_neighborhood_sites'][:,ri,3]>0
                new_tp=~a['native_TP'][:,0]&b['native_TP'][:,0];lost_tp=a['native_TP'][:,0]&~b['native_TP'][:,0]
                gain=~av&bv;loss=av&~bv
                table[str(radius)]=dict(GT=int(mask.sum()),gained_nearby_selection=int((mask&gain).sum()),
                    lost_nearby_selection=int((mask&loss).sum()),new_native_TP50=int((mask&new_tp).sum()),
                    lost_native_TP50=int((mask&lost_tp).sum()),new_TP50_with_nearby_selection_gain=int((mask&new_tp&gain).sum()),
                    lost_TP50_with_nearby_selection_loss=int((mask&lost_tp&loss).sum()))
            pairs[str(seed)][group]=table
    report=dict(status='ANALYZED_FIXED12_ACTUAL_TOPK_LOCATION_AND_GEOMETRY_DESCRIPTION',pairs=pairs,
        seeds=[1,2],arms=['native_stem_scratch','original_stem'],spec_sha256=sha(spec_path),
        paired_stage_manifest_sha256=spec['paired_stage_manifest_sha256'],levels=list(LEVELS),radii=list(RADII),
        causal_root_proven=False,novelty_proven=False,actual_model_efficacy_inferred_here=False,
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,
        scope=p['scope'])
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files={x.relative_to(output).as_posix():sha(x) for x in output.rglob('*') if x.is_file()}))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();print(analyze(a.spec,a.output)['status'])

