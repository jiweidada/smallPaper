"""Spatial location join across native TopK records; no cross-run feature/query identity."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def support_partition(common_support,unique_support):
    return np.where(common_support&unique_support,3,np.where(common_support,1,np.where(unique_support,2,0)))
def main(seed):
    spec=read(HERE/'protocol.json');assert spec['reader_sha256']==sha(__file__)
    for n,h in spec['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    sys.path.insert(0,str(ROOT/'英文小论文/code/native_selection_analysis12_20261011_v5'));import analyze as selection
    p,helper,parts,aggregate=selection.prepared();stage_p,geo,_,_,readers=parts;base=helper.cached_baseline(seed,stage_p,readers['full']);ann=base['annotation'];ids=base['values']['annotation_id'];lookup={int(g):i for i,g in enumerate(ids)}
    grouped={i:[] for i in range(548)}
    for a in ann['annotations']:grouped[a['image_id']].append(a)
    arms=['original_stem','native_stem_scratch','highfreq_rgb'];roots={};values={};descriptions={}
    allroot=ROOT/'outputs/native_selection_analysis12_20261011_v5/actual_six'
    complete=read(allroot/'report.json');assert complete['status']=='ANALYZED_SIX_FIXED12_ACTUAL_TOPK_AND_SAVED_GEOMETRY_DESCRIPTIONS'
    assert complete['seeds']==[1,2] and set(complete['arms'])==set(arms)
    assert complete['protocol_sha256']==sha(ROOT/'英文小论文/code/native_selection_analysis12_20261011_v5/protocol.json')
    m=read(allroot/'manifest.json');assert all(sha(allroot/n)==h for n,h in m['files'].items())
    jobs=read(ROOT/'outputs/native_selection_analysis12_20261011_v5/six_actual_jobs.json')['jobs'];selection.validate_job_set(jobs)
    for arm in arms:
        job=next(j for j in jobs if j['seed']==seed and j['arm']==arm)
        label=arm+'_seed'+str(seed);root=allroot/label
        assert complete['descriptions'][label]['stages']['report_sha256']==job['report_sha256']==sha(job['report'])
        assert complete['descriptions'][label]['selection']['diagnostic_verification_sha256']==job['verification_sha256']
        values[arm]=npz(root/'per_GT.npz');descriptions[arm]=npz(root/'selection_per_GT.npz');roots[arm]=Path(job['diagnostic'])
        for key in ('annotation_id','category_id','image_id','original_area'):assert np.array_equal(values[arm][key],base['values'][key])
    groups=helper.groups(ann,base['values']);cats=base['values']['category_id']
    for cid in range(10):groups[f'COCO_small_class{cid}']=groups['COCO_small']&(cats==cid)
    pairs={left:dict(common_selection_images=[],left_partition=np.zeros((len(ids),2),np.int8),full_partition=np.zeros((len(ids),2),np.int8)) for left in arms[1:]}
    level_counts={a:np.zeros(3,np.int64) for a in arms};offset_delta={a:[] for a in arms[1:]}
    for iid,gtrows in grouped.items():
        target=np.array([g['bbox'] for g in gtrows],np.float64).reshape(-1,4);positions=np.array([lookup[g['id']] for g in gtrows],np.int64);stages={};records={};matrices={}
        for arm in arms:
            records[arm]=npz(roots[arm]/f'selection_epoch12/image_{iid:06d}.npz');stages[arm]=npz(roots[arm]/f'stages_epoch12/image_{iid:06d}.npz');r=records[arm];s=stages[arm];ix=r['selected_indices']
            assert r['image_id']==iid and np.array_equal(r['original_hw'],s['original_hw'])
            level=np.where(ix<6400,0,np.where(ix<8000,1,2));level_counts[arm]+=np.bincount(level,minlength=3)
            boxes=s['encoder_initial_boxes'];good=(boxes[:,2:]>0).all(1);m=np.zeros((len(gtrows),300));m[:,good]=geo.iou_matrix(target,geo.pixel_xywh(boxes[good],s['original_hw']));matrices[arm]=m
            if len(gtrows):assert np.array_equal(m.max(1),values[arm]['max_iou'][positions,0])
        for left in arms[1:]:
            L,F=records[left],records['original_stem'];assert np.array_equal(L['dense_anchor_logits'],F['dense_anchor_logits'])
            common,li,fi=np.intersect1d(L['selected_indices'],F['selected_indices'],return_indices=True);lu=np.ones(300,bool);fu=np.ones(300,bool);lu[li]=False;fu[fi]=False
            assert np.array_equal(L['selected_anchor_logits'][li],F['selected_anchor_logits'][fi])
            offset_delta[left].append(float(np.linalg.norm((L['selected_bbox_offsets'][li]-F['selected_bbox_offsets'][fi]).astype(np.float64))/np.sqrt(max(1,len(common)*4))))
            pairs[left]['common_selection_images'].append(dict(image_id=iid,common_selected_locations=len(common),left_unique=300-len(common),full_unique=300-len(common)))
            for ti,t in enumerate((.5,.75)):
                lc=(matrices[left][:,li]>=t).any(1);fc=(matrices['original_stem'][:,fi]>=t).any(1);ls=(matrices[left][:,lu]>=t).any(1);fs=(matrices['original_stem'][:,fu]>=t).any(1)
                pairs[left]['left_partition'][positions,ti]=support_partition(lc,ls);pairs[left]['full_partition'][positions,ti]=support_partition(fc,fs)
                assert np.array_equal(lc|ls,values[left]['max_iou'][positions,0]>=t) and np.array_equal(fc|fs,values['original_stem']['max_iou'][positions,0]>=t)
    output=ROOT/f'outputs/native_selection_common_location12_20261011_v3/seed{seed}';output.mkdir(exist_ok=False);comparisons={}
    labels=['none','common_selected_locations_only','full_or_left_unique_selected_locations_only','common_and_unique_locations']
    for left,arrays in pairs.items():
        rows={};L,F=values[left],values['original_stem'];lc,fc=arrays['left_partition'],arrays['full_partition']
        near_left=descriptions[left]['selected_neighborhood_sites'][:,1,3]>0;near_full=descriptions['original_stem']['selected_neighborhood_sites'][:,1,3]>0
        for group,mask in groups.items():
            rows[group]={}
            for ti,(t,nativeti) in enumerate(((.5,0),(.75,5))):
                init_gain=mask&(lc[:,ti]==0)&(fc[:,ti]>0);init_loss=mask&(lc[:,ti]>0)&(fc[:,ti]==0)
                gain=mask&~L['native_TP'][:,nativeti]&F['native_TP'][:,nativeti];loss=mask&L['native_TP'][:,nativeti]&~F['native_TP'][:,nativeti]
                row=dict(GT=int(mask.sum()),left_initial_supported=int((mask&(lc[:,ti]>0)).sum()),full_initial_supported=int((mask&(fc[:,ti]>0)).sum()),initial_support_gained=int(init_gain.sum()),initial_support_lost=int(init_loss.sum()),
                    initial_gains_with_full_common_selected_location_support=int((init_gain&np.isin(fc[:,ti],[1,3])).sum()),initial_gains_only_full_unique_selected_location_support=int((init_gain&(fc[:,ti]==2)).sum()),
                    initial_losses_with_left_common_selected_location_support=int((init_loss&np.isin(lc[:,ti],[1,3])).sum()),initial_losses_only_left_unique_selected_location_support=int((init_loss&(lc[:,ti]==2)).sum()),
                    native_TP_gained=int(gain.sum()),native_TP_lost=int(loss.sum()),
                    gained_native_TP_by_full_initial_source={n:int((gain&(fc[:,ti]==i)).sum()) for i,n in enumerate(labels)},lost_native_TP_by_left_initial_source={n:int((loss&(lc[:,ti]==i)).sum()) for i,n in enumerate(labels)},
                    nearby_selection_proxy=dict(left_GT_with_nearby_selected_site=int((mask&near_left).sum()),full_GT_with_nearby_selected_site=int((mask&near_full).sum()),gained_native_TP_with_nearby_selection_gain=int((gain&~near_left&near_full).sum()),lost_native_TP_with_nearby_selection_loss=int((loss&near_left&~near_full).sum())))
                assert row['initial_gains_with_full_common_selected_location_support']+row['initial_gains_only_full_unique_selected_location_support']==row['initial_support_gained']
                assert sum(row['gained_native_TP_by_full_initial_source'].values())==row['native_TP_gained'] and sum(row['lost_native_TP_by_left_initial_source'].values())==row['native_TP_lost']
                rows[group][str(t)]=row
        comparisons[left+'_to_original_stem']=dict(groups=rows,image_selection_intersection=arrays['common_selection_images'],common_selected_locations_mean=float(np.mean([x['common_selected_locations'] for x in arrays['common_selection_images']])),median_common_site_bbox_offset_RMS_difference=float(np.median(offset_delta[left])))
        np.savez_compressed(output/(left+'_partitions.npz'),annotation_id=ids,left_partition=lc,full_partition=fc)
    report=dict(status='SIX_RECORD_MEMBER_SHARED_PHYSICAL_LOCATION_INITIAL_REFERENCE_AND_NATIVE_TP_PARTITIONS',seed=seed,images=548,GT=38759,small_GT=26586,actual_original_TopK_locations=True,all_three_EMA_anchor_buffers_equal=True,
        selected_location_count=300,level_selection_counts={a:dict(P3=int(c[0]),P4=int(c[1]),P5=int(c[2]),sum=int(c.sum())) for a,c in level_counts.items()},comparisons=comparisons,
        source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,root_proven=False,novelty_proven=False,two_seed_analysis_complete=False,
        scope='Physical feature-grid location is shared, not query index or feature content. Identical cached priors plus different selected regression offsets explain different initial boxes at common selected sites algebraically; this does not isolate backbone versus encoder versus regression-head learning. Support partitions are observational GT cohorts, not APs decomposition, unique cause, matched training capacity or counterfactual boxes at unselected sites. NativeTP uses all-area matching stratified by GTsize.')
    assert all(c.sum()==548*300 for c in level_counts.values());assert 'torch' not in sys.modules
    with (output/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    files={x.name:sha(x) for x in output.iterdir() if x.is_file()}
    with (output/'manifest.json').open('x',encoding='utf8') as f:json.dump(dict(files=files),f,indent=2)
    print(json.dumps(dict(status=report['status'],level_counts=report['level_selection_counts'],small={k:v['groups']['COCO_small'] for k,v in comparisons.items()})),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(1,2),required=True);args=parser.parse_args();main(args.seed)
