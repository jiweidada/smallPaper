"""Join actual primary12 geometry and native-score-checked final ranking; no replay claim."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PRIMARY=ROOT/'outputs/stem_primary_collection_20261011_v1'
OUT=ROOT/'outputs/stem_primary_joint_analysis12_20261011_v1'
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
tools=module('primary_joint_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
p,geometry,onset,rank,readers=tools.prepared()
joint_path=ROOT/'英文小论文/code/stem_joint_error_analysis_20261011_v2'
joint_protocol=read(joint_path/'protocol.json')
assert sha(joint_path/'analyze.py')==joint_protocol['analysis_sha256']
joint=module('primary_joint_partition',joint_path/'analyze.py')
assert list(joint.LABELS)==joint_protocol['labels']
# Validate both primary packets before creating this analysis output.
evidence={};bases={}
for seed in (1,2):
    folder=PRIMARY/f'seed{seed}_original_stem_v1'
    verification=read(folder/'primary_verification_v1.json')
    assert verification['status']=='PASS_DOWNLOADED_EXACT12_PRIMARY_ARTIFACTS_ONLY'
    assert verification['identity']['seed']==seed and verification['identity']['arm']=='original_stem'
    assert verification['actual_stage_images']==548 and verification['all605_collected_files_verified']
    assert not verification['saved_EMA_independent_replay_verified'] and not verification['formal_terminal_reader_passed']
    verifier=ROOT/f'outputs/verify_stem_primary_seed{seed}_20261011_v1.py'
    assert sha(verifier)==verification['verification_script_sha256']
    packet=Path(verification['downloaded_packet']);manifest=read(packet/'primary_manifest.json')
    assert sha(packet/'primary_manifest.json')==verification['primary_manifest_sha256']
    assert len(manifest['files'])==605
    for n,v in manifest['files'].items():
        assert (packet/n).stat().st_size==v['bytes'] and sha(packet/n)==v['sha256'],n
    bases[seed]=tools.cached_baseline(seed,p,readers['full'])
    base=bases[seed];identity=verification['identity']
    for n in ('common_init_sha256','baseline_checkpoint_sha256','val_annotation_sha256','evaluator_binding'):
        assert identity[n]==base['summary']['identity'][n],n
    summary,flags,_=readers['full'].native_endpoint(packet/'run/endpoint_epoch12',base['annotation'])
    assert summary['identity']==identity and summary['native_stats']==verification['native_stats']
    readers['full'].stages(packet/'run',identity,base['annotation'])
    evidence[seed]=dict(folder=folder,packet=packet,verification=verification,summary=summary,flags=flags)
assert not OUT.exists()
OUT.mkdir(parents=True)
results=[];source_checks={}
for seed in (1,2):
    e=evidence[seed];packet=e['packet'];base=bases[seed]
    dest=OUT/f'seed{seed}_original_stem';dest.mkdir()
    def image(iid):
        with np.load(packet/f'run/stages_epoch12/image_{iid:06d}.npz',allow_pickle=False) as z:
            return {n:z[n].copy() for n in z.files}
    stage=geometry.collect(base['annotation'],image,e['flags'])
    for n in ('annotation_id','image_id','category_id','original_area'):
        assert np.array_equal(stage[n],base['values'][n])
    rank.analyze(packet,'original_stem',tools.HERE/'ranking_protocol.json',dest/'ranking',
                 base['packet']/'native/engine/deim/postprocessor.py',base['packet']/'reference/val_annotations.json')
    rr=read(dest/'ranking/report.json')
    assert rr['identity']==e['verification']['identity']
    with np.load(dest/'ranking/per_GT.npz',allow_pickle=False) as z:
        ranking={n:z[n].copy() for n in z.files}
    rank_folder=ROOT/p['baseline_caches'][str(seed)]['rank']
    with np.load(rank_folder/'per_GT.npz',allow_pickle=False) as z:
        base_ranking={n:z[n].copy() for n in z.files}
    for n in ('annotation_id','image_id','category_id','original_area'):
        assert np.array_equal(stage[n],ranking[n]) and np.array_equal(stage[n],base_ranking[n])
    masks=tools.groups(base['annotation'],stage)
    side=ranking['input_shortside']
    assert np.array_equal(masks['input_under4'],side<4)
    masks['COCO_small_input_under4']=masks['COCO_small']&(side<4)
    categories={int(c['id']):c['name'] for c in base['annotation']['categories']}
    for cid in sorted(categories):
        masks['COCO_small_class_'+str(cid)+'_'+categories[cid]]=masks['COCO_small']&(stage['category_id']==cid)
    counts={};transitions={};geometry_pairs={}
    joint_arrays={n:stage[n] for n in ('annotation_id','image_id','category_id','original_area')}
    for threshold,ti,tag in ((.5,0,'50'),(.75,5,'75')):
        label,pattern=joint.partition(stage,ranking,threshold)
        blabel,bpattern=joint.partition(base['values'],base_ranking,threshold)
        joint_arrays['joint_label_'+tag]=label;joint_arrays['BASE_joint_label_'+tag]=blabel
        joint_arrays['support_pattern_'+tag]=pattern;joint_arrays['BASE_support_pattern_'+tag]=bpattern
        support=stage['max_iou']>=threshold;bsupport=base['values']['max_iou']>=threshold
        changed=support!=bsupport;any_change=changed.any(1);first=changed.argmax(1)
        counts[str(threshold)]={};transitions[str(threshold)]={};geometry_pairs[str(threshold)]={}
        for name,mask in masks.items():
            counts[str(threshold)][name]=dict(
                BASE=joint.summarize(mask,blabel,bpattern,base_ranking,threshold),
                enhanced_primary=joint.summarize(mask,label,pattern,ranking,threshold))
            matrix=np.zeros((len(joint.LABELS),len(joint.LABELS)),dtype=np.int64)
            np.add.at(matrix,(blabel[mask],label[mask]),1)
            assert matrix.sum()==mask.sum()
            lost=mask&base['values']['native_TP'][:,ti]&~stage['native_TP'][:,ti]
            gained=mask&~base['values']['native_TP'][:,ti]&stage['native_TP'][:,ti]
            lost_counts={n:int((lost&(label==i)).sum()) for i,n in enumerate(joint.LABELS)}
            assert lost_counts['native_TP']==0 and sum(lost_counts.values())==lost.sum()
            transitions[str(threshold)][name]=dict(
                GT=int(mask.sum()),BASE_to_primary_joint_cell_counts=matrix.tolist(),
                BASE_native_TP_lost=int(lost.sum()),primary_native_TP_gained=int(gained.sum()),
                lost_BASE_native_TP_primary_outcome=lost_counts,
                lost_TP_first_changed_observed_support={
                    n:int((lost&any_change&(first==i)).sum()) for i,n in enumerate(geometry.STAGES)},
                lost_TP_no_observed_support_change=int((lost&~any_change).sum()))
            row=geometry.pair_metrics(base['values'],stage,mask);row.pop('transition_annotation_ids')
            geometry_pairs[str(threshold)][name]=row
    np.savez_compressed(dest/'stage_per_GT.npz',**stage)
    np.savez_compressed(dest/'joint_per_GT.npz',**joint_arrays)
    row=dict(seed=seed,native_primary_AP_percent={n:100*e['summary']['native_stats'][i] for n,i in (('AP',0),('APs',3))},
       cached_BASE12_AP_percent={n:100*base['summary']['native_stats'][i] for n,i in (('AP',0),('APs',3))},
       delta_BASE_AP_pp={n:100*(e['summary']['native_stats'][i]-base['summary']['native_stats'][i]) for n,i in (('AP',0),('APs',3))},
       all548_images=True,all_valid_GT=len(stage['annotation_id']),COCO_small_GT=int(masks['COCO_small'].sum()),
       counts=counts,transitions=transitions,geometry_pairs=geometry_pairs,
       ranking_score_mismatch_images=rr['score_mismatch_images'],ranking_ambiguous_images=rr['ambiguous_images'],
       native_TP_proxy_conflicts=rr['native_TP_proxy_conflicts'],
       native_score_list_max_abs_error=rr['native_score_list_max_abs_error'])
    write(dest/'description.json',row);results.append(row)
    source_checks[str(seed)]=dict(primary_verification_sha256=sha(e['folder']/'primary_verification_v1.json'),
      primary_manifest_sha256=sha(packet/'primary_manifest.json'),
      primary_stage_manifest_sha256=sha(packet/'run/stages_epoch12/manifest.json'),
      primary_endpoint_manifest_sha256=sha(packet/'run/endpoint_epoch12/manifest.json'),
      cached_BASE_stage_manifest_sha256=base['cache_stage_manifest_sha256'],
      cached_BASE_rank_manifest_sha256=base['cache_rank_manifest_sha256'])
review=any(r['ranking_score_mismatch_images'] or r['native_TP_proxy_conflicts'] for r in results)
report=dict(status='ANALYZED_BOTH_PRIMARY_FIXED12_JOINT_PARTITION_REPLAY_PENDING' if not review else 'PRIMARY_JOINT_PARTITION_SCORE_REVIEW_REQUIRED',
  seeds=results,labels=list(joint.LABELS),provenance=source_checks,
  analysis_sha256=sha(__file__),prepared_analysis_protocol_sha256=sha(tools.PROTOCOL),
  joint_partition_protocol_sha256=sha(joint_path/'protocol.json'),
  saved_EMA_independent_replay_verified=False,formal_terminal_reader_passed=False,
  verification_status='ANALYZED',new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
  new_training_epochs=0,two_seed_Stem_improvement_proven=False,causal_root_proven=False,
  scope='Both completed training-native EMA endpoints; exact same-GT geometry/ranking joins, all548 images and all38759 valid GT, not independent saved-weight replay. Native score list consistency checked; cutoff/tie uncertainty retained. Three observed boundaries and final true-class ranking describe propagation and selection outcomes, not dense8400 regressed-box coverage, Hungarian training history, AP decomposition or a unique Stem/encoder/TopK cause. BASE remains existing cached tensors; no new BASE training/forwards.')
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
write(OUT/'report.json',report)
write(OUT/'manifest.json',dict(files={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file()}))
print(json.dumps(dict(status=report['status'],seeds=[
 dict(seed=r['seed'],AP=r['native_primary_AP_percent'],small50=r['counts']['0.5']['COCO_small'],
      lost_small_TP50=r['transitions']['0.5']['COCO_small'],under4_50=r['counts']['0.5']['input_under4'],
      score_mismatch_images=r['ranking_score_mismatch_images'],ambiguous_images=r['ranking_ambiguous_images'],
      native_TP_proxy_conflicts=r['native_TP_proxy_conflicts']) for r in results]),ensure_ascii=False))

