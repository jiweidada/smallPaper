"""Describe one actual completed primary12 run; saved-EMA replay is still pending."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PRIMARY=ROOT/'outputs/stem_primary_collection_20261011_v1/seed2_original_stem_v1'
OUT=ROOT/'outputs/stem_primary_stage_analysis_20261011_v1/seed2_original_stem'
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
verification=read(PRIMARY/'primary_verification_v1.json')
assert verification['status']=='PASS_DOWNLOADED_EXACT12_PRIMARY_ARTIFACTS_ONLY'
assert verification['identity']['seed']==2 and verification['identity']['arm']=='original_stem'
assert verification['actual_stage_images']==548 and not verification['saved_EMA_independent_replay_verified']
packet=Path(verification['downloaded_packet'])
assert sha(packet/'primary_manifest.json')==verification['primary_manifest_sha256']
for n,m in read(packet/'primary_manifest.json')['files'].items():assert sha(packet/n)==m['sha256'],n
main=module('primary_paired_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
p,geometry,onset,rank,readers=main.prepared()
base=main.cached_baseline(2,p,readers['full'])
annotation=base['annotation'];identity=verification['identity']
for n in ('common_init_sha256','baseline_checkpoint_sha256','val_annotation_sha256','evaluator_binding'):
    assert identity[n]==base['summary']['identity'][n],n
summary,flags,_=readers['full'].native_endpoint(packet/'run/endpoint_epoch12',annotation)
assert summary['identity']==identity and summary['native_stats']==verification['native_stats']
readers['full'].stages(packet/'run',identity,annotation)
def image(iid):
    with np.load(packet/f'run/stages_epoch12/image_{iid:06d}.npz',allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
values=geometry.collect(annotation,image,flags)
for n in ('annotation_id','image_id','category_id','original_area'):assert np.array_equal(values[n],base['values'][n])
masks=main.groups(annotation,values)
assert len(values['annotation_id'])==38759 and masks['COCO_small'].sum()==26586
OUT.mkdir(parents=True,exist_ok=False)
np.savez_compressed(OUT/'primary_per_GT.npz',**values)
comparisons={};descriptions={};changes={}
for group,mask in masks.items():
    row=geometry.pair_metrics(base['values'],values,mask);row.pop('transition_annotation_ids')
    comparisons[group]=row
    descriptions[group]={str(t):dict(BASE=onset(base['values'],mask,t,ti),primary=onset(values,mask,t,ti))
                         for t,ti in ((.5,0),(.75,5))}
    changes[group]={}
    for t,ti in ((.5,0),(.75,5)):
        lost_tp=mask&base['values']['native_TP'][:,ti]&~values['native_TP'][:,ti]
        sa=base['values']['max_iou']>=t;sb=values['max_iou']>=t
        changed=sa!=sb;any_changed=changed.any(1);earliest=changed.argmax(1)
        no_any=~sb.any(1);lost_final=sb[:,:2].any(1)&~sb[:,2];final_supported=sb[:,2]
        cells=dict(native_TP_lost=int(lost_tp.sum()),
            lost_TP_no_support_at_any_three_boundaries=int((lost_tp&no_any).sum()),
            lost_TP_earlier_support_lost_final=int((lost_tp&lost_final).sum()),
            lost_TP_final_geometry_present=int((lost_tp&final_supported).sum()),
            lost_TP_initial_support_lost=int((lost_tp&sa[:,0]&~sb[:,0]).sum()),
            lost_TP_first_changed_observed_boundary={stage:int((lost_tp&any_changed&(earliest==i)).sum()) for i,stage in enumerate(geometry.STAGES)},
            lost_TP_no_observed_support_pattern_change=int((lost_tp&~any_changed).sum()))
        assert cells['native_TP_lost']==sum(cells[n] for n in ('lost_TP_no_support_at_any_three_boundaries','lost_TP_earlier_support_lost_final','lost_TP_final_geometry_present'))
        assert cells['native_TP_lost']==cells['lost_TP_no_observed_support_pattern_change']+sum(cells['lost_TP_first_changed_observed_boundary'].values())
        changes[group][str(t)]=cells
delta={name:100*(summary['native_stats'][i]-base['summary']['native_stats'][i]) for i,name in ((0,'AP'),(3,'APs'))}
report=dict(status='ANALYZED_PRIMARY_NATIVE_FIXED12_STAGE_DESCRIPTION_REPLAY_PENDING',
    seed=2,arm='original_stem',all548_images=True,all_valid_GT=38759,COCO_small_GT=26586,
    native_primary_AP_percent={'AP':summary['native_stats'][0]*100,'APs':summary['native_stats'][3]*100},
    cached_BASE12_AP_percent={'AP':base['summary']['native_stats'][0]*100,'APs':base['summary']['native_stats'][3]*100},
    delta_AP_pp=delta,comparisons=comparisons,descriptions=descriptions,lost_native_TP_boundaries=changes,
    primary_verification_sha256=sha(PRIMARY/'primary_verification_v1.json'),
    primary_stage_manifest_sha256=sha(packet/'run/stages_epoch12/manifest.json'),
    cached_BASE_stage_analysis_manifest_sha256=base['cache_stage_manifest_sha256'],
    cached_BASE_source_report_sha256=base['source_report_sha256'],analysis_sha256=sha(__file__),
    verification_status='ANALYZED',saved_EMA_independent_replay_verified=False,causal_root_proven=False,
    new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
    scope='One completed seed2 native primary validation, not independent saved-weight verification or two-seed efficacy. Same-GT annotation/image/category/area joins; no cross-model query identity. Three geometric boundaries describe where coverage differs, not a unique Stem/encoder/TopK root. Final geometry without native TP combines class scores, ranking, duplicates and greedy matching. TP counts are not AP decomposition; COCO-small and input-shortside-under4 cohorts remain separate.')
write(OUT/'report.json',report)
write(OUT/'manifest.json',dict(files={f.name:sha(f) for f in OUT.iterdir() if f.is_file()}))
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
print(json.dumps(dict(status=report['status'],native_AP=report['native_primary_AP_percent'],delta=delta,
    small_IoU50=descriptions['COCO_small']['0.5'],lost_small_native_TP50=changes['COCO_small']['0.5']),ensure_ascii=False))

