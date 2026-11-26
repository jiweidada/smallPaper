"""Check actual GT capacity and gain cohorts before proposing more queries or one causal story."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
annpath=ROOT/'outputs/scale_protection12_20261010_v1/terminal_collection/seed2_target_protect_v1/extracted/reference/val_annotations.json';ann=read(annpath)
counts=np.zeros((548,10),np.int64);small=np.zeros_like(counts)
for a in ann['annotations']:
    assert not a.get('iscrowd',0) and not a.get('ignore',0)
    counts[a['image_id'],a['category_id']]+=1
    if a['area']<=1024:small[a['image_id'],a['category_id']]+=1
assert counts.sum()==38759 and small.sum()==26586
global_excess=int(np.maximum(counts.sum(1)-300,0).sum())
class_excess=int(np.maximum(counts-100,0).sum())
images_global=np.flatnonzero(counts.sum(1)>300)
class_cap_details=[dict(image_id=int(i),category_id=int(c),GT=int(counts[i,c]),small_GT=int(small[i,c]),excess_above100=int(counts[i,c]-100)) for i,c in zip(*np.nonzero(counts>100))]
p=ROOT/'outputs/user_backbone_primary_analysis12_20261011_v2/seed2_p2_skip';manifest=read(p/'manifest.json')
for n,h in manifest['files'].items():assert sha(p/n)==h,n
r=read(p/'report.json');out={}
for t in ('0.5','0.75'):
    out[t]={}
    for group in ('COCO_small','COCO_small_input_under4'):
        g=r['outcomes'][t][group];changed=g['gained_TP_first_observed_geometric_support_change'];same=g['gained_TP_unchanged_three_boundary_support']
        assert sum(changed.values())+same==g['native_all_area_TP_gained']
        out[t][group]=dict(GT=g['GT'],gained_current_native_TP=g['native_all_area_TP_gained'],lost_current_native_TP=g['native_all_area_TP_lost'],net_current_native_TP=g['native_all_area_TP_net'],first_observed_binary_geometric_support_change_among_gained_TP=changed,all_three_boundary_support_unchanged_among_gained_TP=same,unchanged_support_fraction_of_gained_TP=same/g['native_all_area_TP_gained'])
report=dict(status='ACTUAL_GT_CAPACITY_AND_GAIN_SCOPE_AUDITED_WITHOUT_MODEL_OR_MATCHING',annotation_sha256=sha(annpath),images=548,GT=38759,small_GT=26586,max_GT_in_image=int(counts.sum(1).max()),images_with_more_GT_than300=int(len(images_global)),GT_in_those_images=int(counts[images_global].sum()),small_GT_in_those_images=int(small[images_global].sum()),unavoidable_GT_excess_if_final_global_detection_budget300=global_excess,image_class_cells_above_native100=len(class_cap_details),unavoidable_GT_excess_if_class100_only=class_excess,image_class_GT_cap_details=class_cap_details,
scope='Budget bounds apply to exact GT count without assuming prediction localization or class accuracy. GlobalTop300 and class100 bounds overlap and must not be added. Fixed native all-area cap is perimage/category. These raw recall bounds do not quantify anAPs contribution, and increased decoder query count alone does not change finalTop300/class100 budgets.',P2_same_GT_gain_scope=out,source_P2_stage_manifest_sha256=sha(p/'manifest.json'),
inference='Do not equate all P2 native TP gains with increased initial-reference support: some first change appears later and some gained GT are already geometrically supported at all three boundaries. Selection, regression, classification, retention and matching remain separate candidates; cross-run query identity is not shared.',root_proven=False,novelty_proven=False,new_model_forwards=0,new_training_epochs=0,new_evaluator_calls=0,new_matching_calls=0,source_sha256=sha(__file__))
dest=ROOT/'outputs/query_budget_and_P2_gain_scope12_20261011_v1';dest.mkdir(exist_ok=False)
with (dest/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
np.savez_compressed(dest/'image_class_GT_counts.npz',GT=counts,COCO_small_GT=small)
with (dest/'manifest.json').open('x',encoding='utf8') as f:json.dump(dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}),f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k in ('status','max_GT_in_image','images_with_more_GT_than300','GT_in_those_images','small_GT_in_those_images','unavoidable_GT_excess_if_final_global_detection_budget300','image_class_cells_above_native100','unavoidable_GT_excess_if_class100_only','P2_same_GT_gain_scope')}),flush=True)
