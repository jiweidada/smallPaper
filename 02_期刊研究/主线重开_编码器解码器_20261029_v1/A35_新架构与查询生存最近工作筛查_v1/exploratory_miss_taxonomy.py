"""Post-screen descriptive audit of native misses; not a registered confirmatory test."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;E=D/'downloaded_evidence'
spec=importlib.util.spec_from_file_location('a35_frozen_metric_helpers',D/'query_retention_analysis.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);h=m.helper_functions()
raw=json.loads((D/'raw_annotations_verified.json').read_text(encoding='utf-8'))
cases=h['prepare_cases'](raw['cases'])
rows=np.load(D/'native_support_rows.npz',allow_pickle=False)
threshold=float(np.load(D/'paired_support_bootstrap.npz',allow_pickle=False)['thresholds'][0])
counts={group:{name:0 for name in ['native_misses','no_raw_box_IoU05','raw_geometry_low_trueclass_weak_foreground','raw_geometry_low_trueclass_strong_otherclass','raw_trueclass_passes_prefix_but_native_missed','raw_best_IoU_within_1e4_of_05']} for group in ['small','larger']}
details=[]
for i,c in enumerate(cases):
    with np.load(E/'captures'/('case%02d.npz'%i),allow_pickle=False) as z:
        b=z['depth3_boxes'][0].astype(np.float64);p=z['depth3_probabilities'][0].astype(np.float64)
    xywh=b.copy();xywh[:,:2]-=xywh[:,2:]/2
    xywh*=np.array([c['width'],c['height'],c['width'],c['height']])
    overlap=h['iou_xywh'](xywh,c['gt_xywh'])
    gt_rows=np.flatnonzero(rows['image']==i)
    for g,j in enumerate(gt_rows):
        if rows['native_TP_score'][j]>=threshold:continue
        group='small' if bool(rows['small'][j]) else 'larger';counts[group]['native_misses']+=1
        valid=overlap[:,g]>=.5;best_iou=float(overlap[:,g].max());label=int(c['gt_labels'][g])
        if abs(best_iou-.5)<=1e-4:counts[group]['raw_best_IoU_within_1e4_of_05']+=1
        if not valid.any():stage='no_raw_box_IoU05';true=-1.;foreground=-1.
        else:
            true=float(p[valid,label].max());foreground=float(p[valid,1:11].max())
            if true>=threshold:stage='raw_trueclass_passes_prefix_but_native_missed'
            elif foreground<threshold:stage='raw_geometry_low_trueclass_weak_foreground'
            else:stage='raw_geometry_low_trueclass_strong_otherclass'
        counts[group][stage]+=1
        details.append(dict(image_index=i,file=c['file'],raw_GT_row=int(c['gt_row_indices'][g]),label=label,group=group,
                            stage=stage,best_raw_IoU=best_iou,best_trueclass_score=true,best_any_validclass_score=foreground))
out=dict(status='POSTHOC_DESCRIPTIVE_NATIVE_MISS_TAXONOMY_ONLY',native_FP_prefix_threshold=threshold,counts=counts,rows=details,
         limits=['Designed after the registered query-retention screen;not confirmatory evidence.',
                'Raw500 boxes give class-agnostic geometric overlap,not evidence that useful semantic information exists or is recoverable.',
                'No unique-matching raw oracle,pruned inference,learned readout or algorithm benefit evaluated.',
                'Normalized raw boxes transformed in float64 for this descriptive audit;native metric uses float32 postprocessing;near-boundary count disclosed.',
                'Weak foreground score does not prove background interference,encoder information loss or classifier bias.'],new_model_calls=0)
with (D/'exploratory_miss_taxonomy_result.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=out['status'],threshold=threshold,counts=counts),ensure_ascii=False))
