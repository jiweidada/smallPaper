from pathlib import Path
import importlib.util,json,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;E=D/'downloaded_evidence'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('a35_helpers',D/'query_retention_analysis.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases=m.helper_functions()['prepare_cases'](read(D/'raw_annotations_verified.json')['cases'])
data=read(D/'exploratory_miss_taxonomy_result.json');threshold=data['native_FP_prefix_threshold']
counts={k:{key:0 for key in v} for k,v in data['counts'].items()};checked=pairs=0;max_error=0.
for i,c in enumerate(cases):
    records=[r for r in data['rows'] if r['image_index']==i]
    byraw={int(x):j for j,x in enumerate(c['gt_row_indices'])}
    with np.load(E/'captures'/('case%02d.npz'%i),allow_pickle=False) as z:
        b=z['depth3_boxes'][0].astype(float);p=z['depth3_probabilities'][0].astype(float)
    for r in records:
        g=byraw[r['raw_GT_row']];gt=c['gt_xywh'][g]/np.array([c['width'],c['height'],c['width'],c['height']])
        best_iou=0.;best_true=best_fg=-1.
        for box,prob in zip(b,p):
            x,y,w,h=float(box[0]-box[2]/2),float(box[1]-box[3]/2),float(box[2]),float(box[3])
            iw=max(0.,min(x+w,gt[0]+gt[2])-max(x,gt[0]));ih=max(0.,min(y+h,gt[1]+gt[3])-max(y,gt[1]))
            inter=iw*ih;union=w*h+gt[2]*gt[3]-inter;value=inter/union if union>0 else 0.;pairs+=1
            best_iou=max(best_iou,value)
            if value>=.5:best_true=max(best_true,float(prob[r['label']]));best_fg=max(best_fg,max(map(float,prob[1:11])))
        if best_iou<.5:stage='no_raw_box_IoU05'
        elif best_true>=threshold:stage='raw_trueclass_passes_prefix_but_native_missed'
        elif best_fg<threshold:stage='raw_geometry_low_trueclass_weak_foreground'
        else:stage='raw_geometry_low_trueclass_strong_otherclass'
        assert stage==r['stage'];assert best_true==r['best_trueclass_score'] and best_fg==r['best_any_validclass_score']
        max_error=max(max_error,abs(best_iou-r['best_raw_IoU']));assert abs(best_iou-r['best_raw_IoU'])<1e-10
        group=r['group'];counts[group]['native_misses']+=1;counts[group][stage]+=1
        if abs(best_iou-.5)<=1e-4:counts[group]['raw_best_IoU_within_1e4_of_05']+=1
        checked+=1
assert counts==data['counts'] and checked==620
receipt=dict(status='PASS_SCALAR_NORMALIZED_COORDINATE_POSTHOC_TAXONOMY',missed_GT_rows_checked=checked,
             scalar_query_GT_IoU_pairs=pairs,max_raw_IoU_error=max_error,new_model_calls=0,
             scope='Descriptive taxonomy numerical check;not a registered hypothesis test or proof of useful semantic information.')
with (D/'independent_taxonomy_verification.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt))
