import json
import pathlib
import numpy as np
from scipy.optimize import linear_sum_assignment

root = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012/artifacts/round2_20261014_v1/addons_v2/prelaunch_four_images')
analysis = json.loads((root/'analysis_cli_vs_fused_dump/analysis.json').read_text())
metrics = analysis['original_coordinate_appendix']
left = metrics['reference']['coco_original_coordinate']['metrics_raw_0_to_1']
right = metrics['candidate']['coco_original_coordinate']['metrics_raw_0_to_1']
first = json.loads((root/'test_only_eval/pred_bbox.json').read_text())
second = json.loads((root/'dump_fused/predictions.json').read_text())
keys = sorted({(row['image_id'],row['category_id']) for row in first+second})
maximum_box_difference=0.;maximum_score_difference=0.
groups=[]
for image,category in keys:
    a=np.asarray([[*row['bbox'],row['score']] for row in first if row['image_id']==image and row['category_id']==category])
    b=np.asarray([[*row['bbox'],row['score']] for row in second if row['image_id']==image and row['category_id']==category])
    assert len(a)==len(b)
    if not len(a):continue
    cost=np.abs(a[:,None,:4]-b[None,:,:4]).sum(-1)+np.abs(a[:,None,4]-b[None,:,4])*1000
    i,j=linear_sum_assignment(cost)
    box=float(np.max(np.abs(a[i,:4]-b[j,:4])));score=float(np.max(np.abs(a[i,4]-b[j,4])))
    maximum_box_difference=max(maximum_box_difference,box);maximum_score_difference=max(maximum_score_difference,score)
    groups.append({'image_id':image,'category_id':category,'n':len(a),'max_box_difference':box,'max_score_difference':score})
print(json.dumps({'reference_metrics':left,'candidate_metrics':right,'metrics_max_abs_difference':max(abs(a-b) for a,b in zip(left,right)),
                  'assignment_max_box_difference':maximum_box_difference,'assignment_max_score_difference':maximum_score_difference,
                  'custom_bin_AP_delta_pp':analysis['AP']['paired_delta_pp'],'groups':groups},indent=2))
