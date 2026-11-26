"""Descriptive audit of existing annotation numbers only.

No image read, augmentation execution, model import, stochastic simulation,
training, inference, or evaluation. Source annotations are never modified.
"""
from pathlib import Path
from collections import defaultdict
import hashlib, json, math

root=Path(__file__).resolve().parent
source=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\相位机制论文_20261018\修订_20261018_v2\data\annotations\train\converted.json')
raw=source.read_bytes()
data=json.loads(raw)
images={row['id']:row for row in data['images']}
classes={row['id']:row['name'] for row in data['categories']}
assert len(images)==6471 and len(data['annotations'])==343205
assert len(classes)==10
scales=[0.5,1.0,1.5]
epsilon=0.001

def empty():
    return {'gt':0,'input_area_lt_1024':0,'native_min_side_le_2':0,
            'size_gate_only':{str(s):{'le_2':0,'strict_below_2_minus_eps':0,'le_2_plus_eps':0,'boundary_band':0} for s in scales},
            'sum_analytic_probability':0.0,'images_with_min_side_le_2':set()}

total=empty(); per_class=defaultdict(empty)
invalid=0; outside=0
for annotation in data['annotations']:
    x,y,w,h=map(float,annotation['bbox'])
    image=images[annotation['image_id']]
    iw,ih=int(image['width']),int(image['height'])
    if not all(math.isfinite(v) for v in [x,y,w,h]) or w<=0 or h<=0 or iw<=0 or ih<=0:
        invalid+=1;continue
    if x < -0.01 or y < -0.01 or x+w > iw+0.01 or y+h > ih+0.01:
        outside+=1
    r=640/max(iw,ih)
    rw=min(math.ceil(iw*r),640) if r!=1 else iw
    rh=min(math.ceil(ih*r),640) if r!=1 else ih
    pw,ph=w*rw/iw,h*rh/ih
    a=min(pw,ph)
    # Analytic probability under a declared uniform scale on [.5,1.5],
    # ignoring clipping and all other filtering; no random draws are made.
    probability=max(0.0,min(1.0,2.0/a-0.5))
    for bucket in [total,per_class[annotation['category_id']]]:
        bucket['gt']+=1
        bucket['input_area_lt_1024']+=pw*ph < 1024
        bucket['native_min_side_le_2']+=min(w,h)<=2
        bucket['sum_analytic_probability']+=probability
        if a<=2:
            bucket['images_with_min_side_le_2'].add(annotation['image_id'])
        for s in scales:
            value=a*s
            item=bucket['size_gate_only'][str(s)]
            item['le_2']+=value<=2.0
            item['strict_below_2_minus_eps']+=value<2.0-epsilon
            item['le_2_plus_eps']+=value<=2.0+epsilon
            item['boundary_band']+=abs(value-2.0)<=epsilon

def finish(bucket):
    result={k:v for k,v in bucket.items() if k!='images_with_min_side_le_2'}
    result['images_with_min_side_le_2']=len(bucket['images_with_min_side_le_2'])
    result['analytic_mean_probability_no_clipping']=result['sum_analytic_probability']/max(1,result['gt'])
    for row in result['size_gate_only'].values():
        row['percent_of_gt']=100*row['le_2']/max(1,result['gt'])
    return result

result={
 'date':'2026-10-28',
 'analysis_type':'descriptive annotation arithmetic; no image/model/augmentation experiment',
 'source_path':str(source),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),
 'source_kind':'existing COCO-format conversion of train annotations; not newly obtained original integer annotation files',
 'images':len(images),'annotations':len(data['annotations']),'classes':classes,
 'geometry':'long side 640 with ceil dimensions as static BaseDataset.load_image rule; bbox dimensions rescaled only; no clipping/cropping/rotation',
 'box_candidate_condition':'w>2 and h>2 in fixed-version RandomPerspective.box_candidates',
 'counterfactual_scales_not_run_augmentations':scales,
 'rounding_sensitivity_epsilon_input_pixels':epsilon,
 'overall':finish(total),
 'per_class':{classes[key]:finish(value) for key,value in sorted(per_class.items())},
 'invalid_boxes':invalid,'boxes_outside_native_image_by_more_than_0_01px':outside,
 'limits':[
   'These are potential size-gate exposure counts, not observed dropped-label rates in a real training dataloader.',
   'Mosaic selection, clipping, area/aspect filtering, and caches were not executed.',
   'Converted coordinates are slightly rounded; epsilon-band counts expose threshold sensitivity.',
   'No conclusion about visual recognizability, model AP, harmful gradient magnitude, or novelty follows from counts alone.',
   'The source belongs to a prior project directory but only common annotation numbers were reused; no old model or method was resumed.'
 ],
 'execution':{'image_reads':0,'image_encodes':0,'augmentations_run':0,'random_simulations':0,'model_imports':0,'training_runs':0,'model_forwards':0,'test_dev_reads':0,'remote_jobs':0}
}
(root/'annotation_size_metadata_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'images':len(images),'gt':len(data['annotations']),'overall':result['overall'],
                  'class_exposure_at_scale_1':{k:v['size_gate_only']['1.0']['le_2'] for k,v in result['per_class'].items()},
                  'execution':result['execution']},ensure_ascii=True))
