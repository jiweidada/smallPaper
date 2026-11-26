"""Descriptive capacity audit of an existing exact native augmentation trace."""
from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parents[3]
source=root/'outputs/sod_problem_reasoning_20261010/native_training_geometry_trace_seed1_n64_v3.json'
v=json.loads(source.read_text(encoding='utf-8-sig'))
assert v['status']=='COMPLETE_NATIVE_GEOMETRY_TRACER_EXACT_OUTPUT_EQUIVALENCE'
rows=[]
for s in v['samples']:
    last=s['stages'][-1]
    assert last['stage']=='9:ConvertBoxes' and last['normalized_cxcywh']
    assert len(last['ids'])==len(last['pixel_wh'])
    count=len(last['ids'])
    rows.append(dict(position=s['position'],image_id=s['image_id'],mosaic_used=s['mosaic_used'],pre_collate_GT_rows=count,normal_queries=300,pre_collate_minimum_capacity_excess=max(0,count-300),input640_short_below4_rows=sum(min(w,h)<4 for w,h in last['pixel_wh'])))
counts=np.asarray([r['pre_collate_GT_rows'] for r in rows])
report=dict(status='DESCRIPTIVE_PRECOLLATE_CAPACITY_AUDIT_EXISTING_NATIVE_TRACE',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),epoch_zero_based=v['epoch_zero_based'],seed=v['random_seed'],images=len(rows),normal_queries=300,pre_collate_GT_total=int(counts.sum()),pre_collate_GT_max=int(counts.max()),pre_collate_GT_median=float(np.median(counts)),pre_collate_images_above300=int((counts>300).sum()),pre_collate_minimum_excess_GT_total=sum(r['pre_collate_minimum_capacity_excess'] for r in rows),native_mixup_batches=sum(b['mixup_items']>0 for b in v['batches']),total_batches=len(v['batches']),all_sample_counts=rows,validation_source_precheck=dict(images=548,valid_GT_total=38759,COCO_small_total=26586,maximum_GT=317,images_above300=3,minimum_excess_above300=34,source_annotation_sha256='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd',scope='Read original annotations on authorized server, valid noncrowd clamped boxes; no model execution'),limits='Existing64-image seed1 epoch5 augmentation panel is not actual whole-training history or a representative sampling estimate. Counts are AFTER per-image Mosaic/crop/ConvertBoxes, BEFORE native batch MixUp. Post-MixUp query coverage and size/class identities need actual supervision observation. These counts cannot establish AP benefit or a leading cause; DN, union regression and auxiliary losses can still supervise GT absent from final O2O.',model_forward_calls_added=0,optimizer_steps=0,training_epochs_added=0)
target=root/'outputs/assignment_mechanism_diagnostic_20261010_v1/query_capacity_precheck.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:x for k,x in report.items() if k not in ['all_sample_counts','limits','source']},ensure_ascii=False))
