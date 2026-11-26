"""Freeze a secondary saved-prediction analysis; no model/torch import or training."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'postprocess_ignore_strata'
OUT.mkdir(exist_ok=True)
target=OUT/'protocol_FIXED_v1.json'
assert not target.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
audit=read(ROOT/'annotations_audit_v1'/'image_rows.json')['val']
gtpath=ROOT/'completed_B_v1'/'arm_B'/'epoch12_EMA_GT.json';gt=read(gtpath)
byname={r['file']:r for r in audit}
assert len(byname)==len(gt['images'])==548
groups={'with_explicit_ignore':[],'without_explicit_ignore':[]}
for im in gt['images']:
    r=byname[im['file_name']]
    name='with_explicit_ignore' if r['ignored_rectangles']>0 else 'without_explicit_ignore'
    groups[name].append(im['id'])
assert len(groups['with_explicit_ignore'])==340 and len(groups['without_explicit_ignore'])==208
assert not set(groups['with_explicit_ignore'])&set(groups['without_explicit_ignore'])
assert set(sum(groups.values(),[]))=={x['id'] for x in gt['images']}
obs_path=ROOT/'observer_36_before_secondary_plan.json';obs=json.loads(read(obs_path)['stdout'])
assert not obs['arms']['M']['evaluations'] and obs['arms']['Q']['progress'] is None
rows={}
for name,ids in groups.items():
    allowed=set(ids);anns=[a for a in gt['annotations'] if a['image_id'] in allowed]
    rows[name]={'images':len(ids),'evaluated_GT':len(anns),'evaluated_small_GT_area_le1024':sum(a['area']<=1024 for a in anns)}
protocol={
    'status':'FIXED_SECONDARY_ANALYSIS_NOT_PRIMARY_GATE','created_local_utc':datetime.now(timezone.utc).isoformat(),
    'purpose':'Describe where a fixed12 effect occurs using saved predictions. Does not rescue a failed primary gate or establish a causal mechanism.',
    'timing_scope':'After B fixed12 was known; latest inspected observation had M epoch3, no M validation metric, and Q unstarted. Not a preregistration before all data/outcomes.',
    'latest_inspected_observation':str(obs_path),'latest_inspected_observation_sha256':sha(obs_path),
    'selection':'Image-level only, explicit clipped class0 rectangle count>0 from the already frozen complete annotation audit. No distance threshold, outcome-selected images, size threshold search or GT object subset deletion.',
    'groups_image_ids':groups,'group_counts':rows,
    'endpoint':'Only each arm fixedepoch12EMA saved GT/predictions; no best epoch.',
    'metrics':['AP','AP50','AP75','APs','APs75','APm','APl','ARs'],
    'evaluator':'Same pycocotools2.0.11, maxDets[1,10,300], native COCO area/IoU ranges and original clipped GT area; stored class0 filtering retained. Re-evaluate each image group with all its GT.',
    'verification':'For each arm, first recalculate full548-image metrics from saved predictions and require agreement with cached metrics within1e-8pp. Do not call this rerunning the detector.',
    'CPU_budget':{'COCO_evaluate_calls_per_arm_max':3,'COCO_evaluate_calls_total_max':9,'whole_model_forwards':0,'optimizer_updates':0},
    'scope_limits':['Exploratory secondary interpretation, not causal mediation or subgroup novelty.','Different groups have different image/class composition. Within-arm group gaps are not causal treatment effects.','Any subgroup gain cannot overturn the original fixed12 primary gates.','No new model inference, training, threshold tuning or additional training queue.'],
    'primary_training_protocol':str(ROOT/'protocol_FROZEN.json'),'primary_training_protocol_sha256':sha(ROOT/'protocol_FROZEN.json'),
    'source_files_sha256':{str(p):sha(p) for p in [ROOT/'annotations_audit_v1'/'image_rows.json',gtpath,ROOT/'evaluate.py']},
    'dependency_plan':'Isolated local package files only; match remote pycocotools2.0.11. Initial pip download was explicitly stopped with exit1 after no output; no global package install.'
}
with target.open('x',encoding='utf-8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'FROZEN','groups':rows,'model_forwards':0,'optimizer_updates':0,'protocol_sha256':sha(target)},ensure_ascii=False))
