"""Preserve v1 and fix its default100 summary extraction. No model operations."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
old=ROOT/'run_cached_error_audit_v1.py';planpath=ROOT/'protocol_FIXED_v1.json'
plan=json.loads(planpath.read_text(encoding='utf-8'))
assert sha(old)==plan['source_files_sha256'][str(old)]
assert not (ROOT/'error_audit_result_v1.json').exists()
source=old.read_text(encoding='utf-8')
needle="    metrics={k:float(ev.stats[i]*100) for k,i in {'AP':0,'AP50':1,'AP75':2,'APs':3,'APm':4,'APl':5,'ARs':9}.items()}\n    metrics['APs75']=float(sm75.mean()*100)"
replacement="""    precision=ev.eval['precision'];recall=ev.eval['recall']
    def ap(area=0,iou=None):
        z=precision[:,:,:,area,-1]
        if iou is not None:z=z[np.isclose(ev.params.iouThrs,iou)]
        z=z[z>=0];return float(z.mean()*100) if len(z) else None
    def ar(area=0):
        z=recall[:,:,area,-1];z=z[z>=0];return float(z.mean()*100) if len(z) else None
    metrics={'AP':ap(),'AP50':ap(iou=.5),'AP75':ap(iou=.75),'APs':ap(1),'APs75':ap(1,.75),'APm':ap(2),'APl':ap(3),'ARs':ar(1)}"""
assert source.count(needle)==1
source=source.replace(needle,replacement)
assert source.count('ev.evaluate();ev.accumulate();ev.summarize()')==1
source=source.replace('ev.evaluate();ev.accumulate();ev.summarize()','ev.evaluate();ev.accumulate()')
source=source.replace('_v1','_v2')
target=ROOT/'run_cached_error_audit_v2.py'
with target.open('x',encoding='utf-8') as f:f.write(source)
legacy=ROOT.parent/'A49_下一主线有界筛选_20261101_v1/evaluate.py'
save(ROOT/'v1_metric_extraction_failure.json',{
    'status':'FAILED_METRIC_EXTRACTION_GUARD_NOT_MODEL_NEGATIVE','COCO_evaluate_calls':1,
    'new_model_forwards':0,'new_optimizer_updates':0,
    'source_sha256':sha(old),'protocol_sha256':sha(planpath),
    'cause':'Local cocoeval.summarize stats[0] calls maxDets=100 by default; configured maxDets are1,10,300, so it returns sentinel-1. A49 native extractor directly averages precision at maxDets300.',
    'AP_minus_reference_pp':-124.17823103759345,
    'other_metric_differences':'AP50/AP75/APm/APl/ARs/APs75 exactly0; APs -3.552713678800501e-15pp.',
    'failure_stdout_file_sha256':sha(ROOT/'COCO_evaluation_stdout_v1.txt'),
    'tolerance_not_relaxed':1e-8,'frozen_v1_files_modified':False})
plan.update(revision=2,parent_protocol=str(planpath),parent_protocol_sha256=sha(planpath),
    repair='Use the unchanged A49 precision/recall array extraction; remove default100 summarize call. Matching, groups and thresholds unchanged.',
    prior_failed_metric_extraction_CPU_evaluations=1,
    CPU_budget={'COCO_evaluate_calls_this_revision_max':1,'COCO_evaluate_calls_A53_total_max':2,
        'model_forwards':0,'optimizer_updates':0,'training_epochs':0})
plan['source_files_sha256'].update({str(target):sha(target),str(legacy):sha(legacy)})
save(ROOT/'protocol_FIXED_v2.json',plan)
print(json.dumps({'status':'REPAIR_FROZEN','v2_protocol_sha256':sha(ROOT/'protocol_FIXED_v2.json'),
    'v1_preserved':True,'model_operations':0,'total_CPU_evaluate_cap':2}))
