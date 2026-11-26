"""Freeze one descriptive cached-output audit. No model or server operations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent
A52=BASE/'A52_不同机制候选有界筛选_20261101_v1'
A49=BASE/'A49_下一主线有界筛选_20261101_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    target=ROOT/'protocol_FIXED_v1.json'; assert not target.exists()
    current=load(A52/'CURRENT_A52.json'); receipt=Path(current['receipt'])
    assert sha(receipt)==current['receipt_sha256']=='b2bd6bca56e1b1a793ee86e3715d4e4c04ef5871789ed51b7feb1b55367eebfb'
    prior=load(receipt); assert not prior['pilot_pass'] and not prior['owned_jobs_live']
    assert all(sha(Path(p))==h for p,h in prior['current_files_sha256'].items())
    archive=ROOT/'before_A53'; archive.mkdir(exist_ok=False)
    preserved={}
    for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        p=BASE/name; dest=archive/name; shutil.copy2(p,dest)
        assert sha(dest)==prior['current_files_sha256'][str(p)]
        preserved[str(p)]={'preserved_path':str(dest),'sha256':sha(dest)}
    shutil.copy2(A52/'CURRENT_A52.json',archive/'CURRENT_A52_before_pointer_refresh.json')
    current.update(current_priority_candidate=None,bridge_session_if_alive=None,
        latest_observation=str(A52/'observer_65_terminal_exit.json'),latest_verified_interim='FIXED12_TERMINAL',
        M_observed_optimizer_update_lower_bound=9708,Q_observed_optimizer_update_lower_bound=9708,
        Q_recorded_flow_verified_through_epoch=12,
        next='A52 retired; A53 starts a descriptive baseline cached-output audit. No new training queued.')
    (A52/'CURRENT_A52.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[A49/'completed_B_v1/arm_B'/n for n in ['epoch12_EMA_GT.json','epoch12_EMA_predictions.json','epoch12_EMA_metrics.json']]
    files += [A49/'val_data_manifest.json',ROOT/'run_cached_error_audit_v1.py']
    dep=A49/'postprocess_ignore_strata/deps/pycocotools'
    files += [dep/n for n in ['coco.py','cocoeval.py','mask.py','_mask.pyd','__init__.py']]
    plan={
        'phase':'A53','status':'FIXED_DESCRIPTIVE_AUDIT_NO_CANDIDATE',
        'created_local_utc':datetime.now(timezone.utc).isoformat(),
        'goal_status':'active','mainline_confirmed':False,
        'question':'Which geometric/class/output-coverage deficits remain in the fixed12 native B output?',
        'timing':'After B/M/Q12 outcomes were known; exploratory description, not an outcome-blind preregistration.',
        'previous_terminal_receipt':str(receipt),'previous_terminal_receipt_sha256':sha(receipt),
        'previous_signed_inputs_preserved':preserved,
        'source_files_sha256':{str(p):sha(p) for p in files},
        'endpoint':'A49 B fixedepoch12 EMA only; cached ignore-filtered predictions and its exact scored GT.',
        'CPU_budget':{'COCO_evaluate_calls_max':1,'model_forwards':0,'optimizer_updates':0,'training_epochs':0},
        'evaluation':'pycocotools2.0.11 retained local package; original clipped GT area; maxDets1,10,300; full548 metric agreement<=1e-8pp.',
        'matching':'Native score-ordered class-aware one-to-one COCO matching at IoU0.50 and0.75, area=all, maxDet300.',
        'unmatched50_disjoint_description':[
            'A retained correct-class box has IoU>=0.5 but no COCO one-to-one match',
            'No such correct-class box; a retained other-class box has IoU>=0.5',
            'No retained box has IoU>=0.5; maximum any-class IoU is in[0.1,0.5)',
            'Maximum any-class IoU is<0.1, including no retained box'],
        'groups':{
            'native_small':'clipped original GT area<=1024, not input-canvas area',
            'input_min_side':'min(GTwidth*640/imagewidth,GTheight*640/imageheight); fixed bins<2,[2,4),[4,8),>=8 pixels',
            'same_class_GT_overlap':'max IoU with any other scored same-class GT>=0.1; descriptive overlap, not full occlusion/crowding labels',
            'class':'all ten scored categories, without choosing favorable classes'},
        'limits':[
            'All retained final boxes are used without an extra confidence threshold.',
            'Existence of a geometrically compatible box is an idealized coverage bound, not an attainable AP gain.',
            'No query IDs, full pre-topK pool, intermediate features or FDR bins are available in this cache.',
            'Missing final coverage does not locate a cause in the input, encoder, proposal selection, decoder, supervision or ranking.',
            'Group differences are observational and may reflect size/class/image composition.',
            'No candidate selected, no training queue, no parameter/threshold sweep; A52 primary decision is unchanged.'],
        'cumulative_counts_unchanged':{'whole_model_forwards':185612,'formal_optimizer_updates':155854,'capacity_updates':49,'all_optimizer_updates':155903,'formal_epochs':192}
    }
    save(target,plan)
    save(ROOT/'00_A53开始与历史保留.json',{'status':'PASS','protocol_sha256':sha(target),
        'previous_receipt':str(receipt),'previous_receipt_sha256':sha(receipt),
        'preserved':preserved,'unsigned_A52_pointer_corrected_only':True,
        'canonical_files_modified':False,'new_model_forwards':0,'new_optimizer_updates':0})
    print(json.dumps({'status':'FROZEN','protocol_sha256':sha(target),'new_model_operations':0}))
if __name__=='__main__':main()
