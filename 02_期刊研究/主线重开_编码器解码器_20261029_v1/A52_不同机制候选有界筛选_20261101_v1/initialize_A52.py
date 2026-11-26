from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
A51=BASE/'A51_跨分辨率守恒交换_20261101_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
receipt=A51/'terminal_A51_preservation_receipt.json'
assert sha(receipt)=='02835f48befa1033c107edaeb275b0f5eedaded28daa90fb47917762c01d7631'
r=load(receipt)
assert r['status']=='PASS' and not r['pilot_pass'] and not r['owned_jobs_live']
assert all(sha(Path(p))==h for p,h in r['current_files_sha256'].items())
archive=ROOT/'before_A52_screen'
archive.mkdir(exist_ok=False)
preserved={}
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    p=BASE/name;q=archive/name
    shutil.copy2(p,q)
    assert sha(q)==r['current_files_sha256'][str(p)]
    preserved[str(p)]={'preserved_path':str(q),'sha256':sha(q)}
entry={'phase':'A52','status':'BOUNDED_SELECTION_NO_METHOD_SELECTED','goal_status':'active',
 'recorded_utc':datetime.now(timezone.utc).isoformat(),
 'previous_receipt':str(receipt),'previous_receipt_sha256':sha(receipt),
 'preserved_canonical_files':preserved,'canonical_files_modified':False,
 'A51_exact_recipe_retired':True,'A51_final_Q_minus_B_APs_pp':0.12080780885908382,
 'A51_final_Q_minus_M_APs_pp':-0.06951170510686033,
 'candidate_mechanisms_to_check':['image-border observation and localization supervision','fixed-budget adaptive image-space sampling'],
 'selection_rule':'Check actual scope of historical routes and primary nearest methods. Credible concrete problem plus a distinguishable operation and affordable test suffice; no exhaustive novelty proof.',
 'no_host_threshold_layer_seed_rebranding':True,'new_training_queued':False,
 'model_operations_added':0,'optimizer_updates_added':0,
 'cumulative_counts':{'whole_model_forwards':165284,'formal_optimizer_updates':136438,'capacity_only_updates':43,'all_optimizer_updates':136481,'formal_epochs':168}}
p=ROOT/'00_A52开始与历史保留.json';assert not p.exists()
p.write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'CURRENT_A52.json').write_text(json.dumps({'status':entry['status'],'goal_status':'active','mainline_confirmed':False,
 'current_priority_candidate':None,'live_training':False,'new_training_queued':False,
 'latest_record':str(p),'receipt_sha256':sha(p)},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':entry['status'],'previous_receipt_verified':True,'old_results_preserved':True,'model_operations_added':0}))

