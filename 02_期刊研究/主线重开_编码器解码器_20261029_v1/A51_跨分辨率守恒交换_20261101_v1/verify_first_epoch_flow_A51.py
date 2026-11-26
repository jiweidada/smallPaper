from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
m=load(ROOT/'M_startup/epoch01_input_flow.json')
b=load(ROOT.parent/'A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B/epoch01_input_flow.json')
keys=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
checks={k:m[k]==b[k] for k in keys};assert all(checks.values())
init=load(ROOT/'M_startup/initialization.json')
assert init['all_native_initial_tensors_identical'] and init['scratch_no_capacity_weights']
result={'status':'PASS','arm':'M','epoch':1,'optimizer_updates':809,
 'checks_against_archived_native_B':checks,'new_module_initialization_sha256':init['new_module_initialization_sha256'],
 'source_epoch1_sha256':hashlib.sha256((ROOT/'M_startup/epoch01_input_flow.json').read_bytes()).hexdigest(),
 'scope':'B reuse comparison verified for first epoch actual inputs/RNG; no AP or full12 efficacy claim',
 'new_model_forwards_by_verification':0,'new_optimizer_updates_by_verification':0}
p=ROOT/'09_M第一轮输入流与初始化核验.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
current=load(ROOT/'CURRENT_A51.json');current['latest_observation']=str(ROOT/'observer_07.json')
current['M_completed_epochs_at_observation']=1;current['M_observed_optimizer_update_lower_bound']=1010
current['first_epoch_input_flow_check']=str(p)
(ROOT/'CURRENT_A51.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
