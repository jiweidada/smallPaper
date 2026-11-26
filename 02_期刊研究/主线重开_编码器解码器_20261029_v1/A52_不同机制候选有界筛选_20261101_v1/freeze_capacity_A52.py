from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
A51=ROOT.parent/'A51_跨分辨率守恒交换_20261101_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((A51/'protocol_FROZEN_v1.json').read_text(encoding='utf-8'))
assert sha(A51/'terminal_A51_preservation_receipt.json')=='02835f48befa1033c107edaeb275b0f5eedaded28daa90fb47917762c01d7631'
assert json.loads((ROOT/'math_check_v1.json').read_text())['status']=='PASS'
files=['runtime_A52.py','distribution_relation.py','capacity_A52.py']
plan={'phase':'A52','protocol_type':'CAPACITY_ONLY','arms':['B','M','Q'],
 'formal_training_authorized_by_this_file':False,'formal_epochs':0,
 'whole_model_forward_cap':9,'temporary_optimizer_update_cap':6,'automatic_retries':0,
 'new_parameters_M_Q':6480,'total_parameters_M_Q':9809592,'initial_FP32_prediction_absolute_tolerance':1e-5,
 'capacity_batches':'Actual first epoch order: first, highest max per-image GT, highest distinct sum-GT batch',
 'new_module_seed':20261102,'training_data_seed_if_later_frozen':20261101,
 'ordinary_queries':300,'DN_hard_mask_unchanged':True,'BF16_training_FP32_loss':True,
 'full_pipeline_peak_allocated_limit_GiB':23,
 'fail_action':'STOP exact implementation; no formal run or automatic retry from this file.',
 'files_sha256':{n:sha(ROOT/n) for n in files},'external_sha256':prior['external_sha256'],
 'previous_phase_terminal_receipt_sha256':sha(A51/'terminal_A51_preservation_receipt.json')}
p=ROOT/'capacity_protocol_FROZEN_v1.json';assert not p.exists();p.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({'status':'CAPACITY_FROZEN','sha256':sha(p),'whole_forward_cap':9,'temporary_update_cap':6,'formal_epochs':0}))

