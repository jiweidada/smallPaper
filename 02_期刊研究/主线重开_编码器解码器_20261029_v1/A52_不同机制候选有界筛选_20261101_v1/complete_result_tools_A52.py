from pathlib import Path
import ast,json,hashlib,sys
ROOT=Path(__file__).resolve().parent
p=ROOT/'verify_completed_arm_A52_local.py';text=p.read_text(encoding='utf-8')
start=text.index('    from distribution_relation import DistributionRelation\n')
stop=text.index("    ckpt=torch.load",start)
new="""    exact=dest/'new_module_untrained_initial_state.pth'
    init_receipt=json.loads((dest/'new_module_untrained_initial_state_receipt.json').read_text())
    assert sha(exact)==init_receipt['file_sha256']=='673ebec76356bc8b6e5a8435dd4fd23da5f9d11fb7adf8096f80ea69c7425766'
    module_state=torch.load(exact,map_location='cpu',weights_only=True)['state_dict']
    h=hashlib.sha256()
    for key,value in module_state.items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
    initialization=json.loads((folder/'initialization.json').read_text())
    assert h.hexdigest()==initialization['new_module_initialization_sha256']==init_receipt['tensor_sha256']
    initial.update({'decoder.decoder.a52_relation.'+k:v for k,v in module_state.items()})
"""
text=text[:start]+new+text[stop:];ast.parse(text);p.write_text(text,encoding='utf-8')
p=ROOT/'collect_completed_arm_A52.py';text=p.read_text(encoding='utf-8')
old="ROOT/'capacity_result.json',ROOT/'capacity_protocol_FROZEN_v1.json']"
new="ROOT/'capacity_result.json',ROOT/'capacity_protocol_FROZEN_v1.json',ROOT/'new_module_untrained_initial_state.pth',ROOT/'new_module_untrained_initial_state_receipt.json']"
assert text.count(old)==1;text=text.replace(old,new);ast.parse(text);p.write_text(text,encoding='utf-8')
sys.path.insert(0,str(ROOT.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from distribution_relation import DistributionRelation
torch.set_num_threads(4)
r=json.loads((ROOT/'new_module_untrained_initial_state_receipt.json').read_text())
assert hashlib.sha256((ROOT/'new_module_untrained_initial_state.pth').read_bytes()).hexdigest()==r['file_sha256']
remote=torch.load(ROOT/'new_module_untrained_initial_state.pth',map_location='cpu',weights_only=True)['state_dict']
torch.manual_seed(20261102);local=DistributionRelation(8,4,300,'M',4).state_dict()
diff={k:float((remote[k]-v).abs().max()) for k,v in local.items()}
result={'status':'PASS_EXACT_REMOTE_UNTRAINED_STATE_AVAILABLE','prepared_tools_syntax_checked':True,
 'old_utility_CPU_RNG_assertion_failed':True,'specific_difference_cause_not_proved':True,
 'local_torch':str(torch.__version__),'remote_untrained_state_file_sha256':r['file_sha256'],
 'remote_tensor_sha256':r['tensor_sha256'],'local_remote_state_max_abs_differences':diff,
 'verifier_uses_exact_archived_untrained_state_not_recreated_CPU_values':True,
 'frozen_training_source_or_weights_changed':False,'new_detector_forwards':0,'new_optimizer_updates':0}
(ROOT/'result_tool_preparation_receipt_v2.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

