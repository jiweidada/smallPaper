from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parent;A51=ROOT.parent/'A51_跨分辨率守恒交换_20261101_v1'
protocol='c13b61bdbc09d81a051108f7a28516f14ae3a3521a81d8247574129ca413d027'
oldproto='a301ec5fb1f6a5c4675ef129f11e8359fa46c5d49640f636ccbeb343703cb8e8'
collect=(A51/'collect_completed_arm_A51.py').read_text(encoding='utf-8')
collect=collect.replace('/root/uav_a51_20261101_v1','/root/uav_a52_20261101_v1').replace(oldproto,protocol)
collect=collect.replace('9876858','9809592').replace('73746','6480').replace('conservative_scale_exchange.py','distribution_relation.py').replace('runtime_A51.py','runtime_A52.py').replace('train_A51.py','train_A52.py')
collect=collect.replace("ROOT/'capacity_result.json',ROOT/'capacity_protocol_FROZEN_v1.json'",
 "ROOT/'capacity_v2/capacity_result.json',ROOT/'capacity_v2/capacity_protocol_FROZEN_v1.json',ROOT/'capacity_result.json',ROOT/'capacity_protocol_FROZEN_v1.json'")
verify=(A51/'verify_completed_arm_A51_local.py').read_text(encoding='utf-8').replace(oldproto,protocol)
verify=verify.replace('9876858','9809592').replace('73746','6480').replace('{9708:490}','{9708:491}')
verify=verify.replace('from conservative_scale_exchange import ScaleExchange','from distribution_relation import DistributionRelation')
verify=verify.replace('module=ScaleExchange(192,a)','module=DistributionRelation(8,4,300,a,4)')
verify=verify.replace("'encoder.a51_exchange.'","'decoder.decoder.a52_relation.'")
audit=(A51/'audit_terminal_inputs_A51.py').read_text(encoding='utf-8').replace(oldproto,protocol).replace("'A51_TERMINAL_ONLY'","'A52_TERMINAL_ONLY'")
for n,s in [('collect_completed_arm_A52.py',collect),('verify_completed_arm_A52_local.py',verify),('audit_terminal_inputs_A52.py',audit)]:
    ast.parse(s);p=ROOT/n;assert not p.exists();p.write_text(s,encoding='utf-8')
# Construct the small new module only; no detector construction/forward/optimizer.
sys.path.insert(0,str(ROOT.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from distribution_relation import DistributionRelation
torch.manual_seed(20261102);m=DistributionRelation(8,4,300,'M',4)
h=hashlib.sha256()
for k,v in m.state_dict().items():h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
assert h.hexdigest()=='69da2b6c3fd9b4c7fa879d807880542cb3a31243004895a67502d8c418829bd0'
p=ROOT/'local_module_init_consistency_v1.json'
p.write_text(json.dumps({'status':'PASS','new_module_initialization_sha256':h.hexdigest(),
 'matches_M_formal_startup':True,'detector_model_forwards':0,'optimizer_updates':0,
 'scope':'Small module CPU initialization and prepared archive utility syntax only; no detector evaluation.'},indent=2),encoding='utf-8')
print(json.dumps({'status':'TOOLS_PREPARED_SYNTAX_CHECKED_NOT_EXECUTED','module_init_cpu_remote_equal':True,'new_detector_operations':0}))

