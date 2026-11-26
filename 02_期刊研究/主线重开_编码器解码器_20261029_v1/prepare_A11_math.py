from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
out=root/'A11_坐标插值相位与分布扩散审计_v1'
assert not out.exists()
out.mkdir()
source=root/'official_assets_v1'/'Dome_DETR_2dde3bc_v1'/'source'
files=['src/zoo/dome/dome_utils.py','src/zoo/dome/dome_decoder.py']
p={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_NUMERICAL_OPERATOR_AUDIT_ANALYTICAL_EXPECTATIONS_ALREADY_DERIVED','type':'Source-bound synthetic mathematical audit;no dataset or detector predictions.','source_commit':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861','checkpoint_sha256':'42874302fd97e17c2a23a46a8ec35d2ea4d2ffe479f0d632c1327fafbf910eb7','source_hashes':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files},'reg_max':32,'alphas':[.1,.5,.9],'seed':2026103011,'cases':'All32 source-grid intervals x3 fixed interpolation phases;128 Dirichlet distributions;one pre-specified same-mean entropy-floor counterexample.','source_roundtrip_tolerance':2e-6,'gradient_tolerance':1e-12,'excess_variance_tolerance':1e-12,'question':'Can exact coordinate interpolation have phase-dependent entropy/variance,and can raw-entropy descent alter an otherwise exact coordinate?','not_claimed':['New theorem','Real data mechanism','Calibrated predictive uncertainty','Detection/training gain','Novel algorithm'],'new_full_model_forwards':0,'new_fits':0,'main_model_updates':0,'cumulative_full_model_forwards':200,'next_gate':'Real saved edge-distribution evidence and nearest-method clearance required before any architecture increment. ExistingA8 outputs do not contain these distributions.'}
(out/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(out/'freeze_manifest.json').write_text(json.dumps({'files':[{'file':n,'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in ['a11_coordinate_phase_math.py','prepare_A11_math.py']]},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':p['status'],'cases':96,'random_distributions':128,'new_model_forwards':0}))
