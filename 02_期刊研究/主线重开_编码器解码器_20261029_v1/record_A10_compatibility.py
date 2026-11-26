from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
root=Path(__file__).resolve().parent
out=root/'A10_跨实例负去噪参考框几何审计_v1'
manifest=json.loads((out/'freeze_manifest.json').read_text(encoding='utf-8'))
for r in manifest['files']:
    assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256']
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'ANNOTATION_LOADING_FIX_BEFORE_DATASET_SAMPLING','first_attempt':{'terminal_exit_code':1,'exception':"NameError: name 'Tensor' is not defined",'phase':'AST function annotation evaluation; no synthetic case or dataset noise draw completed.'},'original_frozen_files_unchanged':True,'protocol_changes':0,'numeric_source_changes':0,'new_wrapper':'a10_author_operator_equivalence_compat_v1.py','new_wrapper_sha256':hashlib.sha256((root/'a10_author_operator_equivalence_compat_v1.py').read_bytes()).hexdigest(),'change':'Provide Tensor=torch.Tensor in the execution namespace and add integrity metadata to the receipt.','new_full_model_forwards':0,'new_fits':0}
with (out/'compatibility_amendment.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':receipt['status'],'original_frozen_files_unchanged':True}))
