from pathlib import Path
import hashlib,json,runpy,sys
import numpy as np

D=Path(__file__).resolve().parent
assert len(sys.argv)==1
manifest=json.loads((D/'execution_inputs_manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files_sha256'].items():
    assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
assert not (D/'analysis').exists()
def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported JSON artifact type: '+type(value).__name__)
observed=json.loads(json.dumps({'gate':np.bool_(True),'integer':np.int64(2),'metric':np.float64(.625)},default=scalar))
assert observed=={'gate':True,'integer':2,'metric':.625}
with (D/'analysis_scalar_export_guard_receipt.json').open('x',encoding='utf-8') as output:
    json.dump(dict(status='ARTIFACT_ONLY_SCALAR_JSON_GUARD_BEFORE_DATA_FITS',
        reason='Synthetic audit found NumPy comparison bool scalars are not natively JSON serializable.',
        original_frozen_analysis_unchanged=True,original_frozen_metric_functions_unchanged=True,
        source_analysis_sha256=manifest['files_sha256']['analyze_features.py'],
        source_semantic_math_sha256=manifest['files_sha256']['semantic_math.py'],
        scientific_numeric_values_changed=False,new_author_model_calls=0,data_probe_fits_so_far=0),output,indent=2)
original_dump=json.dump;original_dumps=json.dumps
def guarded_dump(obj,fp,*args,**kwargs):
    kwargs.setdefault('default',scalar);return original_dump(obj,fp,*args,**kwargs)
def guarded_dumps(obj,*args,**kwargs):
    kwargs.setdefault('default',scalar);return original_dumps(obj,*args,**kwargs)
json.dump=guarded_dump;json.dumps=guarded_dumps
sys.argv=[str(D/'analyze_features.py'),'--evidence',str(D/'downloaded_evidence'),'--output',str(D/'analysis')]
runpy.run_path(str(D/'analyze_features.py'),run_name='__main__')
