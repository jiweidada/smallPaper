from pathlib import Path
import ast,base64,hashlib,json
import numpy as np
ROOT=Path.cwd();code=ROOT/'英文小论文/code/p2_linear_equivalence_audit12_20261011_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ast.parse((code/'audit.py').read_text(encoding='utf8'))
old=json.loads((ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2/protocol.json').read_bytes())
spec=dict(schema='P2_actual_fixed12_EMA_CPU_equivalence_v1',audit_sha256=sha(code/'audit.py'),user28_sources=old['user_files_sha256'],existing_trainers=old['existing_trainers'],matrix_rank_rtol=1e-8,cases=[[1,20,20],[2,21,31],[1,40,60],[1,160,160],[1,208,208]],tolerances=dict(float64=dict(atol=1e-11,rtol=1e-10,grad_atol=1e-9,grad_rtol=1e-10),float32=dict(atol=3e-5,rtol=3e-4,grad_atol=.02,grad_rtol=1e-3)),new_training_epochs=0,new_full_detector_forwards=0,new_BASE_forwards=0,new_optimizer_updates=0,new_evaluator_calls=0,training_allowed=False)
with (code/'protocol.json').open('x',encoding='utf8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
# Independent NumPy basis mapping and its negative phase-order fixture before remote dispatch.
rng=np.random.default_rng(44);W=rng.normal(size=(256,64));K=W.reshape(256,16,2,2);basis=np.eye(64).reshape(64,16,2,2)
unshuffle=basis.reshape(64,64);a=unshuffle@W.T;b=np.einsum('bcij,ocij->bo',basis,K)
assert np.array_equal(a,b) and not np.allclose(a,np.einsum('bcij,ocij->bo',basis,K[:,:,:,::-1]))
review=dict(status='PASS64_INDEPENDENT_BASIS_MAPPING_AND_WRONG_PHASE_REJECTION',actual_remote_trained_weight_tests=0,full_model_or_AP_claim=False,files={p.name:sha(p) for p in code.iterdir() if p.is_file()})
with (ROOT/'outputs/p2_linear_equivalence_audit12_20261011_v1_source_review.json').open('x',encoding='utf8') as f:json.dump(review,f,indent=2)
print(json.dumps({p.name:base64.b64encode(p.read_bytes()).decode() for p in code.iterdir() if p.is_file()}))
