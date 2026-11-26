"""Freeze runtime-only interventions; independent NumPy rank/norm review."""
from pathlib import Path
import ast,base64,hashlib,json
import numpy as np
ROOT=Path.cwd();code=ROOT/'英文小论文/code/p2_subspace_diagnostic12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
names=['projection.py','prepare.py','evaluate.py','wait.py']
for n in names:ast.parse((code/n).read_text(encoding='utf8'))
old=read(ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2/protocol.json')
spec=dict(schema='P2_saved12_rank16_norm_matched_runtime_diagnostic_v1',remote_sources={n:sha(code/n) for n in names},user28_sources=old['user_files_sha256'],
    interventions=['dc16_norm','svd16_norm'],numeric_rank_rtol=1e-5,
    GPU_uuid={'1':'GPU-d29c55c5-02ab-1e69-c978-0b28f2102003','2':'GPU-b8084aa9-0176-df73-41fb-03307ded1be3'},
    python={'1':'/25085400120/envs/deim/bin/python','2':'/24085403014/deim_migration_runtime/envs/deim/bin/python'},
    predecessor_handles={'1':[dict(role='original_user_queue',pid=24488,start_ticks='479180746'),dict(role='native8400_diagnostic_queue',pid=23061,start_ticks='478791394'),dict(role='Warm_dispatcher',pid=20772,start_ticks='478420101')],
                         '2':[dict(role='original_user_queue',pid=26229,start_ticks='256371187'),dict(role='native8400_diagnostic_queue',pid=24825,start_ticks='255984376'),dict(role='Warm_dispatcher',pid=22543,start_ticks='255612973')]},
    max_wait_seconds=86400,parent_fixed12_training_required=True,parent_positive_native_gate_required=True,parent_independent_saved_EMA_replay_PASS_required=True,
    validation_images=548,candidate_forward_batches_each=137,candidate_forward_batches_total=274,new_native_evaluator_runs=2,new_BASE_forwards=0,new_training_epochs=0,new_optimizer_updates=0,automatic_retry=False,training_allowed=False,
    comparison='Both alter only the trained P2 restore weights. DC keeps within-cell means; SVD keeps the strongest16 linear directions. Both have numerical rank16 and match the original weight Frobenius norm.',
    inferential_limit='Runtime dependence, not a retrained ablation. Feature/output energy and training capacity are not matched; a performance difference does not establish phase as a unique cause. No two-seed training efficacy or novelty claim.')
write(code/'protocol.json',spec)
# Independent engineering matrices; real checkpoint is checked separately on remoteCPU.
rng=np.random.default_rng(77);W=rng.normal(size=(256,64))
dc=np.repeat(W.reshape(256,16,4).mean(2,keepdims=True),4,axis=2).reshape(256,64)
u,s,vh=np.linalg.svd(W,full_matrices=False);svd=(u[:,:16]*s[:16])@vh[:16]
cases=[]
for name,P in [('dc16_norm',dc),('svd16_norm',svd)]:
    P=P*(np.linalg.norm(W)/np.linalg.norm(P));P=P.astype(np.float32)
    vals=np.linalg.svd(P.astype(np.float64),compute_uv=False)
    rank=int((vals>vals[0]*spec['numeric_rank_rtol']).sum());assert rank==16
    norm_error=float(abs(np.linalg.norm(P.astype(np.float64))/np.linalg.norm(W)-1));assert norm_error<1e-6
    basis=np.eye(64).reshape(64,16,2,2)
    assert np.array_equal(basis.reshape(64,64)@P.T,np.einsum('bcij,ocij->bo',basis,P.reshape(256,16,2,2)))
    assert not np.allclose(P,W) and np.isfinite(P).all()
    cases.append(dict(mode=name,rank=rank,Frobenius_norm_relative_error=norm_error,operator_relative_error=float(np.linalg.norm(P-W)/np.linalg.norm(W)),basis64_equal=True))
assert cases[1]['operator_relative_error']<=cases[0]['operator_relative_error']
assert np.array_equal(dc.reshape(256,16,4),np.repeat(dc.reshape(256,16,4)[:,:,:1],4,axis=2))
# Reversing phase order or passing an unprojected matrix must not masquerade as rank16 control.
assert not np.allclose(basis.reshape(64,64)@W.T,np.einsum('bcij,ocij->bo',basis,W.reshape(256,16,2,2)[:,:,:,::-1]))
assert np.linalg.matrix_rank(W)==64
review=dict(status='PASS_SOURCE_AST_TWO_NUMPY_RANK16_NORM_AND64_BASIS_ENGINEERING_CASES',cases=cases,negative_wrong_phase_mapping_rejected=True,unprojected_rank64_rejected=True,
    remote_actual_trained_weight_validation_pending=True,GPU_native_evaluation_pending=True,training_allowed=False,new_training_epochs=0,new_BASE_forwards=0,files={p.name:sha(p) for p in code.iterdir() if p.is_file()},
    source_sha256=sha(__file__),native_model_AP_claim=False)
write(ROOT/'outputs/p2_subspace_diagnostic12_20261011_v1_source_review.json',review)
print(json.dumps({p.name:base64.b64encode(p.read_bytes()).decode() for p in code.iterdir() if p.is_file()}))
