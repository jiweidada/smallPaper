"""Prespecified coefficient-error control from existing matrices, without model calls."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
R=Path(__file__).resolve().parent.parent;OUT=R/'outputs/p2_coefficient_error_control12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT.mkdir(exist_ok=False)
reports=[]
for seed in (1,2):
    old=R/('outputs/p2_direction_diagnostic12_20261011_v2/seed1_CPU' if seed==1 else 'outputs/p2_subspace_diagnostic12_20261011_v1/seed2_CPU')
    prep,completion=read(old/'report.json'),read(old/'completion.json')
    assert sha(old/'report.json')==completion['report_sha256'] and sha(old/'weights.npz')==completion['weights_npz_sha256']
    with np.load(old/'weights.npz',allow_pickle=False) as z:W=z['original'].copy();DC=z['dc16_norm'].copy()
    assert W.shape==DC.shape==(256,64,1,1) and W.dtype==DC.dtype==np.float32 and np.isfinite(W).all() and np.isfinite(DC).all()
    matrix=W.reshape(256,64).astype(np.float64);dc=DC.reshape(256,64).astype(np.float64)
    u,s,vh=np.linalg.svd(matrix,full_matrices=False)
    # Canonicalize signs before pairing the first16 output directions with the
    # next16 right directions. No validation labels, predictions or AP used.
    for i in range(64):
        peak=int(np.argmax(np.abs(vh[i])))
        if vh[i,peak]<0:vh[i]*=-1;u[:,i]*=-1
    retained=np.sum(s[:16]**2)/np.sum(s**2);r=float(np.sqrt(retained))
    target_error=float(np.linalg.norm(dc-matrix)/np.linalg.norm(matrix))
    cosine=(1-target_error**2/2)/r
    assert 0<cosine<1
    angle=float(np.arccos(cosine));basis=np.cos(angle)*vh[:16]+np.sin(angle)*vh[16:32]
    gram=basis@basis.T;gram_residual=float(np.max(np.abs(gram-np.eye(16))))
    assert gram_residual<1e-12
    ideal=(u[:,:16]*s[:16])@basis
    ideal*=np.linalg.norm(matrix)/np.linalg.norm(ideal)
    projected=ideal.astype(np.float32).reshape(256,64,1,1)
    m=projected.reshape(256,64).astype(np.float64);sv=np.linalg.svd(m,compute_uv=False)
    rank=int(np.sum(sv>sv[0]*1e-5));assert rank==16
    norm_error=float(abs(np.linalg.norm(m)/np.linalg.norm(matrix)-1))
    actual_error=float(np.linalg.norm(m-matrix)/np.linalg.norm(matrix))
    relative_error_difference=actual_error-target_error
    assert norm_error<1e-6 and abs(relative_error_difference)<1e-6
    assert not np.array_equal(projected,DC)
    output=OUT/f'seed{seed}_CPU';output.mkdir()
    np.savez_compressed(output/'weights.npz',original=W,svd16_dc_error_norm=projected)
    report=dict(status='PREPARED_NUMPY_RANK16_NORM_AND_DC_COEFFICIENT_ERROR_CONTROL',seed=seed,
        parent_checkpoint_sha256=prep['checkpoint_sha256'],parent_CPU_report_sha256=sha(old/'report.json'),parent_CPU_weights_sha256=sha(old/'weights.npz'),
        original_weight_SHA256=hashlib.sha256(W.tobytes()).hexdigest(),projected_weight_SHA256=hashlib.sha256(projected.tobytes()).hexdigest(),
        weights_npz_sha256=sha(output/'weights.npz'),mode='svd16_dc_error_norm',rank=rank,numeric_rank_rtol=1e-5,
        relative_weight_norm_error=norm_error,target_DC_relative_coefficient_error=target_error,
        actual_relative_coefficient_error=actual_error,coefficient_error_difference=relative_error_difference,
        prespecified_operator_constraint_tolerance=1e-6,rotation_angle_degrees=float(np.degrees(angle)),right_basis_rows='canonical first16 rotated toward canonical rows17_to32',
        rotated_right_basis_orthogonality_residual=gram_residual,
        definition='Rank16 SVD left factors and singular values retained; right directions rotated toward the next16 so normalized Frobenius error matches existing DC16. Direction pairing/sign conventions use coefficient data only, with no AP search.',
        controls_not_matched=['Operator spectral norm versus DC16','Feature/output energy and covariance','Foreground/background responses','Per-GT geometry or classification sensitivity'],
        new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,new_optimizer_updates=0,
        actual_remote_Torch_CPU_basis_check_completed=False,new_GPU_evaluation_completed=False,
        model_replay_tolerance_changed=False,formal_efficacy_or_unique_root_proven=False)
    with (output/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    reports.append(report)
summary=dict(status='BOTH_SAVED_WEIGHT_COEFFICIENT_ERROR_CONTROLS_NUMPY_PREPARED_NOT_REMOTE_EVALUATED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reports=reports,
    falsifiable_question='If this rotated rank16 matrix with DC-matched unweighted coefficient error still outperforms DC16, coefficient Frobenius reconstruction error alone is insufficient to explain their conditional AP difference.',
    boundary='A result would not isolate image frequency, feature energy, task-aware useful signal, training efficacy or a unique root. Parent replay failures remain preserved; no parent/BASE forward or training retry is part of this preparation.',
    next_required='Review a separate once-only runtime diagnostic, run real CPU operator checks, then wait the already-owned jobs before any new altered-weight GPU forward.',
    new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,root_proven=False,novelty_proven=False,producer_sha256=sha(__file__))
with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=summary['status'],seeds=[dict(seed=x['seed'],rank=x['rank'],rotation_angle_degrees=x['rotation_angle_degrees'],weight_norm_error=x['relative_weight_norm_error'],coefficient_error_difference=x['coefficient_error_difference']) for x in reports],GPU_evaluations=0)))
