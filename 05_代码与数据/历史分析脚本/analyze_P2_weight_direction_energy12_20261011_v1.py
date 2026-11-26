"""Coefficient-space spectra and local phase patterns, not feature signal energy."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent.parent;OUT=R/'outputs/P2_weight_direction_energy12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence={};rows=[]
H=.5*np.asarray([[1,1,1,1],[-1,-1,1,1],[-1,1,-1,1],[1,-1,-1,1]],dtype=np.float64)
assert np.array_equal(H@H.T,np.eye(4))
labels=['spatial_constant','row_contrast','column_contrast','checkerboard']
old=R/'英文小论文/code/p2_subspace_diagnostic12_20261011_v1'
for seed in (1,2):
    folder=R/('outputs/p2_direction_diagnostic12_20261011_v2/seed1_CPU' if seed==1 else 'outputs/p2_subspace_diagnostic12_20261011_v1/seed2_CPU')
    prep,done=read(folder/'report.json'),read(folder/'completion.json')
    assert done['exit_code']==0 and prep['seed']==seed and prep['original_numerical_rank']==64
    assert prep['status']=='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS'
    assert sha(folder/'report.json')==done['report_sha256'] and sha(folder/'weights.npz')==done['weights_npz_sha256']==prep['weights_npz_sha256']
    assert prep['source_sha256']==sha(old/'prepare.py') and prep['protocol_sha256']==sha(old/'protocol.json')
    if seed==1:
        spec=read(R/'英文小论文/code/p2_direction_diagnostic12_20261011_v2/protocol.json')
        assert sha(folder/'report.json')==spec['CPU_report_sha256'] and sha(folder/'weights.npz')==spec['CPU_weights_sha256']
    for p in folder.iterdir():
        if p.is_file():evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
    with np.load(folder/'weights.npz',allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
    assert set(weights)=={'original','dc16_norm','svd16_norm'}
    W=weights['original'];assert W.shape==(256,64,1,1) and W.dtype==np.float32 and np.isfinite(W).all()
    assert hashlib.sha256(W.tobytes()).hexdigest()==prep['parent_restore_weight_SHA256']
    matrix=W.reshape(256,64).astype(np.float64);singular=np.linalg.svd(matrix,compute_uv=False)
    retained=float(np.sum(singular[:16]**2)/np.sum(singular**2))
    constant=np.repeat(matrix.reshape(256,16,4).mean(2,keepdims=True),4,axis=2).reshape(256,64)
    dc_retained=float(np.sum(constant**2)/np.sum(matrix**2))
    modes=[]
    for name,w in weights.items():
        assert w.shape==W.shape and w.dtype==W.dtype and np.isfinite(w).all()
        m=w.reshape(256,64).astype(np.float64);total=float(np.sum(m*m))
        coefficients=m.reshape(256,16,4)@H.T
        energy=np.sum(coefficients**2,axis=(0,1))
        row=dict(mode=name,coefficient_Frobenius_norm=float(np.sqrt(total)),
            phase_pattern_energy_fraction={label:float(v/total) for label,v in zip(labels,energy)},
            orthonormal_transform_energy_residual=float(np.sum(energy)-total),
            normalized_operator_error_vs_original=float(np.linalg.norm(m-matrix)/np.linalg.norm(matrix)))
        if name!='original':
            stored=next(x for x in prep['cases'] if x['mode']==name)
            assert hashlib.sha256(w.tobytes()).hexdigest()==stored['weight_SHA256']
            sv=np.linalg.svd(m,compute_uv=False);assert int(np.sum(sv>sv[0]*stored['numeric_rank_rtol']))==16
            row['relative_weight_norm_error']=float(abs(np.linalg.norm(m)/np.linalg.norm(matrix)-1))
            assert row['relative_weight_norm_error']<1e-6
        if name=='dc16_norm':assert all(row['phase_pattern_energy_fraction'][k]==0 for k in labels[1:])
        modes.append(row)
    original_phase=next(x for x in modes if x['mode']=='original')['phase_pattern_energy_fraction']
    # Numerical agreement of two algebraic coefficient-energy expressions is
    # recorded explicitly, without changing any model/replay tolerance.
    rows.append(dict(seed=seed,checkpoint_sha256=prep['checkpoint_sha256'],coefficient_matrix_shape=[256,64],
        original_numerical_rank=64,top16_singular_coefficient_energy_fraction=retained,
        original_DC_projection_coefficient_energy_fraction=dc_retained,
        DC_energy_algebraic_residual=dc_retained-original_phase['spatial_constant'],
        SVD16_normalization_scale_from_ideal_coefficient_spectrum=1/np.sqrt(retained),
        DC16_normalization_scale_from_ideal_projection=1/np.sqrt(dc_retained),
        singular_values=singular.tolist(),modes=modes))
OUT.mkdir(exist_ok=False)
params=dict(original_P2_branch=17440,learned_rank16_branch=6176,original_P2_detector=10249437,learned_rank16_detector=10238173)
params.update(saved_parameters=11264,branch_fraction_saved=11264/17440,whole_detector_fraction_saved=11264/10249437)
report=dict(status='TWO_ACTUAL_SAVED_P2_WEIGHT_COEFFICIENT_SPECTRA_AND_LOCAL_PHASE_ENERGIES_ANALYZED',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,parameter_comparison=params,
    local_phase_basis=dict(phase_order=['top_left','top_right','bottom_left','bottom_right'],patterns={k:v.tolist() for k,v in zip(labels,H)}),
    interpretation=['Top16 SVD energy is squared matrix-coefficient energy. It does not measure foreground information, feature-output energy, mutual information, or AP importance.',
      'DC16 keeps only a constant2x2 pattern per reduced channel; SVD16 retains arbitrary joint channel/phase directions. Both have rank16 and matched weight norm, but their operator errors and input responses differ.',
      'This local2x2 coefficient basis is not an attention mechanism and is not a claim that image high frequencies or small-target signal have this energy share.',
      'SVD is optimal for unweighted coefficient Frobenius approximation. Actual feature covariance and task sensitivity are not measured here.',
      'Saving64.59% of the branch parameters saves only0.1099% of the full detector parameters. No latency or FLOP advantage has been measured.',
      'The two new learned-rank configurations are randomly initialized from common native states. These trained-SVD coefficients were not used to initialize them.'],
    seed1_original_exact_replay_status='CANDIDATE_RELOAD_MISMATCH_NO_RETRY',
    new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,
    root_proven=False,novelty_proven=False,formal_new_rank16_training_efficacy_proven=False)
with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['P2训练后矩阵的方向分析：只读两种子已有权重，不运行模型。','']
for x in rows:
    modes={m['mode']:m for m in x['modes']}
    lines.append(f'种子{x["seed"]}：前16个奇异方向保留{x["top16_singular_coefficient_energy_fraction"]:.2%}的矩阵系数能量；空间常量方向保留{x["original_DC_projection_coefficient_energy_fraction"]:.2%}。')
    lines.append(f'SVD16范数匹配需放大约{x["SVD16_normalization_scale_from_ideal_coefficient_spectrum"]:.4f}倍；DC16约{x["DC16_normalization_scale_from_ideal_projection"]:.4f}倍。')
    for name in ('original','dc16_norm','svd16_norm'):
        f=modes[name]['phase_pattern_energy_fraction']
        lines.append(f'  {name}: 常量{f["spatial_constant"]:.2%}；行对比{f["row_contrast"]:.2%}；列对比{f["column_contrast"]:.2%}；棋盘对比{f["checkerboard"]:.2%}。')
    lines.append('')
lines+=['这些比例是矩阵系数能量，不是实际特征能量、小目标有效信息量或AP贡献。',
    'DC16和SVD16的秩/权重范数相同，但前者只保留局部空间常量，后者混合通道和局部位置；二者算子误差及输入响应仍不同。',
    '因此“同范数”不足以证明只改变信息方向，更不能把SVD优于DC直接写成高频或去冗余根因。',
    '', '参数减少11,264：分支减少64.59%，整个检测器只减少约0.1099%；没有实测速度/FLOP收益。',
    '新rank16训练没有使用这组训练后SVD权重初始化，仍须按两个种子固定第12轮比较。',
    'A原P2严格重放失败保留；该系数分析没有重试父模型或增加基线前向。']
(OUT/'方向与参数分析.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
for p in (old/'prepare.py',old/'projection.py',old/'protocol.json',R/'英文小论文/code/p2_lowrank12_20261011_v3/lowrank_branch.py',R/'英文小论文/code/p2_lowrank12_20261011_v3/user_p2.py'):
    evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
with (OUT/'manifest.json').open('x',encoding='utf8') as f:json.dump(dict(files=evidence,report_sha256=sha(OUT/'report.json'),text_sha256=sha(OUT/'方向与参数分析.txt'),producer_sha256=sha(__file__)),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=report['status'],rows=[{k:x[k] for k in ('seed','top16_singular_coefficient_energy_fraction','original_DC_projection_coefficient_energy_fraction','SVD16_normalization_scale_from_ideal_coefficient_spectrum','DC16_normalization_scale_from_ideal_projection')} for x in rows],parameter_comparison=params,new_model_forwards=0)))
