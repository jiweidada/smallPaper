"""Distinct complete detector recipe; preserve all original user experiment sources."""
from pathlib import Path
import ast,hashlib,json,zipfile
R=Path(__file__).resolve().parent.parent
old=R/'英文小论文/code/user_backbone12_20261011_v1'
new=R/'英文小论文/code/p2_lowrank12_20261011_v1';new.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_manifest={p.name:sha(p) for p in old.iterdir() if p.is_file()}
assert len(source_manifest)==28
for p in old.iterdir():
    if not p.is_file():continue
    s=p.read_text(encoding='utf8')
    s=s.replace('user_backbone12_20261011_v1','p2_lowrank12_20261011_v1').replace('user_backbone12_training_v1','p2_lowrank12_training_v1').replace('user_backbone12_closeout_v1','p2_lowrank12_closeout_v1').replace('user_backbone12_terminal_v1','p2_lowrank12_terminal_v1').replace('user_backbone12_identity','p2_lowrank12_identity').replace('UserBackboneDEIM','LowRankBackboneDEIM').replace('load_user_common','load_lowrank_common')
    if p.name in ('runtime.py','prepare_training.py','results.py','preflight.py'):
        s=s.replace("('backbone_silu', 'backbone_attention', 'p2_skip')","('p2_lowrank16',)").replace("('baseline', 'backbone_silu', 'backbone_attention', 'p2_skip')","('baseline', 'p2_lowrank16')").replace("arm=='p2_skip'","arm=='p2_lowrank16'").replace("plan['arm']=='p2_skip'","plan['arm']=='p2_lowrank16'")
    if p.name=='runtime.py':s=s.replace("'user_sources.json','dispatch.py')","'user_sources.json','dispatch.py','lowrank_branch.py')")
    if p.name=='preflight.py':
        s=s.replace('PASS_USER_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES','PASS_LOW_RANK_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES')
        s=s.replace("assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in p2.reduce.parameters())","assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in [p2.mix.weight,*p2.reduce.parameters()])")
    if p.name=='prepare_training.py':
        s=s.replace("assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in branch.reduce.parameters())","assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in [branch.mix.weight,*branch.reduce.parameters()])")
        s=s.replace("scope='One independent historical user backbone change per configuration.","scope='One learned rank16 P2 bridge, random orthonormal mix and zero restore; not initialized from trained SVD.")
    (new/p.name).write_text(s,encoding='utf8')
(new/'user_model.py').write_bytes((R/'outputs/lowrank_detector_model_source12_20261011_v1.txt').read_bytes())
(new/'dispatch.py').write_bytes((R/'outputs/lowrank_detector_dispatch_source12_20261011_v1.txt').read_bytes())
(new/'lowrank_branch.py').write_bytes((R/'英文小论文/code/p2_lowrank_branch_review12_20261011_v2/branch.py').read_bytes())
spec=json.loads((new/'training_protocol.json').read_bytes())
spec.update(execution_allowlist={'p2_lowrank16':[1,2]},mechanism='Compressed width16 P2 -> four spatial phases -> learned64-to16 mix -> zero-initialized16-to256 restore -> P3 before later backbone stages. Only the rank constraint differs from historical P2.',
    training_start='All794 common native states exact,8 additional branch states. Same seed reduce initialization; random orthonormal mix,zero restore. No trained/SVD initialization.',
    expected_parameters=10238173,expected_extra_parameters=6176,
    controls={'p2_lowrank16':'New learned rank16 bridge; compare fixed12 against existing native BASE12 and report separately against original fixed12 P2.'})
spec['concurrency']['reserved_budget_MiB']={'p2_lowrank16':6144}
spec['joint_release']='Requires both original P2 independentEMA proofs and both rank16 runtime controls, with SVD AP/APs not below originalP2. No early release from a single seed.'
spec['new_training_hypothesis']='Restricting spatial mixing to a trainable compact subspace may preserve early geometric information while reducing unnecessary directions. Fixed-weight SVD evidence does not prove optimization from random factors works.'
spec['acceptance']='Primary: both fixed12 seeds APs>BASE12 and AP>=BASE12. Also report both deltas versus originalP2; a gain versus BASE alone is not a gain versus originalP2. No best-epoch substitute, extension,failed configuration retry or newBASE pass.'
spec['novelty']='Rank-limited stride2 convolution and factorization are known mechanisms; this screen does not establish novelty or a unique root.'
(new/'training_protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
u=dict(schema='lowrank_bridge_source_v1',original_user_files_sha256=source_manifest,branch_review_source_sha256=sha(new/'lowrank_branch.py'),original_P2_parameters=17440,new_P2_parameters=6176,training_effect_unproven=True)
(new/'user_sources.json').write_text(json.dumps(u,indent=2),encoding='utf8')
close=json.loads((new/'closeout_protocol.json').read_bytes());close['files']={n:sha(new/n) for n in close['files']};close['files']['lowrank_branch.py']=sha(new/'lowrank_branch.py')
(new/'closeout_protocol.json').write_text(json.dumps(close,indent=2),encoding='utf8')
for p in new.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
assert {p.name:sha(p) for p in old.iterdir() if p.is_file()}==source_manifest
assert sha(new/'geometry.py')==sha(old/'geometry.py') and sha(new/'stage_trace.py')==sha(old/'stage_trace.py')
assert sha(new/'lowrank_branch.py')==sha(R/'英文小论文/code/p2_lowrank_branch_review12_20261011_v2/branch.py')
manifest={p.name:sha(p) for p in new.iterdir() if p.is_file()};assert len(manifest)==29
out=R/'outputs/p2_lowrank12_20261011_v1';out.mkdir(exist_ok=False)
with (out/'source_files.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2)
with zipfile.ZipFile(out/'reviewed_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(new.iterdir()):z.writestr(p.name,p.read_bytes())
review=dict(status='PREPARED_SOURCE_AST_AND_ORIGINAL_SOURCE_PRESERVATION_PASS',files=manifest,original_source_unchanged=True,branch_byte_identical_to_both_host_CPU_review=True,
    two_host_full_detector_CPU_initialization_verified=False,actual_full_detector_AMP_preflight_verified=False,training_started=False,new_BASE_forwards=0,new_training_epochs=0,
    prepared_source_zip_sha256=sha(out/'reviewed_source.zip'),training_protocol_sha256=sha(new/'training_protocol.json'),builder_sha256=sha(__file__))
(out/'source_review.json').write_text(json.dumps(review,indent=2),encoding='utf8')
print(json.dumps(dict(status=review['status'],files=len(manifest),zip_bytes=(out/'reviewed_source.zip').stat().st_size,training_started=False)))
