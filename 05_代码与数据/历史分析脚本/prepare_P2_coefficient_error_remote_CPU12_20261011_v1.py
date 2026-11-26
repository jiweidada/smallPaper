"""Package CPU-only checks of the two prespecified new coefficient controls."""
from pathlib import Path
import hashlib,json,ast,zipfile
R=Path(__file__).resolve().parent.parent
root=R/'英文小论文/code/p2_coefficient_error_CPU12_20261011_v1';root.mkdir(exist_ok=False)
out=R/'outputs/p2_coefficient_error_control12_20261011_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
source='''"""Actual CPU coefficient and operator checks; no detector or GPU forward."""
from pathlib import Path
import os,sys,json,hashlib,datetime,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main(seed):
    assert seed in (1,2);p=read(HERE/'protocol.json');assert sha(__file__)==p['CPU_check_sha256']
    root=HERE.parent;input=root/('seed%d_input'%seed);proposal=read(input/'report.json')
    expected=p['inputs'][str(seed)]
    assert sha(input/'report.json')==expected['report_sha256'] and sha(input/'weights.npz')==expected['weights_sha256']
    assert proposal['seed']==seed and proposal['mode']=='svd16_dc_error_norm'
    prefix=Path('/25085400120' if seed==1 else '/24085403014')
    old=prefix/'DEIM/experiments/p2_subspace_diagnostic12_20261011_v1'/('seed%d_CPU'%seed)
    assert sha(old/'report.json')==proposal['parent_CPU_report_sha256'] and sha(old/'weights.npz')==proposal['parent_CPU_weights_sha256']
    oldreport=read(old/'report.json');assert oldreport['checkpoint_sha256']==proposal['parent_checkpoint_sha256']
    with np.load(input/'weights.npz',allow_pickle=False) as z:original=z['original'].copy();projected=z['svd16_dc_error_norm'].copy()
    with np.load(old/'weights.npz',allow_pickle=False) as z:assert np.array_equal(original,z['original']);dc=z['dc16_norm'].copy()
    assert hashlib.sha256(original.tobytes()).hexdigest()==proposal['original_weight_SHA256']
    assert hashlib.sha256(projected.tobytes()).hexdigest()==proposal['projected_weight_SHA256']
    torch.set_num_threads(2);assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    W,P,D=[torch.from_numpy(x) for x in (original,projected,dc)]
    assert W.shape==P.shape==D.shape==(256,64,1,1) and W.dtype==P.dtype==D.dtype==torch.float32
    assert torch.isfinite(P).all()
    sv=torch.linalg.svdvals(P.double().reshape(256,64));rank=int((sv>sv[0]*proposal['numeric_rank_rtol']).sum());assert rank==16
    norm_error=float(abs(P.double().norm()/W.double().norm()-1))
    target=float((D-W).double().norm()/W.double().norm());error=float((P-W).double().norm()/W.double().norm())
    assert norm_error<p['coefficient_constraint_tolerance'] and abs(error-target)<p['coefficient_constraint_tolerance']
    basis=torch.eye(64,dtype=torch.float32).reshape(64,16,2,2)
    unshuffled=torch.nn.functional.conv2d(torch.nn.functional.pixel_unshuffle(basis,2),P)
    equivalent=torch.nn.functional.conv2d(basis,P.reshape(256,16,2,2),stride=2)
    assert torch.equal(unshuffled,equivalent)
    output=root/('seed%d_CPU_checked'%seed);output.mkdir(exist_ok=False)
    result=dict(status='PASS_ACTUAL_CPU_RANK16_NORM_DC_ERROR_AND64_BASIS_OPERATOR_CHECK',seed=seed,
        proposal_sha256=sha(input/'report.json'),weights_npz_sha256=sha(input/'weights.npz'),rank=rank,
        norm_error=norm_error,DC_relative_coefficient_error=target,projected_relative_coefficient_error=error,
        error_difference=error-target,basis64_exact=True,CUDA_initialized=torch.cuda.is_initialized(),
        full_detector_forwards=0,BASE_forwards=0,parent_forwards=0,evaluator_calls=0,optimizer_updates=0,training_epochs=0,
        code_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),parent_checkpoint_sha256=proposal['parent_checkpoint_sha256'],
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_evaluation_completed=False,root_or_efficacy_proven=False)
    with (output/'report.json').open('x') as f:json.dump(result,f,indent=2)
    with (output/'completion.json').open('x') as f:json.dump(dict(exit_code=0,report_sha256=sha(output/'report.json'),automatic_retry=False),f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);main(a.parse_args().seed)
'''
ast.parse(source);(root/'check_cpu.py').write_text(source,encoding='utf8')
spec=dict(schema='P2_coefficient_error_control_real_CPU_v1',CPU_check_sha256=sha(root/'check_cpu.py'),
    inputs={str(s):dict(report_sha256=sha(out/f'seed{s}_CPU/report.json'),weights_sha256=sha(out/f'seed{s}_CPU/weights.npz')) for s in (1,2)},
    coefficient_constraint_tolerance=1e-6,model_replay_tolerance_unchanged=0,full_detector_or_training_commands=0,
    scope='CPU check only of prespecified rank16, norm and DC coefficient-error control. No AP, model forward, parent replay or BASE inference.')
(root/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
archives=[]
for seed in (1,2):
    target=out/f'seed{seed}_CPU_deployment_v1.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in root.iterdir():z.write(p,'code/'+p.name)
        for p in (out/f'seed{seed}_CPU').iterdir():z.write(p,f'seed{seed}_input/'+p.name)
    archives.append(dict(seed=seed,archive_sha256=sha(target),bytes=target.stat().st_size))
(out/'remote_CPU_source_review.json').write_text(json.dumps(dict(status='REVIEWED_CPU_ONLY_NO_NATIVE_MODEL_OR_TRAINING_CALLS',source_sha256=sha(root/'check_cpu.py'),protocol_sha256=sha(root/'protocol.json'),archives=archives),indent=2),encoding='utf8')
print(json.dumps(dict(status='REVIEWED_TWO_COEFFICIENT_ERROR_CPU_PACKAGES',archives=archives)))
