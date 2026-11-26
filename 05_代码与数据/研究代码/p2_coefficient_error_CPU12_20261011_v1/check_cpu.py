"""Actual CPU coefficient and operator checks; no detector or GPU forward."""
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
