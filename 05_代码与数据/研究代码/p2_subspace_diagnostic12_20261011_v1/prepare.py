"""Prepare source/checkpoint-bound CPU projections before any GPU diagnostic."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from projection import project
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False)
def live(r,pid,ticks):
    try:
        f=Path('/proc/'+str(r[pid])+'/stat').read_text().split();return f[21]==str(r[ticks]) and f[2]!='Z'
    except FileNotFoundError:return False
def main(prefix,seed):
    assert seed in (1,2) and prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    spec=read(HERE/'protocol.json');assert {n:sha(HERE/n) for n in spec['remote_sources']}==spec['remote_sources']
    user=Path(prefix)/'DEIM/experiments/user_backbone12_20261011_v1'
    assert {p.name:sha(p) for p in (user/'code').iterdir() if p.is_file()}==spec['user28_sources']
    screen=user/f'seed{seed}_p2_skip';run=screen/'run';receipt=read(screen/'bundle/receipt.json');done=read(run/'training_completion.json')
    assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
    assert done['identity']==receipt['identity'] and done['identity']['arm']=='p2_skip' and done['identity']['seed']==seed
    assert done['native_gate_passed'] and not done['automatic_retry']
    assert not live(read(screen/'candidate_start.json'),'child_pid','child_start_ticks') and not live(read(screen/'controller_start.json'),'pid','start_ticks')
    assert read(screen/'candidate_completion.json')['exit_code']==0 and read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    checkpoint=run/'last.pth';assert sha(checkpoint)==done['last_checkpoint_sha256']
    out=HERE.parent/f'seed{seed}_CPU';out.mkdir(exist_ok=False)
    start=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=seed,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),identity=done['identity'],checkpoint_sha256=done['last_checkpoint_sha256'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False)
    write(out/'start.json',start)
    try:
        torch.set_num_threads(2);assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['last_epoch']==11 and payload['user_backbone12_identity']==done['identity']
        W=payload['ema']['module']['backbone.p2_skip.restore.weight'].clone();assert W.shape==(256,64,1,1) and W.dtype==torch.float32
        before=W.clone();weights={'original':W.numpy().copy()};cases=[]
        original_sv=torch.linalg.svdvals(W.double().reshape(256,64))
        original_rank=int((original_sv>original_sv[0]*spec['numeric_rank_rtol']).sum());assert original_rank==64
        for mode in spec['interventions']:
            P=project(W,mode);matrix=P.double().reshape(256,64);sv=torch.linalg.svdvals(matrix)
            rank=int((sv>sv[0]*spec['numeric_rank_rtol']).sum());assert rank==16
            relative_norm_error=float(abs(P.double().norm()/W.double().norm()-1));assert relative_norm_error<1e-6
            if mode=='dc16_norm':assert torch.equal(P.reshape(256,16,4),P.reshape(256,16,4)[:,:,:1].expand(256,16,4))
            basis=torch.eye(64,dtype=torch.float32).reshape(64,16,2,2)
            y=torch.nn.functional.conv2d(torch.nn.functional.pixel_unshuffle(basis,2),P)
            expected=torch.nn.functional.conv2d(basis,P.reshape(256,16,2,2),stride=2)
            assert torch.equal(y,expected)
            weights[mode]=P.numpy().copy()
            cases.append(dict(mode=mode,numerical_rank=rank,numeric_rank_rtol=spec['numeric_rank_rtol'],same_weight_shape=True,relative_Frobenius_norm_error=relative_norm_error,basis64_equal=True,weight_coefficients=16384,weight_SHA256=hashlib.sha256(P.numpy().tobytes()).hexdigest(),operator_relative_error=float((P-W).double().norm()/W.double().norm())))
        assert torch.equal(W,before) and cases[1]['operator_relative_error']<=cases[0]['operator_relative_error']+1e-6
        negative=[]
        for name,bad,mode,rank in (('zero',torch.zeros_like(W),'dc16_norm',16),('NaN',torch.full_like(W,float('nan')),'dc16_norm',16),('wrong_shape',W[:,:60],'dc16_norm',16),('wrong_mode',W,'invalid',16),('wrong_rank',W,'svd16_norm',32)):
            try:project(bad,mode,rank)
            except ValueError:negative.append(name)
            else:raise AssertionError('Invalid projection accepted: '+name)
        np.savez_compressed(out/'weights.npz',**weights)
        report=dict(status='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS',identity=done['identity'],seed=seed,checkpoint_sha256=done['last_checkpoint_sha256'],parent_restore_weight_SHA256=hashlib.sha256(W.numpy().tobytes()).hexdigest(),original_numerical_rank=original_rank,weights_npz_sha256=sha(out/'weights.npz'),cases=cases,negative_cases=negative,original_weight_unchanged=True,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),full_detector_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_training_epochs=0,new_optimizer_updates=0,AP_or_APs_measured=False,scope='Two fixed-weight rank16 interventions, matching weight Frobenius norm. Actual feature/output energy is not matched; no claim of equal training capacity or causalAP. CPU basis/operator checks only.')
        assert sha(checkpoint)==done['last_checkpoint_sha256']
        write(out/'report.json',report);write(out/'completion.json',dict(status=report['status'],exit_code=0,receipt=start,report_sha256=sha(out/'report.json'),weights_npz_sha256=sha(out/'weights.npz'),automatic_retry=False))
        print(json.dumps(dict(status=report['status'],seed=seed,cases=cases)),flush=True)
    except BaseException:
        write(out/'failure.json',dict(receipt=start,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args();main(a.prefix,a.seed)
