"""CPU algebra review of the exact user P2 branch with stopped fixed12 EMA weights."""
from pathlib import Path
import argparse,ast,copy,datetime,hashlib,json,os,sys,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False)
def live(r,pid,tick):
    try:
        v=Path('/proc/'+str(r[pid])+'/stat').read_text().split()
        return v[21]==str(r[tick]) and v[2]!='Z'
    except FileNotFoundError:return False
class Equivalent(nn.Module):
    def __init__(self,branch):
        super().__init__();self.reduce=copy.deepcopy(branch.reduce)
        self.restore=nn.Conv2d(16,256,2,stride=2,bias=False).to(branch.restore.weight)
        with torch.no_grad():self.restore.weight.copy_(branch.restore.weight.reshape(256,16,2,2))
    def forward(self,p2,p3):
        d=self.reduce(p2);h,w=d.shape[-2:]
        if h%2 or w%2:d=F.pad(d,(0,w%2,0,h%2))
        return p3+self.restore(d)
def compare(a,b,atol,rtol):
    assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    err=float((a-b).abs().max());torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    return err
def main(prefix,seed):
    assert seed in (1,2) and prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    p=read(HERE/'protocol.json');assert sha(__file__)==p['audit_sha256']
    user=Path(prefix)/'DEIM/experiments/user_backbone12_20261011_v1'
    assert {x.name:sha(x) for x in (user/'code').iterdir() if x.is_file()}==p['user28_sources']
    screen=user/f'seed{seed}_p2_skip';run=screen/'run'
    trainer,controller=read(screen/'candidate_start.json'),read(screen/'controller_start.json')
    frozen=p['existing_trainers'][str(seed)]['p2_skip']
    assert trainer['child_pid']==frozen['pid'] and str(trainer['child_start_ticks'])==str(frozen['start_ticks'])
    assert controller['pid']==frozen['controller_pid'] and str(controller['start_ticks'])==str(frozen['controller_start_ticks'])
    assert not live(trainer,'child_pid','child_start_ticks') and not live(controller,'pid','start_ticks')
    done=read(run/'training_completion.json');receipt=read(screen/'bundle/receipt.json')
    assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
    assert done['identity']==receipt['identity'] and done['identity']['arm']=='p2_skip' and done['identity']['seed']==seed and not done['automatic_retry']
    assert read(screen/'candidate_completion.json')['exit_code']==0 and not read(screen/'candidate_completion.json')['timed_out']
    assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    checkpoint=run/'last.pth';assert sha(checkpoint)==done['last_checkpoint_sha256']
    assert sha(run/'stages_epoch12/manifest.json')==done['stage_manifest_sha256']
    assert sha(run/'endpoint_epoch12/manifest.json')==done['endpoint_manifest_sha256']
    output=HERE.parent/f'seed{seed}';output.mkdir(exist_ok=False)
    started=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=seed,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),checkpoint_sha256=done['last_checkpoint_sha256'],identity=done['identity'],automatic_retry=False,full_detector_forwards=0,BASE_forwards=0,optimizer_updates=0,training_epochs=0)
    write(output/'start.json',started)
    try:
        torch.set_num_threads(2)
        assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
        source=(user/'code/user_p2.py').read_text(encoding='utf8');tree=ast.parse(source)
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('conv_bn_relu','P2ToP3Residual')]
        assert [n.name for n in nodes]==['conv_bn_relu','P2ToP3Residual']
        namespace=dict(torch=torch,nn=nn,F=F,OrderedDict=OrderedDict)
        exec(compile('\n\n'.join(ast.get_source_segment(source,n) for n in nodes),str(user/'code/user_p2.py')+'::exact_branch','exec'),namespace)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['last_epoch']==11 and payload['user_backbone12_identity']==done['identity']
        branch_state={k[len('backbone.p2_skip.'):]:v for k,v in payload['ema']['module'].items() if k.startswith('backbone.p2_skip.')}
        assert len(branch_state)==7 and all(v.device.type=='cpu' for v in branch_state.values())
        Branch=namespace['P2ToP3Residual'];native=Branch(64,256,16).eval();native.load_state_dict(branch_state,strict=True)
        assert sum(x.numel() for x in native.parameters())==17440
        assert torch.count_nonzero(native.restore.weight)>0,'Zero initialization alone would be a vacuous equivalence test'
        before={k:v.clone() for k,v in native.state_dict().items()};del payload
        W=native.restore.weight.detach().double().reshape(256,64)
        K=W.reshape(256,16,2,2);Km=K.mean((2,3),keepdim=True).expand_as(K)
        sv=torch.linalg.svdvals(W);svm=torch.linalg.svdvals(Km.reshape(256,64))
        rtol_rank=p['matrix_rank_rtol'];rank=int((sv>sv[0]*rtol_rank).sum());mean_rank=int((svm>svm[0]*rtol_rank).sum())
        assert rank>mean_rank and mean_rank<=16 and rank<=64
        # All64 reduced-channel/spatial basis vectors, not only a random projection.
        basis=torch.eye(64,dtype=torch.float64).reshape(64,16,2,2)
        one=F.conv2d(F.pixel_unshuffle(basis,2),W.reshape(256,64,1,1))
        two=F.conv2d(basis,K,stride=2)
        basis_error=compare(one,two,1e-14,1e-13)
        assert not torch.allclose(one,F.conv2d(basis,K.flip(-1),stride=2),atol=1e-10,rtol=1e-8),'Wrong phase mapping must be rejected'
        cases=[];branch_forwards=0;control_forwards=0
        for dtype_name in ('float64','float32'):
            dtype=getattr(torch,dtype_name);tol=p['tolerances'][dtype_name]
            for index,(batch,h,w) in enumerate(p['cases']):
                torch.manual_seed(3100+seed*100+index)
                a=copy.deepcopy(native).to(dtype).eval();b=Equivalent(a).eval()
                assert sum(x.numel() for x in a.parameters())==sum(x.numel() for x in b.parameters())==17440
                x=torch.randn(batch,64,h,w,dtype=dtype);z=torch.randn(batch,256,(h+1)//2,(w+1)//2,dtype=dtype)
                x1=x.clone().requires_grad_();x2=x.clone().requires_grad_();z1=z.clone().requires_grad_();z2=z.clone().requires_grad_()
                y1=a(x1,z1);y2=b(x2,z2);branch_forwards+=2
                err=compare(y1,y2,tol['atol'],tol['rtol'])
                torch.manual_seed(9100+seed*100+index);probe=torch.randn_like(y1)
                (y1*probe).sum().backward();(y2*probe).sum().backward()
                graderr=compare(x1.grad,x2.grad,tol['grad_atol'],tol['grad_rtol'])
                p3err=compare(z1.grad,z2.grad,tol['grad_atol'],tol['grad_rtol'])
                grads=[]
                aa,bb=dict(a.named_parameters()),dict(b.named_parameters())
                assert set(aa)==set(bb)
                for name in aa:
                    target=bb[name].grad.reshape_as(aa[name].grad) if name=='restore.weight' else bb[name].grad
                    grads.append(compare(aa[name].grad,target,tol['grad_atol'],tol['grad_rtol']))
                assert float(a.restore.weight.grad.abs().max())>0 and float(a.reduce.conv.weight.grad.abs().max())>0
                with torch.no_grad():
                    d=a.reduce(x);ph,pw=d.shape[-2:]
                    if ph%2 or pw%2:d=F.pad(d,(0,pw%2,0,ph%2))
                    folded=F.pixel_unshuffle(d,2);mean=folded.reshape(batch,16,4,(h+1)//2,(w+1)//2).mean(2,keepdim=True).expand(-1,-1,4,-1,-1).reshape_as(folded)
                    kc=a.restore.weight.reshape(256,16,2,2).mean((2,3),keepdim=True).expand(256,16,2,2).contiguous()
                    control=compare(a.restore(mean),F.conv2d(d,kc,stride=2),tol['atol'],tol['rtol']);control_forwards+=2
                cases.append(dict(dtype=dtype_name,batch=batch,HW=[h,w],output_max_abs_error=err,P2_gradient_max_abs_error=graderr,P3_gradient_max_abs_error=p3err,parameter_gradient_max_abs_error=max(grads),phase_mean_to_tied_kernel_max_abs_error=control,parameters=17440,nonzero_reduction_and_restore_gradients=True))
        assert all(torch.equal(v,native.state_dict()[k]) for k,v in before.items())
        assert sha(checkpoint)==done['last_checkpoint_sha256']
        assert {x.name:sha(x) for x in (user/'code').iterdir() if x.is_file()}==p['user28_sources']
        assert not torch.cuda.is_initialized() and not any(k.startswith('engine') for k in sys.modules)
        result=dict(status='PASS_TRAINED_USER_P2_EMA_LINEAR_EQUIVALENCE_AND_PHASE_MEAN_CAPACITY_AUDIT',seed=seed,identity=done['identity'],checkpoint_sha256=done['last_checkpoint_sha256'],source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),torch_version=torch.__version__,basis64_mapping_verified=True,wrong_phase_kernel_rejected=True,basis_max_abs_error=basis_error,cases=cases,trained_restore_nonzero_coefficients=int(torch.count_nonzero(W)),trained_restore_rank=rank,phase_mean_tied_restore_rank=mean_rank,matrix_rank_relative_threshold=rtol_rank,smallest_original_singular_value=float(sv[-1]),largest_original_singular_value=float(sv[0]),phase_contrast_weight_energy_fraction=float((K-Km).square().sum()/K.square().sum()),parameter_count_both_branches=17440,restore_coefficient_count_original=16384,independent_coefficients_phase_mean=4096,original_EMA_branch_state_unchanged=True,checkpoint_unchanged=True,original28_training_sources_unchanged=True,branch_forwards=branch_forwards,reduced_operator_forwards=control_forwards+3,full_detector_forwards=0,BASE_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,
        exact_algebra='K[o,c,dy,dx]=W[o,4*c+2*dy+dx,0,0]; same bottom/right odd padding. No new information is created by the permutation.',
        inference='This specific unshuffle2+unrestricted1x1 restore is a reparameterization of a learned2x2 stride2 convolution on the nonlinear reduced feature. Phase-mean ties all4 kernel positions and lowers functional rank/independent coefficients, despite retaining nominal17440 parameters. A phase-mean accuracy gap cannot alone prove a pure phase benefit at matched effective capacity. Nonzero trained contrast weights show a used algebraic component, not its causal AP contribution.',
        scientific_AP_or_APs_measured=False,causal_root_proven=False,novelty_proven=False,training_control_approved=False,scope='Exact original branch declarations and actual stoppedfixed12 EMA coefficients on synthetic feature tensors. Float64/float32 output and input/parameter gradients; no complete detector, data loader, criterion, optimizer step, AMP claim, native evaluator, BASE inference, training or learned-state change.')
        write(output/'report.json',result)
        write(output/'completion.json',dict(status=result['status'],exit_code=0,receipt=started,report_sha256=sha(output/'report.json'),automatic_retry=False,new_training_epochs=0,full_detector_forwards=0))
        print(json.dumps(dict(status=result['status'],seed=seed,cases=len(cases),rank=rank,mean_rank=mean_rank,contrast_energy=result['phase_contrast_weight_energy_fraction'])),flush=True)
    except BaseException:
        write(output/'failure.json',dict(status='CPU_EQUIVALENCE_REVIEW_FAILURE_NO_RETRY',receipt=started,traceback=traceback.format_exc()))
        raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',required=True);parser.add_argument('--seed',type=int,choices=(1,2),required=True);a=parser.parse_args();main(a.prefix,a.seed)
