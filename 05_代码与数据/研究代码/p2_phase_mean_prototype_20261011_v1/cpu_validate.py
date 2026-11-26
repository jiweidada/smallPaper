"""Engineering only, on the actual frozen user P2 branch; no detector or training."""
from pathlib import Path
import argparse,ast,copy,datetime,hashlib,json,os,sys,traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from phase_control import controlled_class,phase_mean

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def states_equal(a,b):
    x,y=a.state_dict(),b.state_dict()
    return set(x)==set(y) and all(torch.equal(x[n],y[n]) for n in x)
def main(prefix,seed):
    assert seed in (1,2) and prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    protocol=read(HERE/'protocol.json')
    assert {n:sha(HERE/n) for n in protocol['prototype_sources']}==protocol['prototype_sources']
    user=Path(prefix)/'DEIM/experiments/user_backbone12_20261011_v1/code'
    before={p.name:sha(p) for p in user.iterdir() if p.is_file()}
    assert before==protocol['frozen_user_files'] and len(before)==28
    output=HERE.parent/f'seed{seed}';output.mkdir(exist_ok=False)
    receipt=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=seed,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(HERE/'protocol.json'),
        prototype_sources=protocol['prototype_sources'],frozen_user_files_verified=28,automatic_retry=False,
        candidate_detector_forwards=0,BASE_detector_forwards=0,training_epochs=0,optimizer_updates=0,CUDA_available=False)
    write(output/'start.json',receipt)
    try:
        torch.set_num_threads(2)
        assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
        # Execute the exact two declarations from the frozen file. The unused backbone
        # class/engine import is excluded; no native detector is instantiated or called.
        source=(user/'user_p2.py').read_text(encoding='utf8');tree=ast.parse(source)
        declarations=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('conv_bn_relu','P2ToP3Residual')]
        assert [n.name for n in declarations]==['conv_bn_relu','P2ToP3Residual']
        namespace=dict(torch=torch,nn=nn,F=F,OrderedDict=OrderedDict)
        text='\n\n'.join(ast.get_source_segment(source,n) for n in declarations)
        exec(compile(text,str(user/'user_p2.py')+'::exact_branch_declarations','exec'),namespace)
        Branch=namespace['P2ToP3Residual'];Control=controlled_class(Branch)
        cases=[]
        # Exact permutation and DC/contrast properties, on reduced features only.
        x=torch.arange(24,dtype=torch.float64).reshape(1,2,2,6)
        f=F.pixel_unshuffle(x,2);p=phase_mean(f)
        assert torch.equal(F.pixel_shuffle(f,2),x)
        assert torch.equal(p.reshape(1,2,4,1,3).mean(2),f.reshape(1,2,4,1,3).mean(2))
        assert torch.equal(p,phase_mean(p))
        contrast=torch.tensor([1.,-1.,-1.,1.]).reshape(1,4,1,1)
        assert torch.count_nonzero(phase_mean(contrast))==0 and torch.count_nonzero(contrast)==4
        probe=torch.randn(1,8,3,5,dtype=torch.float64,requires_grad=True)
        phase_mean(probe).square().sum().backward()
        g=probe.grad.reshape(1,2,4,3,5)
        assert torch.equal(g[:,:,0],g[:,:,1]) and torch.equal(g[:,:,0],g[:,:,3])
        for bad in (torch.zeros(1,7,2,2),torch.zeros(4,2,2)):
            try:phase_mean(bad)
            except ValueError:pass
            else:raise AssertionError('Invalid phase shape was accepted')
        cases.append(dict(name='reduced_feature_permutation_DC_and_zero_sum_contrast',passed=True,
            entire_64_to16_branch_lossless=False,phase_mean_keeps_only_within_cell_mean=True,
            negative_invalid_shape_cases=2,projection_gradient_equal_across_phases=True))
        branch_forwards=0
        for h,w in ((160,160),(208,208),(161,159),(5,7)):
            rng_before=torch.get_rng_state().clone()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);a=Branch(64,256,16)
            assert torch.equal(rng_before,torch.get_rng_state())
            b=copy.deepcopy(a);b.__class__=Control
            assert torch.equal(rng_before,torch.get_rng_state())
            # Changing only forward retains the same states without consuming construction RNG.
            assert states_equal(a,b) and len(a.state_dict())==7
            assert sum(v.numel() for v in a.parameters())==sum(v.numel() for v in b.parameters())==17440
            assert torch.count_nonzero(a.restore.weight)==torch.count_nonzero(b.restore.weight)==0
            generator=torch.Generator().manual_seed(seed*10000+h*10+w)
            p2=torch.randn(2,64,h,w,generator=generator)
            p3=torch.randn(2,256,(h+1)//2,(w+1)//2,generator=generator)
            target=torch.randn(p3.shape,generator=generator)
            # Check actual source padding/folding, not a separately invented pooling branch.
            for module in (a,b):
                y=module(p2,p3);branch_forwards+=1
                assert y.shape==p3.shape and torch.equal(y,p3)
                (y-target).square().mean().backward()
                assert module.restore.weight.grad is not None and torch.isfinite(module.restore.weight.grad).all()
                assert torch.count_nonzero(module.restore.weight.grad)>0
                assert all(v.grad is not None and torch.count_nonzero(v.grad)==0 for v in module.reduce.parameters())
            assert states_equal(a,b),'Same reduce inputs must give the same BN updates'
            rg=b.restore.weight.grad.reshape(256,16,4)
            grad_delta=max(float((rg[:,:,0]-rg[:,:,i]).abs().max()) for i in range(1,4))
            assert torch.allclose(rg,rg[:,:,0:1].expand_as(rg),rtol=1e-6,atol=1e-7)
            cases.append(dict(name='actual_zero_initialized_branch_shape_and_gradient',input=[2,64,h,w],output=list(p3.shape),
                identical7_initial_states=True,parameters_each=17440,initial_output_exact_P3=True,
                restore_grad_nonzero=True,reduce_initial_grad_zero=True,mean_restore_phase_gradient_max_abs_difference=grad_delta))
        # Demonstrate a within-cell contrast can reach a phase-aware projection and is
        # removed by the control. This is a feature-space property, not a trained AP result.
        weights=torch.tensor([1.,-1.,-1.,1.]).reshape(1,4,1,1)
        original=F.conv2d(contrast,weights);controlled=F.conv2d(phase_mean(contrast),weights)
        assert original.item()==4 and controlled.item()==0
        cases.append(dict(name='fixed_projection_distinguishes_feature_contrast',original_output=4.,control_output=0.,passed=True))
        assert before=={p.name:sha(p) for p in user.iterdir() if p.is_file()}
        assert not torch.cuda.is_initialized() and not any(n=='engine' or n.startswith('engine.') for n in sys.modules)
        result=dict(status='PASS_P2_PHASE_MEAN_CPU_OPERATOR_AND_BRANCH_ENGINEERING_ONLY',seed=seed,cases=cases,
            CPU_branch_forwards=branch_forwards,candidate_detector_forwards=0,BASE_detector_forwards=0,
            CUDA_calls=0,optimizer_updates=0,training_epochs=0,new_accuracy_experiments=0,
            frozen_user_files_before_after_exact=True,frozen_user_source_SHA=before['user_p2.py'],
            protocol_sha256=sha(HERE/'protocol.json'),source_sha256=sha(__file__),
            branch_declarations_sha256=hashlib.sha256(text.encode('utf8')).hexdigest(),
            scope='Synthetic reduced-feature and actual frozen branch engineering only; no dataset/model loss/AMP/AP proof, no full-detector common-state assertion. Fourfold rearrangement alone is reversible; 64-to16 compression is not.',
            next_gate='Review original P2 fixed12 both seeds, then prepare a separately bound full-detector experiment and preflight if this contrast is scientifically warranted. No automatic training launch or retry.',
            root_proven=False,novelty_proven=False,goal_complete=False)
        write(output/'report.json',result)
        write(output/'completion.json',dict(status=result['status'],exit_code=0,report_sha256=sha(output/'report.json'),receipt=receipt))
    except BaseException:
        write(output/'failure.json',dict(status='CPU_ENGINEERING_FAILURE_NO_TRAINING_OR_RETRY',traceback=traceback.format_exc(),receipt=receipt));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    a=p.parse_args();main(a.prefix,a.seed)
