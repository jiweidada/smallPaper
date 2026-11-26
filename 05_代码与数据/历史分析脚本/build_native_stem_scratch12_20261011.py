"""Build once: native Stem default initialization, all other common tensors retained."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile

BASE=Path('英文小论文/code/original_stem12_20261011_v1')
OUT=Path('英文小论文/code/native_stem_scratch12_20261011_v1')
DELIVERY=Path('outputs/native_stem_scratch12_20261011_v1')
SOURCE=Path('英文小论文/code/p2_candidate_seed1_full132_20261007_v1/engine/backbone/hgnetv2.py')
assert not OUT.exists(), 'Never overwrite or restart a deployed configuration'
OUT.mkdir();DELIVERY.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,s):(OUT/n).write_text(s,encoding='utf8',newline='\n')
def js(n,v):save(n,json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def edit(n,pairs):
    s=(OUT/n).read_text(encoding='utf8')
    for a,b in pairs:
        assert a in s,(n,a)
        s=s.replace(a,b)
    save(n,s)
for p in BASE.iterdir():
    if p.is_file():
        n=p.name.replace('original_stem','native_stem_scratch').replace('original_backbone','native_backbone')
        if p.suffix in ('.py','.json','.txt'):
            text=p.read_text(encoding='utf8').replace('original_stem','native_stem_scratch').replace('OriginalStem','NativeStemScratch').replace('original_backbone','native_backbone')
            save(n,text)
        else:shutil.copyfile(p,OUT/n)

source=SOURCE.read_text(encoding='utf8')
names=('LearnableAffineBlock','ConvBNAct','StemBlock')
parts=[ast.get_source_segment(source,node) for node in ast.parse(source).body
       if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
assert len(parts)==3
save('native_stem_scratch.py','"""Unmodified native Stem definitions; no pretrained Stem state is loaded."""\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'+'\n\n'.join(parts)+'\n')
shutil.copyfile(SOURCE,OUT/'native_backbone_source.txt')
js('native_stem_scratch_source.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),
    extracted_definitions=list(names),extracted_file_sha256=sha(OUT/'native_stem_scratch.py'),
    activation='Native ReLU and LAB, unchanged',
    scope='Native Stem architecture unchanged; its default initialization replaces pretrained Stem tensors and BN statistics only.'))
save('native_stem_scratch_model.py','''"""Native Stem from scratch; no outside-Stem tensor or data RNG change."""
import torch
from geometry import ResidualGeometryDEIM
from native_stem_scratch import StemBlock

class NativeStemScratchDEIM(ResidualGeometryDEIM):
    __inject__=[]
    __share__=[]
    def __init__(self,recipe,arm='baseline'):
        assert arm=='baseline'
        super().__init__(recipe,arm=arm)
        cfg=recipe['HGNetv2']
        assert cfg['name']=='B0' and cfg.get('freeze_at',-1)<0 and cfg.get('freeze_norm') is False
        channels=self.backbone.arch_configs['B0']['stem_channels']
        assert channels==[3,16,16]
        self.old_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed())
            self.backbone.stem=StemBlock(*channels,use_lab=cfg['use_lab'])
        self.new_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())
        assert self.old_stem_parameter_count==self.new_stem_parameter_count==6474
''')

protocol=json.loads((OUT/'training_protocol.json').read_text())
protocol.update(expected_parameters=10231997,expected_extra_parameters=0,
    native_stem_scratch_source_sha256=sha(OUT/'native_stem_scratch.py'),
    mechanism='User-authorized new reference: native Stem default initialization, original native structure. Keep all754 outside-Stem common initialization tensors exact.',
    training_start='All40 native Stem state tensors use seed-specific default initialization in the same forked RNG convention as enhanced Stem; no pretrained Stem tensor/statistic is loaded.',
    acceptance='A reference control, not an improved model claim. Compare same-seed enhanced Stem against this native scratch Stem at fixed12 EMA. Also retain existing pretrained BASE12 comparison and original two-seed APs>BASE/AP>=BASE goal gate.',
    new_reference_authorization='Human requested: 你现在在加一个基线stem 没有预训练权重的 当基线',
    concurrency=dict(max_training_processes_per_gpu=3,minimum_free_before_launch_MiB=6144,
        coordination='Hold only the owned existing component conductor while waiting for a free slot and running this new reference. Existing trainers are uninterrupted. Resume conductor after this reference training exits; serialize saved-EMA replay after existing closeouts.',
        automatic_retry=False),
    existing_component_conductor={
        '1':dict(pid=13381,start_ticks='476867385'),
        '2':dict(pid=15065,start_ticks='254060477')},
    existing_component_code_manifest=json.loads((DELIVERY.parent/'stem_component12_20261011_v1/local_code_manifest.json').read_text()))
js('training_protocol.json',protocol)
edit('runtime.py',[("'preflight.py')","'preflight.py', 'coordinate.py')")])
edit('experiment.py',[
    ("self.initialization_receipt['extra_parameters'] == 5325","self.initialization_receipt['extra_parameters'] == 0"),
    ('added_parameters=5325,','added_parameters=0,'),
    ('Only original Stem changes. Native matching and losses remain. Efficacy not yet established.',
     'Only native Stem initialization differs from pretrained BASE. This is a reference control, not an efficacy claim.')])
edit('training_audit.py',[
    ('no_new_model_parameters=False, added_parameters=5325,','no_new_model_parameters=True, added_parameters=0,'),
    ('historical native constructor plus Stem initialized in forked RNG; outside-Stem tensors unchanged',
     'historical native constructor plus original native Stem initialized in forked RNG; outside-Stem tensors unchanged')])
edit('supervise.py',[(
    "assert uuid == protocol['GPU_uuid'][str(plan['seed'])] and not apps, 'Wrong or busy GPU'",
    "assert uuid == protocol['GPU_uuid'][str(plan['seed'])], 'Wrong GPU'\n    from coordinate import authorize_child\n    authorize_child(root,plan,protocol)")])
edit('finish.py',[(
    "assert same_live(owned,'pid','start_ticks'), 'Owned controller must be observed live at dispatch'",
    "record=read(screen/'controller_start.json')\n    assert record['pid']==owned['pid'] and str(record['start_ticks'])==str(owned['start_ticks'])\n    assert same_live(owned,'pid','start_ticks') or read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'")])
save('preflight.py','''"""Native architecture/equivalence and scratch initialization check; zero updates."""
import argparse,json,torch
from pathlib import Path
from runtime import verify_bundle,activate,register_model,load_common,write
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
receipt,plan,paths=verify_bundle(a.bundle)
activate(paths['native_root']);register_model()
from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.backbone.hgnetv2 import StemBlock as CanonicalNativeStem
dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
cfg=YAMLConfig(str(Path(a.bundle)/'resolved.yml'));model=cfg.model
init=load_common(model,paths['common_init'])
assert init['parameters']==10231997 and init['extra_parameters']==0
assert init['exact_shared_tensors']==754 and init['state_tensors']==794
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(plan['seed'])
    expected=CanonicalNativeStem(3,16,16,use_lab=True)
actual=model.backbone.stem
assert actual.state_dict().keys()==expected.state_dict().keys()
assert all(torch.equal(v,actual.state_dict()[k]) for k,v in expected.state_dict().items())
bn=[m for m in actual.modules() if isinstance(m,torch.nn.BatchNorm2d)]
assert len(bn)==5 and all(torch.equal(m.running_mean,torch.zeros_like(m.running_mean))
    and torch.equal(m.running_var,torch.ones_like(m.running_var)) and m.num_batches_tracked.item()==0 for m in bn)
actual.eval();expected.eval()
x=torch.linspace(-1,1,3*64*64).reshape(1,3,64,64)
with torch.no_grad():assert torch.equal(actual(x),expected(x))
stem=actual.cuda().train();shapes=[]
for side in (640,832):
    x=torch.linspace(-1,1,side*side,device='cuda').reshape(1,1,side,side).expand(4,3,-1,-1).contiguous()
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        y=stem(x);loss=y.float().square().mean()
    assert tuple(y.shape)==(4,16,side//4,side//4) and torch.isfinite(y).all()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stem.parameters())
    shapes.append(list(y.shape));stem.zero_grad(set_to_none=True)
result=dict(status='NATIVE_STEM_SCRATCH_SHAPE_INITIALIZATION_EQUIVALENCE_PASS',seed=plan['seed'],arm=plan['arm'],
    initialization=init,shapes=shapes,all40_initial_stem_tensors_match_seeded_canonical_native=True,
    native_stem_forward_exact=True,all5_BN_initial_statistics_default=True,
    all_stem_trainable_tensors_receive_finite_gradients=True,full_model_forwards=0,optimizer_updates=0,
    pretrained_BASE_forwards=0,peak_allocated_bytes=torch.cuda.max_memory_allocated())
write(paths['screen_root']/'preflight.json',result);print(json.dumps(result),flush=True)
''')
save('coordinate.py','''"""Insert an authorized scratch reference into the existing three-job GPU schedule."""
from pathlib import Path
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from runtime import HERE,read,write,sha

def live(record):
    try:
        fields=Path(f"/proc/{record['pid']}/stat").read_text().split()
        return str(record['start_ticks'])==fields[21] and fields[2]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False

def gpu():
    fields=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return dict(uuid=fields[0].strip(),total=int(fields[1]),used=int(fields[2]),free=int(fields[3]),host_compute_processes=apps)

def invoke(script,args,log):
    with Path(log).open('xb') as f:
        return subprocess.run([sys.executable,'-B',str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')).returncode

def authorize_child(screen,plan,protocol):
    allowance=read(screen/'reference_allowance.json');parent=read(screen.parent/'reference_start.json')
    assert allowance['parent']==parent and os.getppid()==parent['pid'] and live(parent)
    assert allowance['seed']==plan['seed'] and allowance['arm']==plan['arm']
    assert allowance['protocol_sha256']==sha(HERE/'training_protocol.json')
    assert read(screen/'preflight.json')['status']=='NATIVE_STEM_SCRATCH_SHAPE_INITIALIZATION_EQUIVALENCE_PASS'
    held=read(screen.parent/'component_conductor_held.json')['conductor']
    assert live(held) and Path(f"/proc/{held['pid']}/stat").read_text().split()[2]=='T'
    state=gpu()
    assert state['uuid']==protocol['GPU_uuid'][str(plan['seed'])]
    assert state['free']>=protocol['concurrency']['minimum_free_before_launch_MiB']
    assert len(state['host_compute_processes'])<protocol['concurrency']['max_training_processes_per_gpu']
    write(screen/'reference_resource_receipt.json',state)

def main(a):
    root=HERE.parent;protocol=read(HERE/'training_protocol.json')
    assert root==Path(a.prefix)/'DEIM/experiments/native_stem_scratch12_20261011_v1'
    with (root/'reference_once.lock').open('x') as f:f.write(str(os.getpid()))
    parent=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=a.seed,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(HERE/'training_protocol.json'))
    write(root/'reference_start.json',parent)
    oldroot=Path(a.prefix)/'DEIM/experiments/stem_component12_20261011_v1'
    conductor=protocol['existing_component_conductor'][str(a.seed)]
    for n,d in protocol['existing_component_code_manifest'].items():assert sha(oldroot/'code'/n)==d
    assert live(conductor)
    held=False;child=None
    try:
        assert invoke('prepare_training.py',['--prefix',a.prefix,'--seed',str(a.seed)],root/'prepare.log')==0
        screen=root/f'seed{a.seed}_native_stem_scratch'
        assert invoke('preflight.py',['--bundle',str(screen/'bundle')],root/'preflight.log')==0
        assert live(conductor)
        os.kill(conductor['pid'],signal.SIGSTOP);held=True
        while Path(f"/proc/{conductor['pid']}/stat").read_text().split()[2]!='T':time.sleep(.05)
        write(root/'component_conductor_held.json',dict(conductor=conductor,signal='SIGSTOP',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='Only waiting component conductor; existing trainers/controllers and full finisher untouched. Insert new authorized reference at next free GPU slot.'))
        while True:
            state=gpu();assert state['uuid']==protocol['GPU_uuid'][str(a.seed)]
            status=dict(phase='waiting_free_training_slot',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=state)
            temporary=root/'reference_live.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(root/'reference_live.json')
            assert live(conductor)
            if len(state['host_compute_processes'])<3 and state['free']>=6144:break
            time.sleep(15)
        plan=read(screen/'bundle/plan.json')
        write(screen/'reference_allowance.json',dict(parent=parent,seed=a.seed,arm=plan['arm'],protocol_sha256=sha(HERE/'training_protocol.json')))
        with (screen/'supervisor.log').open('xb') as f:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'supervise.py'),'--bundle',str(screen/'bundle')],
                cwd=screen,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        owned=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
        write(root/'reference_dispatch.json',owned)
        while child.poll() is None:
            temporary=root/'reference_live.tmp';temporary.write_text(json.dumps(dict(phase='training_reference',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller=owned,gpu=gpu()),indent=2));temporary.replace(root/'reference_live.json')
            time.sleep(15)
        assert live(conductor)
        os.kill(conductor['pid'],signal.SIGCONT);held=False
        write(root/'component_conductor_resumed.json',dict(conductor=conductor,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_controller_exit=child.returncode))
        assert child.returncode==0, 'Reference failed; no retry'
        # Existing immutable closeouts own their GPU-idle replay phase.
        while live(conductor):time.sleep(15)
        assert (oldroot/'parallel_completion.json').exists() or (oldroot/'parallel_failure.json').exists()
        while gpu()['host_compute_processes']:time.sleep(15)
        assert invoke('finish.py',['--bundle',str(screen/'bundle'),'--controller-pid',str(owned['pid']),
            '--controller-start-ticks',owned['start_ticks']],root/'finish.log')==0
        write(root/'reference_completion.json',dict(status='NATIVE_STEM_SCRATCH_FIXED12_FINISHED',seed=a.seed,
            extra_epochs=0,automatic_retry=False,pretrained_BASE_forwards=0))
    except BaseException:
        write(root/'reference_failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        if held:
            if child is not None:
                while child.poll() is None:time.sleep(15)
            if live(conductor):
                os.kill(conductor['pid'],signal.SIGCONT)
                if not (root/'component_conductor_resumed.json').exists():
                    write(root/'component_conductor_resumed.json',dict(conductor=conductor,finally_cleanup=True,
                        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    main(p.parse_args())
''')
close=json.loads((OUT/'closeout_protocol.json').read_text())
close['files']={n:sha(OUT/n) for n in close['files']}
js('closeout_protocol.json',close)
for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
definitions=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
left,right=definitions(SOURCE),definitions(OUT/'native_stem_scratch.py')
assert all(left[n]==right[n] for n in names)
manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
(DELIVERY/'local_code_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
archive=DELIVERY/'native_stem_scratch12_code_v1.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):z.write(p,p.name)
print(json.dumps(dict(files=len(manifest),zip_sha256=sha(archive),protocol_sha256=sha(OUT/'training_protocol.json'),native_stem_source_AST_exact=True)))
