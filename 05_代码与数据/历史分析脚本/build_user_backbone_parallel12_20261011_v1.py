from pathlib import Path
import ast, hashlib, json, shutil, zipfile

BASE=Path('英文小论文/code/original_stem12_20261011_v1')
OUT=Path('英文小论文/code/user_backbone12_20261011_v1')
DELIVERY=Path('outputs/user_backbone12_20261011_v1')
SOURCES=Path('outputs/user_backbone_parallel_sources_20261011_v1')
USER=Path('C:/Users/jiwei/Desktop/DEIM')
P2=Path('英文小论文/data/01_C主线_已完成/C实际代码_远端_hgnetv2_light.py')
ARMS=('backbone_silu','backbone_attention','p2_skip')
assert not OUT.exists() and not DELIVERY.exists()
OUT.mkdir();DELIVERY.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,s):(OUT/n).write_text(s,encoding='utf8',newline='\n')
def js(n,v):save(n,json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def edit(n,pairs):
    s=(OUT/n).read_text(encoding='utf8')
    for a,b in pairs:
        assert a in s,(n,a);s=s.replace(a,b)
    save(n,s)
for p in BASE.iterdir():
    if p.is_file():shutil.copyfile(p,OUT/p.name)
for n in ('user_model.py','dispatch.py','preflight.py'):
    shutil.copyfile(SOURCES/n,OUT/n)
def extract(path,names,imports):
    source=Path(path).read_text(encoding='utf8');tree=ast.parse(source)
    selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in selected}==set(names)
    text=imports+'\n\n'+'\n\n'.join(ast.get_source_segment(source,n) for n in selected)+'\n'
    target=ast.parse(text)
    extracted={n.name:ast.dump(n,include_attributes=False) for n in target.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    assert all(extracted[n.name]==ast.dump(n,include_attributes=False) for n in selected)
    return text
att=USER/'engine/backbone/hgnetv2_MSCA.py'
msca=USER/'engine/extre_module/custom_nn/attention/MSCA.py'
simam=USER/'engine/extre_module/custom_nn/attention/simam.py'
save('user_attention.py',extract(msca,['MSCA_Dilated','MSCA_Lite'],'import torch\nfrom torch import nn\nfrom torch.nn import functional as F')+'\n'+
    extract(simam,['SimAM'],'')+'\n'+extract(att,['get_activation','LearnableAffineBlock','ConvBNAct','LightConvBNAct','EseModule','HG_Block'],''))
p2text=extract(P2,['conv_bn_relu','P2ToP3Residual'],'import torch\nfrom collections import OrderedDict\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom engine.backbone.hgnetv2 import HGNetv2')
p2text+='''\nclass P2SkipBackbone(HGNetv2):
    def forward(self, x):
        x=self.stem(x);outs=[];p2=None
        for idx,stage in enumerate(self.stages):
            x=stage(x)
            if idx==0:p2=x
            if idx==1:x=self.p2_skip(p2,x)
            if idx in self.return_idx:outs.append(x)
        return outs
'''
save('user_p2.py',p2text)
js('user_sources.json',dict(attention_source_path=str(att.resolve()),attention_source_sha256=sha(att),
    msca_source_path=str(msca.resolve()),msca_source_sha256=sha(msca),simam_source_path=str(simam.resolve()),simam_source_sha256=sha(simam),
    activation_source_path=str((USER/'engine/backbone/hgnetv2_Silu.py').resolve()),activation_source_sha256=sha(USER/'engine/backbone/hgnetv2_Silu.py'),
    p2_source_path=str(P2.resolve()),p2_source_sha256=sha(P2),p2_historical_result_path='英文小论文/data/01_C主线_已完成/C正式配对结果_远端.json',
    p2_historical_result_sha256=sha('英文小论文/data/01_C主线_已完成/C正式配对结果_远端.json'),
    extraction_AST_exact=True,p2_interface_adaptation='Only active P2 path retained; no detail/context branch. Injects bridge before stage2, as original HGNetv2_Light implementation. Decoder keeps3levels.'))
protocol=json.loads((BASE/'training_protocol.json').read_text(encoding='utf8'))
protocol.update(schema='user_backbone12_training_v1',execution_allowlist={a:[1,2] for a in ARMS},
    expected_parameters=None,expected_extra_parameters=None,
    mechanism='Standalone historical user changes: backbone ReLU to SiLU; original four-stage aggregation list; original width16 P2-to-P3 pixel-unshuffle bridge.',
    comparison='Same-seed cached native BASE12; unchanged132 recipe stopped at12; native fixed12 EMA. No new baseline training or forwards.',
    concurrency=dict(max_training_processes_per_gpu=4,reserved_budget_MiB={'backbone_silu':6144,'p2_skip':6144,'backbone_attention':8192},whole_GPU_headroom_MiB=512,
        existing_native_budget_MiB=6144,existing_warm_budget_MiB=8192,automatic_retry=False,serialized_closeouts=True),
    controls=dict(backbone_silu='All backbone ReLU modules become in-place SiLU; all794 common states retained.',
        backbone_attention='Aggregation list ese/simam/msca_lite/msca_dilated from user source. Aggregation conv architecture and affected initialization change; not a gate-only causal estimate.',
        p2_skip='Original compressed width16 P2 pixel-unshuffle residual injected into P3 before later stages; zero restore initialization and all794 native states retained.'))
js('training_protocol.json',protocol)
runtime=(OUT/'runtime.py').read_text(encoding='utf8')
start=runtime.index('def load_common(model, path):');end=runtime.index('def resolved_config(plan, paths):')
runtime=runtime[:start]+'''def load_common(model, path):
    from user_model import load_user_common
    assert model.geometry_arm=='baseline'
    return load_user_common(model,path)


'''+runtime[end:]
save('runtime.py',runtime)
edit('runtime.py',[("'preflight.py')","'preflight.py','user_model.py','user_attention.py','user_p2.py','user_sources.json','dispatch.py')"),
    ('User original Stem-only constructor and historical baseline evidence.','Standalone historical user backbone changes and cached same-seed BASE12.'),
    ('from original_stem_model import OriginalStemDEIM','from user_model import UserBackboneDEIM'),('register()(OriginalStemDEIM)','register()(UserBackboneDEIM)'),
    ("cfg['model']='OriginalStemDEIM'","cfg['model']='UserBackboneDEIM'"),
    ("cfg['OriginalStemDEIM']=cfg.pop('ResidualGeometryDEIM')","cfg['UserBackboneDEIM']=cfg.pop('ResidualGeometryDEIM')\n    cfg['UserBackboneDEIM']['user_arm']=plan['arm']"),
    ("protocol['schema']=='original_stem12_training_v1'","protocol['schema']=='user_backbone12_training_v1'"),
    ("plan['arm']=='original_stem' and plan['seed'] in protocol['execution_allowlist']['original_stem']","plan['arm'] in protocol['execution_allowlist'] and plan['seed'] in protocol['execution_allowlist'][plan['arm']]")])
prepare=(Path('英文小论文/code/warm_stem12_20261011_v1/prepare_training.py')).read_text(encoding='utf8')
save('prepare_training.py',prepare)
edit('prepare_training.py',[
    ('def prepare(prefix,seed):','def prepare(prefix,seed,arm):'),
    ("assert seed in protocol['execution_allowlist']['warm_stem']","assert arm in protocol['execution_allowlist'] and seed in protocol['execution_allowlist'][arm]"),
    ('DEIM/experiments/warm_stem12_20261011_v1','DEIM/experiments/user_backbone12_20261011_v1'),
    ("f'seed{seed}_warm_stem'","f'seed{seed}_{arm}'"),
    ("arm='warm_stem',seed=seed","arm=arm,seed=seed"),
    ("extra_parameters=protocol['expected_extra_parameters']","extra_parameters=None"),
    ('Native backbone.stem retained with the user enhanced residual; native matching/augmentation/loss weights/schedule remain; no new baseline forward.','One independent historical user backbone change per configuration. Common states strictly mapped; native matching/augmentation/loss weights/schedule remain; no new baseline forward.'),
    ("added_parameter_count=protocol['expected_extra_parameters']","added_parameter_count=proof['extra_parameters'],cpu_initialization_sha256=sha(bundle/'cpu_initialization.json')"),
    ("    screen.mkdir(); bundle.mkdir()",'''    from runtime import activate,register_model,load_common
    activate(paths['native_root']);register_model()
    import torch
    torch.set_num_threads(4);torch.manual_seed(seed)
    from user_model import UserBackboneDEIM
    model=UserBackboneDEIM(plan['model_recipe'],arm='baseline',user_arm=arm)
    proof=load_common(model,paths['common_init'])
    assert model.added_construction_rng_unchanged
    cpu=dict(status='PASS_STRICT_SHARED_INITIALIZATION_AND_ISOLATED_NEW_RNG',seed=seed,arm=arm,
        initialization=proof,candidate_detector_forwards=0,BASE_detector_forwards=0,CUDA_calls=0,optimizer_updates=0,training_epochs=0)
    if arm=='p2_skip':
        branch=model.backbone.p2_skip.eval()
        p2=torch.linspace(-1,1,64*20*20).reshape(1,64,20,20)
        p3=torch.linspace(-.5,.5,256*10*10).reshape(1,256,10,10)
        y=branch(p2,p3);assert torch.equal(y,p3)
        y.square().mean().backward()
        assert bool((branch.restore.weight.grad!=0).any())
        assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in branch.reduce.parameters())
        cpu['P2_synthetic_branch_zero_output_and_initial_gradient_pass']=True
    plan['extra_parameters']=proof['extra_parameters'];plan['expected_parameters']=proof['parameters']
    screen.mkdir(); bundle.mkdir()
    write(bundle/'cpu_initialization.json',cpu)'''),
    ("    assert 'torch' not in sys.modules",'    assert not torch.cuda.is_initialized()'),
    ("a=p.parse_args();prepare(a.prefix,a.seed)","p.add_argument('--arm',choices="+repr(ARMS)+",required=True)\n    a=p.parse_args();prepare(a.prefix,a.seed,a.arm)")])
# Every file remains pinned; expected parameter counts are actual CPU constructor
# receipts bound into the immutable plan rather than guessed before execution.
edit('runtime.py',[("    assert sha(paths['image_manifest'])==ident['image_manifest_sha256']","    assert sha(bundle/'cpu_initialization.json')==ident['cpu_initialization_sha256']\n    assert read(bundle/'cpu_initialization.json')['initialization']['parameters']==plan['expected_parameters']\n    assert sha(paths['image_manifest'])==ident['image_manifest_sha256']")])
edit('experiment.py',[
    ("assert self.initialization_receipt['extra_parameters'] == 5325","assert self.initialization_receipt == read(Path(bundle)/'cpu_initialization.json')['initialization']"),
    ("stem_source=read(paths['experiment_root']/'code/original_stem_source.json')","user_source=read(paths['experiment_root']/'code/user_sources.json')"),
    ('added_parameters=5325',"added_parameters=self.initialization_receipt['extra_parameters']"),
    ('Only original Stem changes. Native matching and losses remain. Efficacy not yet established.','One user backbone change per arm. Native matching and losses remain. Efficacy not yet established.'),
    ('original_stem12_identity','user_backbone12_identity')])
edit('training_audit.py',[
    ('no_new_model_parameters=False, added_parameters=5325',"no_new_model_parameters=(sum(p.numel() for p in self.model.parameters())==10231997), added_parameters=sum(p.numel() for p in self.model.parameters())-10231997"),
    ('original_stem_only=True',"user_backbone_arm=self.model.user_arm"),
    ('original Stem candidate','user backbone configuration'),
    ('historical native constructor plus Stem initialized in forked RNG; outside-Stem tensors unchanged','Historical native constructor; one isolated user backbone change; added initialization uses forked RNG; exact shared states recorded in runtime_start.'),
])
edit('results.py',[("ARMS = ('baseline', 'original_stem')","ARMS = "+repr(('baseline',)+ARMS))])
edit('supervise.py',[("assert uuid == protocol['GPU_uuid'][str(plan['seed'])] and not apps, 'Wrong or busy GPU'",
    "assert uuid == protocol['GPU_uuid'][str(plan['seed'])], 'Wrong GPU'\n    from dispatch import authorize_training\n    authorize_training(root,plan,protocol,receipt)")])
save('finish.py',Path('英文小论文/code/warm_stem12_20261011_v1/finish.py').read_text(encoding='utf8'))
edit('finish.py',[("assert same_live(owned,'pid','start_ticks'), 'Owned controller must be observed live at dispatch'",
    "actual=read(screen/'controller_start.json')\n    assert actual['pid']==owned['pid'] and str(actual['start_ticks'])==str(owned['start_ticks'])\n    assert not same_live(owned,'pid','start_ticks') and read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'")])
edit('verify_candidate_reload.py',[
    ('original_stem12_closeout_v1','user_backbone12_closeout_v1'),
    ("assert plan['arm'] == 'original_stem'","assert plan['arm'] in read(code/'training_protocol.json')['execution_allowlist']"),
    ('original_stem12_identity','user_backbone12_identity'),
    ("count == native_protocol['expected_parameters']","count == plan['expected_parameters']")])
edit('collect_terminal.py',[
    ('original_stem12_terminal_v1','user_backbone12_terminal_v1'),
    ("terminal_original_stem_seed{plan['seed']}","terminal_{plan['arm']}_seed{plan['seed']}")])
close=json.loads((OUT/'closeout_protocol.json').read_text(encoding='utf8'))
close['schema']='user_backbone12_closeout_v1'
close['files']={n:sha(OUT/n) for n in close['files']}
for n in ('user_model.py','user_attention.py','user_p2.py','user_sources.json','dispatch.py'):
    close['files'][n]=sha(OUT/n)
js('closeout_protocol.json',close)
for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
(DELIVERY/'local_code_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
archive=DELIVERY/'user_backbone12_code_v1.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):z.write(p,p.name)
print(json.dumps(dict(status='BUILT_FROZEN_STANDALONE_USER_BACKBONE_CONFIGS',files=len(manifest),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),protocol_sha256=sha(OUT/'training_protocol.json'),arms=ARMS)))
