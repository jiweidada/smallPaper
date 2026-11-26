from pathlib import Path
import ast, hashlib, json, shutil, zipfile

BASE=Path('英文小论文/code/original_stem12_20261011_v1')
OUT=Path('英文小论文/code/stem_component12_20261011_v1')
DELIVERY=Path('outputs/stem_component12_20261011_v1')
ARMS=('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')
assert not OUT.exists(), 'Build once; never overwrite deployed code'
OUT.mkdir(); DELIVERY.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,s): (OUT/n).write_text(s,encoding='utf8',newline='\n')
def js(n,v): save(n,json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def edit(n,pairs):
    s=(OUT/n).read_text(encoding='utf8')
    for a,b in pairs:
        assert a in s,(n,a)
        s=s.replace(a,b)
    save(n,s)
for p in BASE.iterdir():
    if p.is_file(): shutil.copyfile(p,OUT/p.name)

save('stem_controls.py','''"""One changed operation per arm; all original parameters retained."""
import torch
from torch.nn import functional as F
from original_stem import EnhancedStemBlock

ARMS=('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')

class ControlledStemBlock(EnhancedStemBlock):
    def set_control(self,control):
        assert control in ARMS or control=='full'
        self.control=control
        if control=='early_final_downsample':
            self.fusion[1].conv.stride=(2,2)
            self.final_downsample.conv.stride=(1,1)

    def forward(self,x):
        if self.control=='detail_lowres':
            detail=self.detail_branch(F.avg_pool2d(x,2,2))
        else:
            detail=F.avg_pool2d(self.detail_branch(x),2,2)
        main=self.main_branch(x)
        source=x if self.control=='highfreq_rgb' else x-F.avg_pool2d(x,3,1,1)
        high=F.interpolate(self.high_freq_enhance(source),size=main.shape[-2:],mode='bilinear',align_corners=False)
        cat=torch.cat([main,detail,high],1)
        att=self.spatial_attention(cat)
        if self.control=='attention_spatial_mean':
            att=att.mean(dim=(-2,-1),keepdim=True)
        return self.final_downsample(self.fusion(cat*att+cat))
''')
edit('original_stem_model.py',[
    ('from original_stem import EnhancedStemBlock','from stem_controls import ControlledStemBlock'),
    ("def __init__(self,recipe,arm='baseline'):","def __init__(self,recipe,arm='baseline',control=None):"),
    ("self.backbone.stem=EnhancedStemBlock(*channels,use_lab=cfg['use_lab'],act_type='relu')","self.backbone.stem=ControlledStemBlock(*channels,use_lab=cfg['use_lab'],act_type='relu')\n            self.backbone.stem.set_control(control)")])
protocol=json.loads((OUT/'training_protocol.json').read_text())
protocol.update(schema='stem_component12_training_v1',execution_allowlist={a:[1,2] for a in ARMS},
    mechanism='Single-factor controls of the user original Stem. All original parameters, initialization, batch4 and native training recipe retained.',
    comparison='Each control versus same-seed full original Stem at fixed12 EMA, and cached BASE12. Full-minus-control is a conditional contribution, not an additive or unique causal decomposition.',
    original_full_code_manifest={p.name:sha(p) for p in BASE.iterdir() if p.is_file()},
    controls={
        'detail_lowres':'Average-pool input before detail convs instead of pooling their output; tests extracting detail at full resolution. Changes effective receptive field and compute.',
        'highfreq_rgb':'Replace x-average_pool3(x) by raw x only; tests explicit high-pass content versus an equally parameterized RGB branch.',
        'attention_spatial_mean':'Replace sigmoid attention map by its spatial mean per image; retains trainable attention and average residual gain, removes spatial variation.',
        'early_final_downsample':'Move stride2 from final conv3 to preceding fusion conv3; output stride4 and all weights unchanged; tests delaying second downsampling.'},
    concurrency=dict(max_training_processes_per_gpu=3,minimum_free_before_launch_MiB=6144,
        capacity_basis='Prior same-recipe Mosaic peak reserved 5179965440 bytes plus CUDA context and Stem overhead; full+2 initial concurrent jobs.',
        queue=list(ARMS),automatic_retry=False,serialize_saved_ema_replay_after_training=True))
js('training_protocol.json',protocol)
edit('runtime.py',[
    ("'preflight.py')","'preflight.py', 'stem_controls.py', 'parallel.py')"),
    ("    cfg['OriginalStemDEIM']=cfg.pop('ResidualGeometryDEIM')","    cfg['OriginalStemDEIM']=cfg.pop('ResidualGeometryDEIM')\n    cfg['OriginalStemDEIM']['control']=plan['arm']"),
    ("protocol['schema']=='original_stem12_training_v1'","protocol['schema']=='stem_component12_training_v1'"),
    ("plan['arm']=='original_stem' and plan['seed'] in protocol['execution_allowlist']['original_stem']","plan['arm'] in protocol['execution_allowlist'] and plan['seed'] in protocol['execution_allowlist'][plan['arm']]")])
edit('prepare_training.py',[
    ('def prepare(prefix,seed):','def prepare(prefix,seed,arm):'),
    ("assert seed in protocol['execution_allowlist']['original_stem']","assert arm in protocol['execution_allowlist'] and seed in protocol['execution_allowlist'][arm]"),
    ('DEIM/experiments/original_stem12_20261011_v1','DEIM/experiments/stem_component12_20261011_v1'),
    ("f'seed{seed}_original_stem'","f'seed{seed}_{arm}'"),
    ("arm='original_stem',seed=seed","arm=arm,seed=seed"),
    ("a=p.parse_args();prepare(a.prefix,a.seed)","p.add_argument('--arm',choices="+repr(ARMS)+",required=True)\n    a=p.parse_args();prepare(a.prefix,a.seed,a.arm)")])
edit('results.py',[("ARMS = ('baseline', 'original_stem')","ARMS = "+repr(('baseline','original_stem')+ARMS))])
edit('verify_candidate_reload.py',[
    ('original_stem12_closeout_v1','stem_component12_closeout_v1'),
    ("assert plan['arm'] == 'original_stem'","assert plan['arm'] in read(code/'training_protocol.json')['execution_allowlist']")])
edit('collect_terminal.py',[
    ('original_stem12_terminal_v1','stem_component12_terminal_v1'),
    ("terminal_original_stem_seed{plan['seed']}","terminal_{plan['arm']}_seed{plan['seed']}")])
edit('finish.py',[("assert same_live(owned,'pid','start_ticks'), 'Owned controller must be observed live at dispatch'",
    "record=read(screen/'controller_start.json')\n    assert record['pid']==owned['pid'] and str(record['start_ticks'])==str(owned['start_ticks'])\n    assert same_live(owned,'pid','start_ticks') or read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'")])
edit('supervise.py',[(
    "assert uuid == protocol['GPU_uuid'][str(plan['seed'])] and not apps, 'Wrong or busy GPU'",
    "assert uuid == protocol['GPU_uuid'][str(plan['seed'])], 'Wrong GPU'\n    from parallel import authorize_child\n    authorize_child(root,plan,protocol)"
)])
save('preflight.py','''"""Test original equivalence, matched initialization and finite controlled gradients; zero updates."""
import argparse,copy,json,torch
from pathlib import Path
from runtime import verify_bundle,activate,register_model,load_common,write
from original_stem import EnhancedStemBlock
from stem_controls import ControlledStemBlock
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
receipt,plan,paths=verify_bundle(a.bundle)
activate(paths['native_root']);register_model()
from engine.core import YAMLConfig
from engine.misc import dist_utils
dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
cfg=YAMLConfig(str(Path(a.bundle)/'resolved.yml'));model=cfg.model
init=load_common(model,paths['common_init'])
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(plan['seed'])
    original=EnhancedStemBlock(3,16,16,use_lab=True,act_type='relu')
assert all(torch.equal(v,model.backbone.stem.state_dict()[k]) for k,v in original.state_dict().items())
# The duplicated forward must be exactly the original at the full setting.
full=ControlledStemBlock(3,16,16,use_lab=True,act_type='relu');full.set_control('full')
full.load_state_dict(original.state_dict());full.eval();original.eval()
x=torch.linspace(-1,1,3*64*64).reshape(1,3,64,64)
with torch.no_grad():
    assert torch.equal(full(x),original(x))
stem=model.backbone.stem.cuda().train();shapes=[]
for side in (640,832):
    x=torch.linspace(-1,1,side*side,device='cuda').reshape(1,1,side,side).expand(4,3,-1,-1).contiguous()
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        y=stem(x);loss=y.float().square().mean()
    assert tuple(y.shape)==(4,16,side//4,side//4) and torch.isfinite(y).all()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stem.parameters())
    shapes.append(list(y.shape));stem.zero_grad(set_to_none=True)
result=dict(status='CONTROL_SHAPE_GRADIENT_AND_INITIALIZATION_PASS',seed=plan['seed'],arm=plan['arm'],
    initialization=init,shapes=shapes,all_initial_stem_tensors_match_full=True,
    full_forward_exactly_original=True,all_stem_trainable_tensors_receive_finite_gradients=True,
    full_model_forwards=0,optimizer_updates=0,baseline_forwards=0,peak_allocated_bytes=torch.cuda.max_memory_allocated())
write(paths['screen_root']/'preflight.json',result);print(json.dumps(result),flush=True)
''')
# Scheduler is a separately authored implementation, copied before freezing.
shutil.copyfile(DELIVERY/'parallel_source.py',OUT/'parallel.py')
close=json.loads((OUT/'closeout_protocol.json').read_text());close['schema']='stem_component12_closeout_v1'
close['files']={n:sha(OUT/n) for n in close['files']};close['files']['stem_controls.py']=sha(OUT/'stem_controls.py')
js('closeout_protocol.json',close)
for p in OUT.glob('*.py'): ast.parse(p.read_text(encoding='utf8'))
manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
(DELIVERY/'local_code_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
with zipfile.ZipFile(DELIVERY/'stem_component12_code_v1.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,p.name)
print(json.dumps(dict(files=len(manifest),zip_sha256=sha(DELIVERY/'stem_component12_code_v1.zip'),protocol_sha256=sha(OUT/'training_protocol.json'))))
