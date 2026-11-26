from pathlib import Path
import ast, hashlib, json, shutil, zipfile

BASE=Path('英文小论文/code/support_matching12_20261011_v1')
OUT=Path('英文小论文/code/original_stem12_20261011_v1')
OUT.mkdir(exist_ok=True)
SOURCE=Path('英文小论文/code/p2_candidate_seed1_full132_20261007_v1/engine/backbone/hgnetv2_MSCA_EnhancedStem.py')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,s): (OUT/n).write_text(s,encoding='utf8',newline='\n')
def change(s,a,b):
    assert a in s, a[:160]
    return s.replace(a,b)
def js(n,v): save(n,json.dumps(v,ensure_ascii=False,indent=2)+'\n')

for n in ('geometry.py','residual.py','results.py','stage_trace.py','evaluator_binding.py','evaluator_sources.json','source_binding.json','reuse_baseline.py','supervise.py'):
    shutil.copyfile(BASE/n,OUT/n)
save('results.py',(BASE/'results.py').read_text().replace("ARMS = ('baseline', 'support_cost')","ARMS = ('baseline', 'original_stem')"))
src=SOURCE.read_text(encoding='utf8')
names=('get_activation','LearnableAffineBlock','ConvBNAct','EnhancedStemBlock')
parts=[ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
assert len(parts)==4
save('original_stem.py','"""Unmodified definitions extracted from the user original backbone source."""\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'+'\n\n'.join(parts)+'\n')
shutil.copyfile(SOURCE,OUT/'original_backbone_source.txt')
js('original_stem_source.json',{'source':str(SOURCE),'source_sha256':sha(SOURCE),'extracted_definitions':list(names),'extracted_file_sha256':sha(OUT/'original_stem.py'),'activation':'relu, as in the native baseline; original module default','scope':'Replace only backbone.stem; do not import the old combined MSCA/ESE/SiLU backbone configuration.'})
save('original_stem_model.py','''"""Replace only the stem after the historical native constructor."""
import torch
from geometry import ResidualGeometryDEIM
from original_stem import EnhancedStemBlock

class OriginalStemDEIM(ResidualGeometryDEIM):
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
        # Isolate new-module initialization from the established data/DN RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed())
            self.backbone.stem=EnhancedStemBlock(*channels,use_lab=cfg['use_lab'],act_type='relu')
        self.new_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())
''')

protocol=json.loads((BASE/'training_protocol.json').read_text())
for k in ('matching_coefficient','matching_core_sha256','native_matcher_sha256','entry_gate_sha256','extra_parameters'): protocol.pop(k,None)
protocol.update(schema='original_stem12_training_v1',execution_allowlist={'original_stem':[1,2]},
    expected_parameters=10237322,expected_extra_parameters=5325,
    original_stem_source_sha256=sha(OUT/'original_stem.py'),
    mechanism='User original EnhancedStemBlock, ReLU and LAB as baseline. Only backbone.stem is replaced; matching/loss/queries/augmentation/schedule unchanged.',
    training_start='Exact common initial tensors outside backbone.stem; seed-specific default initialization of original stem in forked RNG.',
    monitoring='Native every-epoch EMA validation; first100 input/RNG identity; fixed12 raw endpoint and saved EMA reload.',
    acceptance='Both seeds fixed12 EMA APs strictly greater than existing same-seed BASE12 and AP not lower. No new baseline forward/training, no extension or automatic retry.')
js('training_protocol.json',protocol)

s=(BASE/'runtime.py').read_text()
s=change(s,'Frozen baseline constructor with a separately identified FDR grid change.','User original Stem-only constructor and historical baseline evidence.')
a=s.index('IMPLEMENTATION = '); b=s.index('\n\n\ndef sha',a)
files=('runtime.py','prepare_training.py','experiment.py','supervise.py','reuse_baseline.py','training_audit.py','geometry.py','residual.py','original_stem.py','original_stem_model.py','original_stem_source.json','original_backbone_source.txt','results.py','stage_trace.py','evaluator_binding.py','evaluator_sources.json','source_binding.json','training_protocol.json','verify_candidate_reload.py','collect_terminal.py','finish.py','closeout_protocol.json','preflight.py')
s=s[:a]+'IMPLEMENTATION = '+repr(files)+s[b:]
s=change(s,'from geometry import ResidualGeometryDEIM\n    register()(ResidualGeometryDEIM)','from original_stem_model import OriginalStemDEIM\n    register()(OriginalStemDEIM)')
a=s.index('    assert current.keys()==state.keys()');b=s.index('\n\n\ndef resolved_config',a)
s=s[:a]+'''    keep={n:v for n,v in state.items() if not n.startswith('backbone.stem.')}
    assert set(keep)=={n for n in current if not n.startswith('backbone.stem.')}
    assert all(current[n].shape==v.shape and current[n].dtype==v.dtype for n,v in keep.items())
    result=model.load_state_dict(keep,strict=False)
    assert not result.unexpected_keys and all(n.startswith('backbone.stem.') for n in result.missing_keys)
    assert all(torch.equal(model.state_dict()[n],v) for n,v in keep.items())
    count=sum(p.numel() for p in model.parameters())
    protocol=read(HERE/'training_protocol.json')
    assert count==protocol['expected_parameters']
    assert model.new_stem_parameter_count-model.old_stem_parameter_count==protocol['expected_extra_parameters']
    return dict(common_init_sha256=COMMON_INIT_SHA,exact_shared_tensors=len(keep),state_tensors=len(current),parameters=count,
        extra_parameters=count-10231997,old_stem_parameters=model.old_stem_parameter_count,
        new_stem_parameters=model.new_stem_parameter_count,only_missing_prefix='backbone.stem.',
        new_stem_initialization='Original module default initialization, seed-specific forked RNG')
'''+s[b:]
s=change(s,"    cfg.update(resume=None", "    cfg['model']='OriginalStemDEIM'\n    cfg['OriginalStemDEIM']=cfg.pop('ResidualGeometryDEIM')\n    cfg.update(resume=None")
s=s.replace('support_matching12_training_v1','original_stem12_training_v1').replace("'support_cost'","'original_stem'")
s=change(s,"    assert plan['matching_coefficient']==ident['matching_coefficient']==protocol['matching_coefficient']==2.0\n",'')
a=s.index("    assert sha(HERE/'support_matching.py')");b=s.index("    assert sha(bundle/'resolved.yml')",a)
s=s[:a]+"    assert sha(HERE/'original_stem.py')==protocol['original_stem_source_sha256']\n"+s[b:]
s=s.replace('added_parameters=0)','added_parameters=sum(p.numel() for p in model.parameters())-10231997)')
save('runtime.py',s)

s=(BASE/'prepare_training.py').read_text().replace('support_matching12_20261011_v1','original_stem12_20261011_v1').replace('support_cost','original_stem')
s=s.replace('seed=seed,matching_coefficient=2.0','seed=seed').replace('seed=seed,matching_coefficient=2.0,','seed=seed,')
s=s.replace('extra_parameters=0,','extra_parameters=protocol[\'expected_extra_parameters\'],')
s=s.replace('added_parameter_count=0,','added_parameter_count=protocol[\'expected_extra_parameters\'],')
s=s.replace('Only ordinary matching costs change; prediction grid, native augmentation/loss weights/schedule remain; no new baseline forward.','Only original backbone.stem changes; native matching/augmentation/loss weights/schedule remain; no new baseline forward.')
save('prepare_training.py',s)

s=(BASE/'experiment.py').read_text().replace('SupportMatchingSolver','OriginalStemSolver').replace('support_matching12_identity','original_stem12_identity')
a=s.index('            from support_matching import');b=s.index('            assert self.output_dir.resolve()',a)
s=s[:a]+s[b:]
s=change(s,"            assert self.initialization_receipt['extra_parameters'] == 0", "            assert self.initialization_receipt['extra_parameters'] == 5325")
s=change(s,'matching_cost=self.support_binding.metadata(), evaluator_runtime=evaluator_runtime,','stem_source=read(paths[\'experiment_root\']/\'code/original_stem_source.json\'), evaluator_runtime=evaluator_runtime,')
s=s.replace('resume=None, automatic_retry=False, added_parameters=0,','resume=None, automatic_retry=False, added_parameters=5325,')
s=s.replace('Ordinary matching cost changes across five stages; native GO union and all DN supervision remain; no efficacy claim.','Only original Stem changes. Native matching and losses remain. Efficacy not yet established.')
s=s.replace('            self.matching_audit.enter_epoch(epoch)\n','')
a=s.index("        if hasattr(solver,'matching_audit'):");b=s.index('        dist_utils.cleanup()',a)
s=s[:a]+s[b:]
save('experiment.py',s)

s=(BASE/'training_audit.py').read_text().replace('isolated FDR candidate','original Stem candidate')
s=s.replace('same historical native model constructor; no added parameters','historical native constructor plus Stem initialized in forked RNG; outside-Stem tensors unchanged')
s=s.replace('no_new_model_parameters=True,','no_new_model_parameters=False, added_parameters=5325,')
s=s.replace('FDR_matching_coefficient=2.0,','original_stem_only=True,')
save('training_audit.py',s)

s=(BASE/'verify_candidate_reload.py').read_text().replace('support_matching12_closeout_v1','original_stem12_closeout_v1').replace("'support_cost'","'original_stem'").replace('support_matching12_identity','original_stem12_identity')
s=s.replace("len(loaded) == 794","len(loaded) == read(run/'runtime_start.json')['initialization']['state_tensors']")
s=s.replace('count == 10231997','count == native_protocol[\'expected_parameters\']')
s=s.replace('strict_loaded_tensors=794,',"strict_loaded_tensors=read(run/'runtime_start.json')['initialization']['state_tensors'],")
save('verify_candidate_reload.py',s)
s=(BASE/'collect_terminal.py').read_text().replace('support_matching12_terminal_v1','original_stem12_terminal_v1').replace('terminal_support_cost_seed','terminal_original_stem_seed').replace('FDR_FIXED12','STEM_FIXED12')
save('collect_terminal.py',s)
save('finish.py',(BASE/'finish.py').read_text().replace('FDR_FIXED12','STEM_FIXED12').replace('FDR_CLOSEOUT','STEM_CLOSEOUT'))

save('preflight.py','''"""Candidate-only shape/backward check; no optimizer, no baseline forward."""
import argparse,json,torch
from pathlib import Path
from runtime import verify_bundle,activate,register_model,load_common,write
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
receipt,plan,paths=verify_bundle(a.bundle)
activate(paths['native_root']);register_model()
from engine.core import YAMLConfig
from engine.misc import dist_utils
dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
cfg=YAMLConfig(str(Path(a.bundle)/'resolved.yml'));model=cfg.model
init=load_common(model,paths['common_init'])
stem=model.backbone.stem.cuda().train()
shapes=[]
for side in (640,832):
    x=torch.linspace(-1,1,side*side,device='cuda').reshape(1,1,side,side).expand(4,3,-1,-1).contiguous()
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        y=stem(x);loss=y.float().square().mean()
    assert tuple(y.shape)==(4,16,side//4,side//4) and torch.isfinite(y).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in stem.parameters() if p.grad is not None)
    shapes.append(list(y.shape));stem.zero_grad(set_to_none=True)
result=dict(status='STEM_SHAPE_BACKWARD_PASS_NO_UPDATES',seed=plan['seed'],initialization=init,shapes=shapes,
    model_forwards=0,stem_forwards=2,optimizer_updates=0,baseline_forwards=0,
    peak_allocated_bytes=torch.cuda.max_memory_allocated())
write(paths['screen_root']/'preflight.json',result);print(json.dumps(result),flush=True)
''')
close=json.loads((BASE/'closeout_protocol.json').read_text());close['schema']='original_stem12_closeout_v1'
close['files']={n:sha(OUT/n) for n in ('verify_candidate_reload.py','runtime.py','original_stem.py','original_stem_model.py','evaluator_binding.py','results.py','stage_trace.py','training_protocol.json')};js('closeout_protocol.json',close)
for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
for n in ('runtime.py','prepare_training.py','experiment.py','verify_candidate_reload.py'):
    assert 'matching_coefficient' not in (OUT/n).read_text(),n
manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
delivery=Path('outputs/original_stem12_20261011_v1');delivery.mkdir(exist_ok=True)
(delivery/'local_code_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
archive=delivery/'original_stem12_code_v1.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print(json.dumps({'code_files':len(manifest),'archive':str(archive),'archive_sha256':sha(archive),'original_stem_sha256':sha(OUT/'original_stem.py'),'protocol_sha256':sha(OUT/'training_protocol.json')}))
