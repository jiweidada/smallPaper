"""Freeze an isolated initialization-preserving Stem experiment; no training here."""
from pathlib import Path
import ast, hashlib, json, shutil, zipfile

BASE=Path('英文小论文/code/original_stem12_20261011_v1')
PROTO=Path('英文小论文/code/warm_stem_prototype_20261011_v2')
OUT=Path('英文小论文/code/warm_stem12_20261011_v1')
OUT.mkdir(exist_ok=False)
DELIVERY=Path('outputs/warm_stem12_20261011_v1');DELIVERY.mkdir(exist_ok=False)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,s):(OUT/n).write_text(s,encoding='utf-8',newline='\n')
def change(s,a,b):
    assert a in s,a[:150]
    return s.replace(a,b)
def js(n,v):save(n,json.dumps(v,indent=2,ensure_ascii=False)+'\n')

for n in ('results.py','stage_trace.py','evaluator_binding.py','evaluator_sources.json','source_binding.json','reuse_baseline.py','supervise.py','collect_terminal.py','finish.py','verify_candidate_reload.py','prepare_training.py','experiment.py','training_audit.py','runtime.py','original_stem_source.json','original_backbone_source.txt'):
    shutil.copyfile(BASE/n,OUT/n)
for n in ('geometry.py','residual.py','original_stem.py','warm_stem.py','warm_stem_model.py','initialization.py'):
    shutil.copyfile(PROTO/n,OUT/n)
assert all(sha(OUT/n)==sha(BASE/n) for n in ('geometry.py','residual.py','original_stem.py'))

protocol=json.loads((BASE/'training_protocol.json').read_bytes())
protocol.update(schema='warm_stem12_training_v1',execution_allowlist={'warm_stem':[1,2]},
    expected_parameters=10243812,expected_extra_parameters=11815,
    prototype_files_sha256={p.name:sha(p) for p in PROTO.iterdir() if p.is_file()},
    mechanism='Retain the common-init native Stem and add the complete unmodified user EnhancedStem through16 per-channel tanh gains initialized zero. y=native(x)+tanh(gain)*enhanced(x).',
    training_start='All794 original tensors retained exactly;40 native Stem tensors remapped under native.;754 outside-Stem tensors unchanged. New77 enhanced states and1 gain state explicitly retained during strict complete-state loading.',
    acceptance='Both seeds fixed12 EMA APs strictly greater than cached same-seed BASE12 and AP not lower. No best-epoch substitution, BASE retraining, extension or automatic retry.',
    limitations=['Adds11815 parameters and a retained parallel path; not a pure initialization-only causal comparison with direct Stem replacement.','Zero initial gain blocks enhanced-branch gradients initially. Gain gradients can alter global clipping despite equal preclip native-path gradients.','Residual integration is a testable engineering hypothesis, not a novelty or root-cause claim.'],
    concurrency={'max_training_processes_per_gpu':4,'max_existing_apps_before_full_preflight':2,
                 'minimum_free_before_full_preflight_MiB':10240,
                 'candidate_reserved_budget_MiB':8192,'existing_app_reserved_budget_MiB':6144,
                 'whole_GPU_headroom_MiB':1024,'poll_seconds':15,
                 'resource_rule':'GPU total >= candidate8192 + each existing app6144 +1024 headroom; current free >=10240 before full preflight and >=8192 before training. No old process is signalled.'})
js('training_protocol.json',protocol)
source=json.loads((OUT/'original_stem_source.json').read_bytes())
source.update(scope='Retain native common-init Stem; the exact original user EnhancedStem is an additive learned residual with zero initial gain. Backbone attention/activation outside Stem, encoder, decoder, matching, losses and training recipe unchanged.',
    warm_stem_sha256=sha(OUT/'warm_stem.py'),warm_model_sha256=sha(OUT/'warm_stem_model.py'),loader_sha256=sha(OUT/'initialization.py'))
js('original_stem_source.json',source)

s=(BASE/'runtime.py').read_text();a=s.index('IMPLEMENTATION = ');b=s.index('\n\n\ndef sha',a)
files=('runtime.py','prepare_training.py','experiment.py','supervise.py','reuse_baseline.py','training_audit.py','geometry.py','residual.py','original_stem.py','warm_stem.py','warm_stem_model.py','initialization.py','original_stem_source.json','original_backbone_source.txt','results.py','stage_trace.py','evaluator_binding.py','evaluator_sources.json','source_binding.json','training_protocol.json','verify_candidate_reload.py','collect_terminal.py','finish.py','closeout_protocol.json','preflight.py','dispatch.py')
s=s[:a]+'IMPLEMENTATION = '+repr(files)+s[b:]
s=change(s,'from original_stem_model import OriginalStemDEIM\n    register()(OriginalStemDEIM)','from warm_stem_model import WarmStartEnhancedDEIM\n    register()(WarmStartEnhancedDEIM)')
a=s.index('    import torch\n',s.index('def load_common'));b=s.index('\n\n\ndef resolved_config',a)
s=s[:a]+'''    from initialization import load_warm_common
    assert model.geometry_arm=='baseline'
    proof=load_warm_common(model,path)
    protocol=read(HERE/'training_protocol.json')
    count=sum(p.numel() for p in model.parameters())
    assert count==protocol['expected_parameters']
    assert model.new_stem_parameter_count-model.old_stem_parameter_count==protocol['expected_extra_parameters']
    return dict(proof,state_tensors=len(model.state_dict()),parameters=count,
        extra_parameters=count-10231997,old_stem_parameters=model.old_stem_parameter_count,
        new_stem_parameters=model.new_stem_parameter_count,
        new_stem_initialization='Original EnhancedStem defaults in seed-specific forked RNG;16 gain parameters zero')
'''+s[b:]
s=s.replace('OriginalStemDEIM','WarmStartEnhancedDEIM').replace('original_stem12_training_v1','warm_stem12_training_v1').replace("'original_stem'","'warm_stem'")
s=s.replace('User original Stem-only constructor and historical baseline evidence.','Retained native Stem plus zero-initialized user residual; historical baseline evidence.')
save('runtime.py',s)

s=(BASE/'prepare_training.py').read_text().replace('original_stem12_20261011_v1','warm_stem12_20261011_v1').replace('original_stem','warm_stem')
s=s.replace('Only original backbone.stem changes; native matching/augmentation/loss weights/schedule remain; no new baseline forward.','Native backbone.stem retained with the user enhanced residual; native matching/augmentation/loss weights/schedule remain; no new baseline forward.')
s=s.replace('Prepare a new fixed12 FDR candidate','Prepare a distinct fixed12 initialization-preserving Stem candidate')
save('prepare_training.py',s)

s=(BASE/'experiment.py').read_text().replace('OriginalStemSolver','WarmStemSolver').replace('original_stem12_identity','warm_stem12_identity').replace('5325','11815')
s=s.replace('Only original Stem changes. Native matching and losses remain. Efficacy not yet established.','Native Stem retained with a zero-init enhanced residual. Native matching and losses remain. Efficacy not established.')
save('experiment.py',s)
s=(BASE/'training_audit.py').read_text().replace('5325','11815').replace('original_stem_only=True','warm_stem_only=True').replace('original Stem candidate','warm Stem candidate')
s=s.replace('historical native constructor plus Stem initialized in forked RNG; outside-Stem tensors unchanged','historical native constructor; all794 original tensors retained; enhanced branch initialized in forked RNG; gain initially zero')
s=change(s,"        self.live()\n\n    def close(self):",'''        gains={}
        for name,model in (('training',self.model),('EMA',ema.module)):
            gate=model.backbone.stem.residual_gain.detach().float().cpu()
            assert torch.isfinite(gate).all() and gate.numel()==16
            gains[name]={'raw':gate.flatten().tolist(),'effective_tanh':gate.tanh().flatten().tolist()}
        with (self.output/'residual_gain_epochs.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch_zero_based=epoch,gains=gains,
                interpretation='Learned contribution coefficients; magnitude alone does not establish usefulness.'))+'\\n')
        self.live()

    def close(self):''')
save('training_audit.py',s)
save('results.py',(BASE/'results.py').read_text().replace("ARMS = ('baseline', 'original_stem')","ARMS = ('baseline', 'warm_stem')"))
s=(BASE/'verify_candidate_reload.py').read_text().replace('original_stem12_closeout_v1','warm_stem12_closeout_v1').replace("'original_stem'","'warm_stem'").replace('original_stem12_identity','warm_stem12_identity')
save('verify_candidate_reload.py',s)
s=(BASE/'collect_terminal.py').read_text().replace('original_stem12_terminal_v1','warm_stem12_terminal_v1').replace('terminal_original_stem_seed','terminal_warm_stem_seed').replace('new FDR endpoint','new warm Stem endpoint')
save('collect_terminal.py',s)
# Additional runtime code is authored separately before this builder executes.
for n in ('preflight.py','dispatch.py'):
    shutil.copyfile(Path('outputs/warm_stem12_sources_20261011_v1')/n,OUT/n)
s=(BASE/'supervise.py').read_text()
s=change(s,"    assert uuid == protocol['GPU_uuid'][str(plan['seed'])] and not apps, 'Wrong or busy GPU'",'''    assert uuid == protocol['GPU_uuid'][str(plan['seed'])], 'Wrong GPU'
    from dispatch import authorize_training
    authorize_training(root,plan,protocol,receipt)''')
save('supervise.py',s)
s=(BASE/'finish.py').read_text()
s=change(s,"        assert not (screen/'candidate_reload_v1').exists()",'''        from dispatch import wait_old_closeouts_and_idle
        wait_old_closeouts_and_idle(screen,plan,paths)
        assert not (screen/'candidate_reload_v1').exists()''')
save('finish.py',s)
close=json.loads((BASE/'closeout_protocol.json').read_bytes());close['schema']='warm_stem12_closeout_v1'
close['files']={n:sha(OUT/n) for n in ('verify_candidate_reload.py','runtime.py','initialization.py','original_stem.py','warm_stem.py','warm_stem_model.py','evaluator_binding.py','results.py','stage_trace.py','training_protocol.json','dispatch.py')};js('closeout_protocol.json',close)
for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
assert set(files)=={p.name for p in OUT.iterdir() if p.is_file()}
manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
with (DELIVERY/'local_code_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
archive=DELIVERY/'warm_stem12_code_v1.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print(json.dumps(dict(status='ISOLATED_FULL_TRAINING_AND_CLOSEOUT_CODE_FROZEN_NOT_DEPLOYED',files=len(manifest),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,protocol_sha256=sha(OUT/'training_protocol.json'))))
