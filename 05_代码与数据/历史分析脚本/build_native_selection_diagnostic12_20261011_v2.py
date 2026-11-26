"""Extend diagnostic bindings to the actual RGB control; do not change old training/capture code."""
from pathlib import Path
import ast,base64,hashlib,json,shutil,zipfile
OLD=Path('英文小论文/code/native_selection_diagnostic12_20261011_v1')
NEW=Path('英文小论文/code/native_selection_diagnostic12_20261011_v2');NEW.mkdir(exist_ok=False)
DELIVERY=Path('outputs/native_selection_diagnostic12_20261011_v2');DELIVERY.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n in ('selection_trace.py','selection_stage_trace.py','validate_integration_cpu.py'):shutil.copyfile(OLD/n,NEW/n)
s=(OLD/'replay.py').read_text(encoding='utf-8')
def change(a,b):
    global s
    assert a in s,a[:180];s=s.replace(a,b)
change("'native_selection_diagnostic12_v1'","'native_selection_diagnostic12_v2'")
change("assert plan['arm'] == allowed['arm'] and plan['seed'] in (1,2)","assert plan['arm'] in allowed['arms'] and plan['seed'] in (1,2)")
change("dict(original_stem='original_stem12_identity',native_stem_scratch='native_stem_scratch12_identity')", "dict(original_stem='original_stem12_identity',native_stem_scratch='native_stem_scratch12_identity',highfreq_rgb='original_stem12_identity')")
(NEW/'replay.py').write_text(s,encoding='utf-8',newline='\n')
shutil.copyfile('outputs/native_selection_v2_sources_20261011/queue.py',NEW/'queue.py')
p=json.loads((OLD/'diagnostic_protocol.json').read_bytes());p['schema']='native_selection_diagnostic12_v2'
for v in p['allowed_native_roots'].values():v['arms']=[v.pop('arm')]
p['allowed_native_roots']['stem_component12_20261011_v1']=dict(arms=['highfreq_rgb'],files=json.loads(Path('outputs/stem_component12_20261011_v1/local_code_manifest.json').read_bytes()))
p['files']={f.name:sha(f) for f in NEW.glob('*.py')}
p.update(expected_capture_sources_unchanged={n:sha(OLD/n) for n in ('selection_trace.py','selection_stage_trace.py','validate_integration_cpu.py')},
    actual_diagnostic_jobs_per_seed=['original_stem','native_stem_scratch','highfreq_rgb'],
    queue_timeout_seconds=172800,
    queue_rule='Wait until actual reference/component coordinator handles, warm dispatcher, new user-backbone queue and any of its live children stop. Require positive original native closeout completion for all three diagnostic configurations and actual GPU idle. Parent validates all three JSON/source bindings without importing a native runtime; each replay uses a fresh child. No existing process is signalled. Once per saved checkpoint, no training/retry.',
    independent_diagnostic_only=True,
    BASE_model_forwards_added=0,optimizer_updates=0,epochs_added=0,
    scope='New candidate-only saved-EMA selection diagnostic bindings add the completed RGB-input control for full/ref/RGB comparisons on both seeds. Old diagnostic v1, selection capture implementations, all training code and the0-BASE promise remain unchanged. Each actual replay must reproduce native stats, per-GT flags and548 stage arrays exactly before its selection traces are admissible. Dense anchors are unregressed priors; no dense8400 bbox readout or extra head/topk calls.')
(NEW/'diagnostic_protocol.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
for f in NEW.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
manifest={f.name:sha(f) for f in NEW.iterdir() if f.is_file()}
archive=DELIVERY/'diagnostic_code_v2.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(NEW.iterdir()):
        if f.is_file():z.write(f,f.name)
(DELIVERY/'local_code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(base64=base64.b64encode(archive.read_bytes()).decode(),sha256=sha(archive),bytes=archive.stat().st_size,files=manifest)))
