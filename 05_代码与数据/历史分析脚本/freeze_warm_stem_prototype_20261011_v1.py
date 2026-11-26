from pathlib import Path
import base64,hashlib,io,json,zipfile
ROOT=Path.cwd();HERE=ROOT/'英文小论文/code/warm_stem_prototype_20261011_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
files={p.name:sha(p) for p in HERE.iterdir() if p.is_file()}
assert len(files)==8 and 'protocol.json' not in files
for n in ('original_stem.py','geometry.py','residual.py'):
    assert sha(HERE/n)==sha(ROOT/'英文小论文/code/original_stem12_20261011_v1'/n)
assert sha(HERE/'native_stem_scratch.py')==sha(ROOT/'英文小论文/code/native_stem_scratch12_20261011_v3/native_stem_scratch.py')
evidence=ROOT/'outputs/stem_primary_joint_analysis12_20261011_v1/report.json'
assert read(evidence)['status']=='ANALYZED_BOTH_PRIMARY_FIXED12_JOINT_PARTITION_REPLAY_PENDING'
protocol=dict(schema='warm_stem_engineering_prototype_v1',files_sha256=files,
 hypothesis='Replacing the native pretrained Stem changes both structure and initialization. Retaining its exact common-init path and allowing the user enhanced Stem to contribute gradually may retain its structural benefit without the initial replacement.',
 mechanism='y = native_pretrained_stem(x) + tanh(channel_gain) * user_enhanced_stem(x), channel_gain initially zero; both original paths remain trainable.',
 native_common_tensor_count=794,outside_Stem_shared_tensor_count=754,retained_native_Stem_tensors=40,
 native_parameter_count=10231997,prototype_parameter_count=10243812,added_parameters=11815,
 fixed12_acceptance='Both seeds APs strictly above the existing same-seed BASE12 and AP not below it; independently replay the saved12 EMA. No extension, retry or bestepoch substitution.',
 training_implementation_ready=False,training_launched=False,full_detector_GPU_preflight_passed=False,
 evidence={str(evidence.relative_to(ROOT)).replace('\\','/'):sha(evidence),
  'outputs/stem_component12_20261011_v1/initialization_provenance_v1.json':sha(ROOT/'outputs/stem_component12_20261011_v1/initialization_provenance_v1.json')},
 scope='CPU engineering preparation only. No existing frozen configuration, checkpoint, dataset, optimizer, training job or acceptance gate is changed. This is an initialization-conditioned candidate, not a novel attention claim, proof of a unique root or proven AP improvement.',
 limitations=['The existing full/ref comparisons are not yet both fixed12. Both full native12 endpoints are below the pretrained BASE12; formal saved-EMA replay pending.',
  'Initial zero gain blocks enhanced branch gradients on the first backward pass. Gate gradients permit later contribution; speed within12 epochs must be tested.',
  'Extra gate gradients can change global clipping even when native pre-clipping gradients agree.',
  'Capacity and additive integration differ from direct Stem replacement; this cannot isolate initialization as the sole cause.',
  'CPU Stem parity does not verify full detector predictions, CUDA AMP behavior or detection metrics.'])
with (HERE/'protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,indent=2)
b=io.BytesIO()
with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(HERE.iterdir()):z.writestr(p.name,p.read_bytes())
raw=b.getvalue()
print(json.dumps(dict(archive_base64=base64.b64encode(raw).decode(),archive_sha256=hashlib.sha256(raw).hexdigest(),
 files=files|{'protocol.json':sha(HERE/'protocol.json')},archive_bytes=len(raw))))

