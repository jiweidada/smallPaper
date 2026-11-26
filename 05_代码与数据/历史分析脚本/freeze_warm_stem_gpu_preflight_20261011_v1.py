from pathlib import Path
import base64,hashlib,json
ROOT=Path.cwd();HERE=ROOT/'英文小论文/code/warm_stem_gpu_preflight_20261011_v1'
PROTO=ROOT/'英文小论文/code/warm_stem_prototype_20261011_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=dict(schema='warm_stem_gpu_stem_preflight_v1',preflight_sha256=sha(HERE/'preflight.py'),
 prototype_files_sha256={f.name:sha(f) for f in PROTO.iterdir() if f.is_file()},
 CPU_pass_status='PASS_CPU_ORIGINAL794_TENSORS_RNG_ZERO_GAIN_OUTPUT_NATIVE_GRADIENTS_AND_ACTIVE_RESIDUAL_GRAPH',
 GPU_uuid={'1':'GPU-d29c55c5-02ab-1e69-c978-0b28f2102003','2':'GPU-b8084aa9-0176-df73-41fb-03307ded1be3'},
 standalone_Stem_only=True,full_detector_forwards=0,baseline_detector_forwards=0,training_epochs=0,
 resource_condition='Both old queued candidate authorizations complete; actual GPU compute apps<=3 and free>=8192MiB before allocation.',
 scope='Engineering only, no training/evaluator/optimizer or existing process signals. Does not substitute for a full detector CUDA training preflight.')
with (HERE/'protocol.json').open('x',encoding='utf8') as f:json.dump(p,f,indent=2)
print(json.dumps({f.name:dict(base64=base64.b64encode(f.read_bytes()).decode(),sha256=sha(f)) for f in HERE.iterdir() if f.is_file()}))

