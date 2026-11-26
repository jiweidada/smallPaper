from pathlib import Path
import ast,base64,hashlib,json
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'英文小论文/code/stem_component_queue_pulse_20261011_v1';OUT=ROOT/'outputs/stem_component_queue_pulse_20261011_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
for p in CODE.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
old=read(ROOT/'outputs/stem_component12_20261011_v1/local_code_manifest.json')
protocol=dict(schema='owned_component_queue_pulse_v1',queued_arms=['attention_spatial_mean','early_final_downsample'],code_sha256={p.name:sha(p) for p in CODE.glob('*.py')},
    component_code_sha256=old,conductor={'1':dict(pid=13381,start_ticks='476867385'),'2':dict(pid=15065,start_ticks='254060477')},
    reference_controller={'1':[15449,'477378529'],'2':[17181,'254571704']},
    GPU_uuid={'1':'GPU-d29c55c5-02ab-1e69-c978-0b28f2102003','2':'GPU-b8084aa9-0176-df73-41fb-03307ded1be3'},
    helper_timeout_seconds=64800,resource_condition='Actual GPU compute apps<=2 and free>=6144MiB; native unchanged authorizer still enforces max3 including the reference.',
    training_parameters_changed=False,trainers_signalled=False,maximum_epochs_per_configuration=12,new_training_retries=0,
    scope='Resume only the original owned paused CPU conductor until one authorized pending arm is dispatched, then hold it again. Final handoff remains with the existing reference coordinator.')
with (CODE/'protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,indent=2)
OUT.mkdir(exist_ok=False)
payload={p.name:dict(sha256=sha(p),b64=base64.b64encode(p.read_bytes()).decode()) for p in CODE.iterdir() if p.is_file()}
with (OUT/'local_code_manifest.json').open('x') as f:json.dump({n:d['sha256'] for n,d in payload.items()},f,indent=2)
print(json.dumps(payload))
