from pathlib import Path
import ast,base64,hashlib,json
root=Path('英文小论文/code/warm_stem_primary_collection12_20261011_v1')
for n in ('collect.py','wait_stopped.py'):ast.parse((root/n).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=dict(schema='warm_seed1_read_only_primary_waiter12_v1',seeds=[1],max_wait_seconds=7200,
 sources={n:sha(root/n) for n in ('collect.py','wait_stopped.py')},collector_protocol_sha256=sha(root/'protocol.json'),
 warm_files_sha256=json.loads((root/'protocol.json').read_bytes())['warm_files_sha256'],
 trainer={'pid':22346,'start_ticks':'478680526'},training_commands=0,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,
 existing_process_signals=0,automatic_retry=False)
with (root/'waiter_protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,indent=2)
files={n:dict(base64=base64.b64encode((root/n).read_bytes()).decode(),sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in ('collect.py','protocol.json','wait_stopped.py','waiter_protocol.json')}
Path('outputs/warm_primary_seed1_waiter12_20261011_v1_deploy_files.json').write_text(json.dumps(files),encoding='utf8')
print(json.dumps({'status':'WARM_READ_ONLY_WAIT_AND_COLLECTION_SOURCE_BOUND','sha256':{n:i['sha256'] for n,i in files.items()}}))
