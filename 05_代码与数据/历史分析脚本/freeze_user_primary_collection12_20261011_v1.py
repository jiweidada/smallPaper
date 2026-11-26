"""Final predeployment source seals after review; preserve prior engineering fixtures."""
from pathlib import Path
import ast,base64,hashlib,json
root=Path('英文小论文/code/user_backbone_primary_collection12_20261011_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
protocol=json.loads((root/'protocol.json').read_bytes())
protocol['collector_sha256']=sha(root/'collect.py')
protocol['remote_sources']={n:sha(root/n) for n in ('collect.py','wait_stopped.py','guards.py')}
protocol['local_verifier_sha256']=sha(root/'verify_primary.py')
(root/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
review=json.loads(Path('outputs/user_primary_collection12_20261011_v1_CPU_review.json').read_bytes())
assert review['negative_cases']==150 and review['captured_initialization_positive_cases']==6
assert review['sources']['guards.py']==sha(root/'guards.py') and review['sources']['verify_primary.py']==sha(root/'verify_primary.py')
files={n:dict(base64=base64.b64encode((root/n).read_bytes()).decode(),bytes=(root/n).stat().st_size,sha256=sha(root/n)) for n in ('collect.py','wait_stopped.py','guards.py','protocol.json')}
Path('outputs/user_primary_collection12_20261011_v1_deploy_files.json').write_text(json.dumps(files),encoding='utf8')
proof=dict(status='FINAL_USER_READ_ONLY_COLLECTION_SOURCES_SEALED',files={p.name:sha(p) for p in root.iterdir() if p.is_file()},
    CPU_review_sha256=sha('outputs/user_primary_collection12_20261011_v1_CPU_review.json'),
    change_after_fixture_review='Collector rechecks every remote helper hash; waiter checks current receipt implementation against the frozen28 sources. Identity/terminal/owner guard unchanged.',
    new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_training_epochs=0,scientific_endpoints_verified=0)
with Path('outputs/user_primary_collection12_20261011_v1_source_seal.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2)
print(json.dumps(dict(status=proof['status'],remote_files=proof['files'])))
