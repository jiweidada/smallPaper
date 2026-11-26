"""Retain rejection tests and check the corrected native preflight receipt type."""
from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path.cwd();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=ROOT/'outputs/validate_user_primary_collection12_20261011_v1.py'
text=old.read_text(encoding='utf8').split("for p in HERE.glob('*.py'):")[0].replace("HERE=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v1'","HERE=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2'")
scope={'__name__':'v2_fixtures'};exec(compile(text,str(old),'exec'),scope)
g,fixtures=scope['g'],scope['fixtures'];reject=scope['reject']
for seed,prefix in ((1,'/25085400120'),(2,'/24085403014')):
    py=prefix+('/envs/deim/bin/python' if seed==1 else '/deim_migration_runtime/envs/deim/bin/python')
    bundle=prefix+'/DEIM/experiments/user_backbone12_20261011_v1/seed'+str(seed)+'_p2_skip/bundle'
    launch=dict(pid=123,start_ticks='456',command=[py,'-B',prefix+'/DEIM/experiments/user_backbone12_20261011_v1/code/preflight.py','--bundle',bundle])
    completion=dict(exit_code=0,training_epochs=0,optimizer_updates=0)
    g.preflight_launch(launch,completion,prefix,seed,bundle)
    fixtures.append(dict(name=f'positive_actual_dispatch_schema_seed{seed}',passed=True,scientific_endpoint=False))
    for k,v in (('pid',-1),('start_ticks','wrong'),('command',launch['command']+['--retry'])):
        bad=dict(launch);bad[k]=v;reject(f'bad_dispatch_{seed}_{k}',lambda d=bad:g.preflight_launch(d,completion,prefix,seed,bundle))
    for k in completion:
        bad=dict(completion);bad[k]=1;reject(f'bad_dispatch_completion_{seed}_{k}',lambda d=bad:g.preflight_launch(launch,d,prefix,seed,bundle))
new=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2'
protocol=json.loads((new/'protocol.json').read_bytes())
assert {n:sha(new/n) for n in protocol['remote_sources']}==protocol['remote_sources']
assert sha(new/'collect.py')==protocol['collector_sha256'] and sha(new/'verify_primary.py')==protocol['local_verifier_sha256']
assert "detector_preflight_start.json" not in (new/'collect.py').read_text(encoding='utf8')
assert "detector_preflight_dispatch.json" in (new/'collect.py').read_text(encoding='utf8')
for p in new.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
assert 'torch' not in sys.modules
report=dict(status='PASS_PATH_REPAIR_WITH_TERMINAL_AND_OWNERSHIP_GATES_RETAINED',positive_cases=sum(x.get('passed',False) for x in fixtures),negative_cases=sum(x.get('rejected',False) for x in fixtures),fixtures=fixtures,actual_scientific_endpoints=0,training_source_changes=0)
with (ROOT/'outputs/user_primary_collection12_20261011_v2_CPU_review.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!='fixtures'}))
