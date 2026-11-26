"""Engineering fixtures for rejecting incomplete, mismatched or resumed primary endpoints."""
from pathlib import Path
import ast, base64, copy, difflib, hashlib, importlib.util, json, sys
ROOT=Path.cwd();HERE=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('user_primary_guards',HERE/'guards.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
fixtures=[]
def reject(name, call):
    try:call()
    except (AssertionError,KeyError):fixtures.append(dict(name=name,rejected=True));return
    raise AssertionError('Unsafe evidence accepted: '+name)
for seed,v in read(ROOT/'outputs/user_primary_collection_identity12_20261011_v1_inputs.json').items():
    raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256']
    capture=json.loads(raw)
    for arm,j in capture['jobs'].items():
        initialization=dict(j['initialization'],user_arm=arm)
        identity=dict(schema=j['identity_schema'],seed=j['identity_seed'],arm=j['identity_arm'],
                      native_parameter_count=10231997,added_parameter_count=j['plan']['extra_parameters'],
                      baseline_training_epochs_added=0,baseline_model_forwards_added=0)
        plan=dict(j['plan'],schema=identity['schema'],seed=int(seed),arm=arm,recipe_epochs=132)
        g.identity_plan(identity,plan,initialization,arm,int(seed))
        fixtures.append(dict(name=f'captured_identity_and_initialization_seed{seed}_{arm}',passed=True,scientific_endpoint=False))
        changes=[('identity','seed',3),('identity','arm','warm_stem'),('identity','schema','wrong'),
                 ('identity','added_parameter_count',-1),('identity','baseline_training_epochs_added',1),
                 ('identity','baseline_model_forwards_added',1),('plan','max_epochs',13),('plan','recipe_epochs',12),
                 ('plan','fixed_endpoint','best_epoch'),('plan','expected_parameters',1),
                 ('initialization','exact_shared_tensors',1),('initialization','strict_load',False),
                 ('initialization','new_state_unchanged',False)]
        for tag,k,value in changes:
            all_values=copy.deepcopy(dict(identity=identity,plan=plan,initialization=initialization));all_values[tag][k]=value
            reject(f'seed{seed}_{arm}_{tag}_{k}',lambda q=all_values:g.identity_plan(q['identity'],q['plan'],q['initialization'],arm,int(seed)))
        trainer=dict(child_pid=j['trainer']['pid'],child_start_ticks=j['trainer']['start_ticks'],controller_pid=j['trainer']['controller_pid'])
        controller=dict(pid=j['trainer']['controller_pid'],start_ticks=j['trainer']['controller_start_ticks'])
        g.owners(trainer,controller,j['trainer'])
        for k in ('pid','start_ticks','controller_pid','controller_start_ticks'):
            wrong=copy.deepcopy(j['trainer']);wrong[k]='mismatch'
            reject(f'seed{seed}_{arm}_owner_{k}',lambda q=wrong:g.owners(trainer,controller,q))
        done=dict(identity=identity,status='NATIVE_FIT_FINISHED_EXACT12',completed_epochs=12,entered_epochs=list(range(12)),automatic_retry=False)
        c=dict(status='COMPLETE_FIXED12');candidate=dict(exit_code=0,timed_out=False,automatic_retry=False)
        g.positive_terminal(identity,done,c,candidate)
        for tag,k,value in [('done','completed_epochs',11),('done','entered_epochs',list(range(13))),
                            ('done','entered_epochs',[0]*12),('done','automatic_retry',True),
                            ('controller','status','RUNNING'),('candidate','exit_code',1),
                            ('candidate','timed_out',True),('candidate','automatic_retry',True)]:
            all_values=copy.deepcopy(dict(done=done,controller=c,candidate=candidate));all_values[tag][k]=value
            reject(f'seed{seed}_{arm}_terminal_{tag}_{k}',lambda q=all_values:g.positive_terminal(identity,q['done'],q['controller'],q['candidate']))
for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
assert 'torch' not in sys.modules
protocol=read(HERE/'protocol.json');protocol['local_verifier_sha256']=sha(HERE/'verify_primary.py')
assert {n:sha(HERE/n) for n in protocol['remote_sources']}==protocol['remote_sources']
(HERE/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
old=ROOT/'英文小论文/code/warm_stem_primary_collection12_20261011_v1'
diff=''
for before,after in [('collect.py','collect.py'),('verify_primary_v2.py','verify_primary.py')]:
    diff+=''.join(difflib.unified_diff((old/before).read_text().splitlines(keepends=True),(HERE/after).read_text().splitlines(keepends=True),fromfile='verified_Warm_'+before,tofile='user_'+after))
(HERE/'adaptation.diff').write_text(diff,encoding='utf8')
files={n:dict(base64=base64.b64encode((HERE/n).read_bytes()).decode(),bytes=(HERE/n).stat().st_size,sha256=sha(HERE/n)) for n in ('collect.py','wait_stopped.py','guards.py','protocol.json')}
(ROOT/'outputs/user_primary_collection12_20261011_v1_deploy_files.json').write_text(json.dumps(files),encoding='utf8')
result=dict(status='PASS_USER_PRIMARY_READ_ONLY_IDENTITY_OWNER_AND_EXACT12_REJECTION_FIXTURES',
    engineering_fixture_only=True,actual_scientific_endpoints_verified=0,
    captured_initialization_positive_cases=sum(x.get('passed',False) for x in fixtures),negative_cases=sum(x.get('rejected',False) for x in fixtures),fixtures=fixtures,
    sources={p.name:sha(p) for p in HERE.iterdir() if p.is_file()},source_sha256=sha(__file__),
    local_reader_predeployment_fix='Use actual user preflight finite_gradients and P2 branch proof; no added_grad_finite field exists in producer.',
    new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_training_epochs=0,new_optimizer_updates=0)
out=ROOT/'outputs/user_primary_collection12_20261011_v1_CPU_review.json'
with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('fixtures','sources')},ensure_ascii=False))
