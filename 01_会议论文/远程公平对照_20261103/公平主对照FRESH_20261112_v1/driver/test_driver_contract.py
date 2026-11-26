"""Focused closed-manifest and scientific-step checks, without model imports."""
from pathlib import Path
import ast
import hashlib
import json
import types

HERE=Path(__file__).resolve().parent
SCRATCH=HERE/'contract_test_inputs'

def main():
    SCRATCH.mkdir(exist_ok=True)
    plan={'arm':'FULL_FRESH','configuration_sha256':'a'*64,'probe_files':{'initial.pt':'b'*64,'preflight.json':'c'*64}}
    (SCRATCH/'execution_protocol.json').write_text(json.dumps(plan),encoding='utf-8')
    source=(HERE/'backup_protocol.py').read_text(encoding='utf-8')
    ns={'__file__':str(SCRATCH/'backup_protocol.py'),'__name__':'test_protocol'}
    exec(compile(source,str(HERE/'backup_protocol.py'),'exec'),ns)
    def manifest_for(epoch,alias):
        ref={'epoch':min(epoch,295),'checkpoint_sha256':'d'*64,'metrics_sha256':'e'*64}
        m={**ns['identity']('FULL_FRESH',epoch),'nonce':'f'*32,'next_epoch':epoch+1,
           'optimizer_step_attempts':(epoch+1)*1617,'rolling_sha256':'1'*64,'best':ref,'stage_one':ref,
           'evaluated_state_is_next_state':alias,'evaluated_sha256':('1' if alias else '2')*64,
           'evaluated_ema_tensor_sha256':'3'*64,'next_ema_tensor_sha256':('3' if alias else '4')*64}
        sources=ns['expected_sources'](m)
        hashes={'next_epoch.pt':'1'*64,'evaluated_epoch.pt':m['evaluated_sha256'],
                f"selected/epoch{ref['epoch']:04d}/best.pt":ref['checkpoint_sha256'],
                f"selected/epoch{ref['epoch']:04d}/best.metrics.json":ref['metrics_sha256']}
        m['files']=[{'name':n,'source':s,'bytes':20,'sha256':hashes.get(n,'5'*64)} for n,s in sources.items()]
        return m
    cases=[]
    for epoch,alias in [(0,True),(295,True),(296,False),(319,False),(319,True)]:
        m=manifest_for(epoch,alias)
        actual=ns['parse_manifest'](ns['encode'](m),'FULL_FRESH',epoch)
        src=ns['expected_sources'](actual)
        assert src['evaluated_epoch.pt']==('run/next_epoch.pt' if alias else 'run/evaluated_epoch.pt')
        cases.append({'epoch':epoch,'alias':alias,'closed_manifest_passed':True})
    for mutation in ['different_alias_sha','different_alias_ema','missing_evaluated_file','unknown_file','nonbool_role']:
        m=manifest_for(0,True)
        if mutation=='different_alias_sha':m['evaluated_sha256']='2'*64
        elif mutation=='different_alias_ema':m['next_ema_tensor_sha256']='4'*64
        elif mutation=='missing_evaluated_file':m['files']=[f for f in m['files'] if f['name']!='evaluated_epoch.pt']
        elif mutation=='unknown_file':m['files'].append({'name':'unrelated.pt','source':'run/unrelated.pt','bytes':20,'sha256':'9'*64})
        elif mutation=='nonbool_role':m['evaluated_state_is_next_state']=1
        try:ns['parse_manifest'](ns['encode'](m),'FULL_FRESH',0)
        except ns['BackupError']:pass
        else:raise AssertionError(f'Malformed manifest accepted: {mutation}')
    trees={p.name:ast.parse(p.read_text(encoding='utf-8')) for p in HERE.glob('*.py')}
    pre=trees['preflight.py']
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                   (n.func.attr=='evaluate' or (isinstance(n.func.value,ast.Name) and n.func.value.id in ['optimizer','ema'] and n.func.attr in ['step','update'])) for n in ast.walk(pre))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model' for n in ast.walk(pre))==1
    validator=trees['checkpoint_validator.py']
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['step','backward','cuda'] for n in ast.walk(validator))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model' for n in ast.walk(validator))
    audit=json.loads((HERE/'driver_static_semantic_audit.json').read_text(encoding='utf-8'))
    assert audit['training_batch_loop_AST_identical'] and audit['native_load_and_restore_AST_identical']
    assert audit['data_prepare_evaluation_controls_seeds_worker_init_AST_identical']
    report={'status':'FOCUSED_STATIC_CONTRACT_CHECKS_PASSED','closed_manifest_cases':cases,
            'malformed_manifest_cases_rejected':5,'scientific_training_step_AST_unchanged':True,
            'one_preflight_forward_zero_optimizer_or_evaluator_calls':True,'validator_saved_payload_CPU_only':True,
            'model_imports':0,'model_forwards':0,'training_runs':0,'remote_connections':0,
            'scope':'Static function equality plus synthetic closed-backup-manifest contract. Does not validate actual GPU execution or real saved tensors.'}
    (HERE/'driver_contract_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'malformed_rejected':5,'model_imports':0}))

if __name__=='__main__':main()
