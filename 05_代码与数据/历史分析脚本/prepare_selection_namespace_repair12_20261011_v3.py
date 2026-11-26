"""Versioned instrumentation repair; preserve frozen v2 and native training sources."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parent.parent
old=ROOT/'英文小论文/code/native_selection_diagnostic12_20261011_v2'
new=ROOT/'英文小论文/code/native_selection_diagnostic12_20261011_v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=json.loads((old/'diagnostic_protocol.json').read_bytes())
assert {n:sha(old/n) for n in spec['files']}==spec['files']
new.mkdir(exist_ok=False)
for n in ('selection_trace.py','selection_stage_trace.py','validate_integration_cpu.py'):
    (new/n).write_bytes((old/n).read_bytes())
s=(old/'replay.py').read_text(encoding='utf8').replace('native_selection_diagnostic12_v2','native_selection_diagnostic12_v3')
(new/'replay.py').write_text(s,encoding='utf8',newline='\n')
assert ast.dump(ast.parse(s.replace('native_selection_diagnostic12_v3','native_selection_diagnostic12_v2')))==ast.dump(ast.parse((old/'replay.py').read_text(encoding='utf8')))
q=(old/'queue.py').read_text(encoding='utf8')
q=q.replace("    return result\n", "    # Preserve and wait for the old observer, including its actual child.\n    old=base/'native_selection_diagnostic12_20261011_v2'/f'seed{seed}_queue'\n    if (old/'start.json').exists():result.append(read(old/'start.json'))\n    for p in old.glob('*_dispatch.json'):result.append(read(p))\n    for bundle,arm in paths(base.parent.parent,seed):\n        worker=bundle.parent/'native_selection_diagnostic12_v2/start.json'\n        if worker.exists():\n            r=read(worker);result.extend([dict(pid=r['pid'],start_ticks=r['start_ticks']),dict(pid=r['controller_pid'],start_ticks=r['controller_start_ticks'])])\n    return result\n")
# base is prefix/DEIM/experiments; base.parent.parent is prefix.
q=q.replace("        jobs=verify_candidates(prefix,seed,protocol)","        verify_cpu_and_preserved_failure(b,seed,protocol,root)\n        jobs=verify_candidates(prefix,seed,protocol)")
helper='''
def verify_cpu_and_preserved_failure(base,seed,protocol,root):
    cpu=HERE.parent/f'seed{seed}_CPU'
    smoke=read(cpu/'import_verification.json');fixture=read(cpu/'fixture/verification.json');exit=read(cpu/'completion.json')
    assert exit['exit_code']==0 and smoke['status']=='PASS_REAL_ENTRYPOINT_STDLIB_QUEUE_AND_TORCH_IMPORT'
    assert smoke['source_sha256']==protocol['files'] and smoke['protocol_sha256']==sha(HERE/'diagnostic_protocol.json')
    assert smoke['CUDA_initialized'] is False and smoke['model_forwards']==smoke['training_epochs']==0
    assert fixture['status']=='PASS_NATIVE_PRODUCER_AND_STAGE_JOIN_CPU_FIXTURES' and fixture['CUDA_initialized'] is False
    assert fixture['diagnostic_code_sha256']==protocol['files'] and len(fixture['cases'])==2
    assert all(c['exact_fixture_outputs'] and c['additional_head_calls']==0 and c['actual_topk_calls']==1 and c['RNG_unchanged'] for c in fixture['cases'])
    assert exit['import_sha256']==sha(cpu/'import_verification.json') and exit['fixture_sha256']==sha(cpu/'fixture/verification.json')
    old=base/'native_selection_diagnostic12_20261011_v2'
    assert {n:sha(old/'code'/n) for n in protocol['preserved_v2_sources']}==protocol['preserved_v2_sources']
    failed=read(old/f'seed{seed}_queue/failure.json');assert failed['status']=='FAILED_NO_AUTOMATIC_RETRY'
    bundle,arm=paths(base.parent.parent,seed)[0]
    out=bundle.parent/'native_selection_diagnostic12_v2';log=(out/'worker.log').read_text()
    assert "ImportError: cannot import name 'Queue' from 'queue'" in log and 'selection_stage_trace.py' in log
    assert 'queue.py' in log and read(out/'completion.json')['exit_code']==1
    assert not any((out/n).exists() for n in ('verification.json','endpoint_epoch12','stages_epoch12','selection_epoch12'))
    write(root/'repair_precondition.json',dict(status='CPU_IMPORT_AND_NATIVE_FIXTURES_PASSED_OLD_PREMODEL_IMPORT_FAILURE_PRESERVED',
        import_report_sha256=sha(cpu/'import_verification.json'),fixture_sha256=sha(cpu/'fixture/verification.json'),
        old_queue_failure_sha256=sha(old/f'seed{seed}_queue/failure.json'),old_worker_log_sha256=sha(out/'worker.log'),
        old_worker_exit_sha256=sha(out/'completion.json'),new_training_epochs=0,new_BASE_forwards=0,
        old_instrumentation_failure_before_model_creation=True,failed_training_retried=False))

'''
q=q.replace('\ndef main(prefix,seed):', '\n'+helper+'\ndef main(prefix,seed):')
(new/'diagnostic_queue.py').write_text(q,encoding='utf8',newline='\n')
smoke='''"""Run directly from the deployed script directory; no model or CUDA calls."""
from pathlib import Path
import argparse,hashlib,importlib,json,sys,sysconfig
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out):
    assert Path(sys.path[0]).resolve()==HERE and not (HERE/'queue.py').exists()
    import torch
    import queue
    assert Path(queue.__file__).resolve()==(Path(sysconfig.get_path('stdlib'))/'queue.py').resolve()
    q=queue.Queue();q.put('native import witness');assert q.get_nowait()=='native import witness'
    import replay,diagnostic_queue
    assert Path(replay.__file__).resolve()==HERE/'replay.py' and Path(diagnostic_queue.__file__).resolve()==HERE/'diagnostic_queue.py'
    spec=json.loads((HERE/'diagnostic_protocol.json').read_bytes());sources={p.name:sha(p) for p in HERE.glob('*.py')}
    assert sources==spec['files'] and not torch.cuda.is_initialized()
    result=dict(status='PASS_REAL_ENTRYPOINT_STDLIB_QUEUE_AND_TORCH_IMPORT',source_sha256=sources,
        protocol_sha256=sha(HERE/'diagnostic_protocol.json'),entrypoint_directory=str(HERE),stdlib_queue_path=queue.__file__,
        torch_version=str(torch.__version__),CUDA_initialized=False,model_forwards=0,training_epochs=0,optimizer_updates=0)
    with Path(out).open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],torch=result['torch_version'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
'''
(new/'import_smoke.py').write_text(smoke,encoding='utf8',newline='\n')
spec['schema']='native_selection_diagnostic12_v3'
spec['files']={p.name:sha(p) for p in sorted(new.glob('*.py'))}
spec['preserved_v2_sources']={p.name:sha(p) for p in old.iterdir() if p.is_file()}
spec['repair']='Rename queue.py to diagnostic_queue.py; real entrypoint Torch/stdlib Queue import; existing observational hooks and native replay math unchanged. No training retry.'
(new/'diagnostic_protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8',newline='\n')
for p in new.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
assert {n:sha(old/n) for n in spec['preserved_v2_sources']}==spec['preserved_v2_sources']
report=dict(status='SOURCE_REVIEW_PASSED_REMOTE_CPU_CHECKS_REQUIRED_BEFORE_GPU',
    sources={p.name:sha(p) for p in new.iterdir() if p.is_file()},preserved_v2_sources=spec['preserved_v2_sources'],
    replay_AST_identical_after_version_path_normalization=True,observational_hook_sources_byte_identical=True,
    existing_training_sources_modified=False,new_training_epochs=0,new_BASE_forwards=0,
    remote_CPU_passed=False,remote_GPU_started=False)
out=ROOT/'outputs/native_selection_diagnostic12_20261011_v3';out.mkdir(exist_ok=False)
(out/'source_review.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
