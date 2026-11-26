"""Bind a zero-matching-call infrastructure repair, preserving v1 failures."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile
from matcher_source import instrument_source

root=Path.cwd(); here=Path(__file__).resolve().parent
old=root/'outputs/support_matching_20261011_v1'; out=root/'outputs/support_matching_20261011_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
protocol=read(old/'protocol.json')
native=root/'outputs/scale_protection12_20261010_v1/terminal_collection/seed2_target_protect_v1/extracted/native/engine/deim/matcher.py'
assert sha(native)==protocol['native_matcher_sha256']
source=native.read_text(encoding='utf8'); tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HungarianMatcher')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
start=min([fn.lineno]+[n.lineno for n in fn.decorator_list])-1
fsource=textwrap.dedent(''.join(source.splitlines(keepends=True)[start:fn.end_lineno]))
result=instrument_source(fsource)
for p in here.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
failures={}
for seed in (1,2):
    p=old/f'seed{seed}_terminal_failure.json';f=read(p)
    assert f['completion.json']['exit_code']==1 and not f['worker_start.json_alive'] and not f['controller_start.json_alive']
    assert 'IndentationError' in f['log_tail'] and 'line 76, in main' in f['log_tail']
    failures[str(seed)]=dict(record_sha256=sha(p),last_executed_stage='compile instrumented matcher before loops; zero matcher calls, zero detector forwards, zero training updates')
out.mkdir(exist_ok=False)
protocol.update(schema='support_matching_saved_probe_v2_infrastructure_repair',
    parent_protocol_sha256=sha(old/'protocol.json'),preserved_v1_failures=failures,
    repair='Use actual dedented indentation; assert original AST is unchanged after removing the sole added cost block.',
    candidate_science_rule_and_gates_unchanged=True,
    source_code_sha256={p.name:sha(p) for p in here.glob('*.py')})
(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
(out/'local_instrumentation_preflight.json').write_text(json.dumps(dict(status='PASS_COMPILE_AND_SINGLE_INSERTION_AST',
    native_matcher_sha256=sha(native),compiled_source_sha256=hashlib.sha256(result.encode()).hexdigest(),
    model_forwards=0,matcher_calls=0,epochs=0),indent=2),encoding='utf8')
with zipfile.ZipFile(out/'execution_payload.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    z.write(out/'protocol.json','protocol.json')
    for p in here.glob('*.py'):z.write(p,'code/'+p.name)
print(json.dumps(dict(status='FROZEN_INFRASTRUCTURE_REPAIR',protocol_sha256=sha(out/'protocol.json'),
    zip_sha256=sha(out/'execution_payload.zip'),bytes=(out/'execution_payload.zip').stat().st_size)))
