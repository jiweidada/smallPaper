"""New version binds repaired instrumentation; retain strict six-record analysis requirement."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent.parent
old=ROOT/'英文小论文/code/native_selection_analysis12_20261011_v2'
new=ROOT/'英文小论文/code/native_selection_analysis12_20261011_v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=json.loads((old/'protocol.json').read_bytes());assert {n:sha(old/n) for n in prior['files']}==prior['files']
new.mkdir(exist_ok=False)
for p in old.glob('*.py'):
    s=p.read_text(encoding='utf8').replace('actual_selection_offline_analysis_v2','actual_selection_offline_analysis_v3').replace('native_selection_diagnostic12_20261011_v2','native_selection_diagnostic12_20261011_v3').replace('selection_v2_','selection_v3_')
    if p.name=='register_protocol.py':s=s.replace("('queue.py', 'replay.py', 'selection_trace.py'", "('diagnostic_queue.py', 'import_smoke.py', 'replay.py', 'selection_trace.py'")
    if p.name=='selection_core.py':(new/p.name).write_bytes(p.read_bytes())
    else:(new/p.name).write_text(s,encoding='utf8',newline='\n')
    compile(s,str(new/p.name),'exec')
assert sha(new/'selection_core.py')==sha(old/'selection_core.py')
assert {n:sha(old/n) for n in prior['files']}==prior['files']
print('V3 consumer source prepared; six records still required; observation math byte-identical')
