"""Freeze startup-handshake repair before any native scheduler helper is deployed."""
from pathlib import Path
import ast,base64,hashlib,json
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'英文小论文/code/stem_component_queue_pulse_20261011_v1';NEW=ROOT/'英文小论文/code/stem_component_queue_pulse_20261011_v2';OUT=ROOT/'outputs/stem_component_queue_pulse_20261011_v2'
NEW.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ('queue_pulse.py','validate_cpu.py'):
    s=(OLD/name).read_text(encoding='utf8')
    if name=='queue_pulse.py':s=s.replace('stem_component_queue_pulse_20261011_v1','stem_component_queue_pulse_20261011_v2')
    else:
        a="p=Path(sys.argv[1]);(p/'ready').write_text('ready')\nfor i in (1,2):"
        b="p=Path(sys.argv[1]);(p/'ready').write_text('ready')\nwhile not (p/'go').exists():time.sleep(.01)\nfor i in (1,2):"
        assert s.count(a)==1;s=s.replace(a,b)
        a="assert not (root/'dispatch1.json').exists()\n            paths="
        b="assert not (root/'dispatch1.json').exists()\n            (root/'go').write_text('go')\n            paths="
        assert s.count(a)==1;s=s.replace(a,b)
    ast.parse(s)
    with (NEW/name).open('x',encoding='utf8',newline='\n') as f:f.write(s)
p=json.loads((OLD/'protocol.json').read_bytes());p['schema']='owned_component_queue_pulse_v2';p['code_sha256']={n:sha(NEW/n) for n in ('queue_pulse.py','validate_cpu.py')}
p['pre_deployment_repair']='Disposable CPU fixture waits for a parent go file written while held; eliminate fixture startup race. No scheduler helper or trainer had been launched with v1.'
with (NEW/'protocol.json').open('x') as f:json.dump(p,f,indent=2)
payload={p.name:dict(sha256=sha(p),b64=base64.b64encode(p.read_bytes()).decode()) for p in NEW.iterdir() if p.is_file()}
with (OUT/'local_code_manifest.json').open('x') as f:json.dump({n:d['sha256'] for n,d in payload.items()},f,indent=2)
print(json.dumps(payload))
