"""Version the zero-epoch queued reference for the user's four-job authorization."""
from pathlib import Path
import ast, hashlib, json, shutil, zipfile

BASE = Path('英文小论文/code/native_stem_scratch12_20261011_v2')
OUT = Path('英文小论文/code/native_stem_scratch12_20261011_v3')
DELIVERY = Path('outputs/native_stem_scratch12_20261011_v3')
assert not OUT.exists() and not DELIVERY.exists()
OUT.mkdir(); DELIVERY.mkdir()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n, value): (OUT/n).write_bytes(value)
for p in BASE.iterdir():
    if p.is_file(): shutil.copyfile(p, OUT/p.name)
for n in ('prepare_training.py', 'coordinate.py'):
    value = (OUT/n).read_bytes()
    assert b'native_stem_scratch12_20261011_v2' in value
    value = value.replace(b'native_stem_scratch12_20261011_v2', b'native_stem_scratch12_20261011_v3')
    if n == 'coordinate.py':
        assert b"len(state['host_compute_processes'])<3" in value
        value = value.replace(b"len(state['host_compute_processes'])<3", b"len(state['host_compute_processes'])<protocol['concurrency']['max_training_processes_per_gpu']")
        value = value.replace(b'existing three-job GPU schedule', b'authorized four-job GPU schedule')
    save(n, value)
protocol = json.loads((OUT/'training_protocol.json').read_bytes())
assert protocol['concurrency']['max_training_processes_per_gpu'] == 3
protocol['concurrency']['max_training_processes_per_gpu'] = 4
protocol['concurrency']['authorization'] = 'User explicitly requested immediate fourth concurrent trainer per GPU, 2026-10-11; existing training parameters unchanged.'
save('training_protocol.json', (json.dumps(protocol,ensure_ascii=False,indent=2)+'\n').encode())
close = json.loads((OUT/'closeout_protocol.json').read_bytes())
close['files'] = {n:sha(OUT/n) for n in close['files']}
save('closeout_protocol.json',(json.dumps(close,ensure_ascii=False,indent=2)+'\n').encode())
changed = {n for n in (p.name for p in OUT.iterdir()) if sha(OUT/n) != sha(BASE/n)}
assert changed == {'prepare_training.py','coordinate.py','training_protocol.json','closeout_protocol.json'}, changed
for p in OUT.glob('*.py'): ast.parse(p.read_bytes())
manifest = {p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
(DELIVERY/'local_code_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
with zipfile.ZipFile(DELIVERY/'native_stem_scratch12_code_v1.zip','x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,p.name)
(DELIVERY/'scheduling_change.json').write_text(json.dumps(dict(changed_files=sorted(changed),
    new_epochs_before_launch=0,training_implementation_byte_identical=True,
    max_training_processes_per_gpu=4,protocol_sha256=sha(OUT/'training_protocol.json')),indent=2),encoding='utf8')
print(json.dumps(dict(files=len(manifest),changed=sorted(changed),protocol_sha256=sha(OUT/'training_protocol.json'))))
