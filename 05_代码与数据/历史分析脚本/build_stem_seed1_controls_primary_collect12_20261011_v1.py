from pathlib import Path
import difflib,hashlib,json
source=Path('英文小论文/code/stem_new_controls_primary_collect12_20261011_v1/collect_completed.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='c142e69042429352b843ff7e2efaafd97dc45068b430d8d6fc7d9f5c8f8b689b'
old=source.read_text(encoding='utf8');new=old
changes=[('seed2','seed1'),('SEED2','SEED1'),('/24085403014','/25085400120'),("value['seed']==2","value['seed']==1"),("module.collect('/25085400120', 2, arm)","module.collect('/25085400120', 1, arm)")]
for before,after in changes:assert before in new;new=new.replace(before,after)
root=Path('英文小论文/code/stem_seed1_controls_primary_collect12_20261011_v1');root.mkdir(exist_ok=False)
with (root/'collect_completed.py').open('x',encoding='utf8',newline='\n') as f:f.write(new)
with (root/'adaptation.diff').open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(source),tofile=str(root/'collect_completed.py'))))
with (root/'adaptation.json').open('x',encoding='utf8') as f:json.dump(dict(source_sha256=sha(source),new_wrapper_sha256=sha(root/'collect_completed.py'),changes=changes,frozen_collector_and_model_sources_unchanged=True),f,indent=2)
print(json.dumps(dict(status='READ_ONLY_COLLECTOR_SEED1_ADAPTER_PREPARED',sha256=sha(root/'collect_completed.py'))))
