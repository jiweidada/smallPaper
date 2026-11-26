"""Preserve the rejected self-hashing index and write a correct science-file index."""
from pathlib import Path
import hashlib,json
ROOT=Path.cwd();folder=ROOT/'outputs/query_budget_and_P2_gain_scope12_20261011_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((folder/'manifest.json').read_bytes())
assert old['files']['manifest.json']==hashlib.sha256(b'').hexdigest() and sha(folder/'manifest.json')!=old['files']['manifest.json']
files={n:h for n,h in old['files'].items() if n!='manifest.json'}
assert set(files)=={'report.json','image_class_GT_counts.npz'}
for n,h in files.items():assert sha(folder/n)==h,n
new=dict(files=files,supersedes='manifest.json',rejected_v1_manifest_sha256=sha(folder/'manifest.json'),repair_scope='The v1 index opened itself before enumerating files, so included its own empty-file hash. v2 excludes the index. Science report and GT-count arrays unchanged; no model, matching, training or numerical analysis rerun.')
with (folder/'manifest_v2.json').open('x',encoding='utf8') as f:json.dump(new,f,indent=2)
print('PASS_CORRECTED_MANIFEST_SCIENCE_FILES_UNCHANGED_V1_PRESERVED')
