"""Byte integrity / exclusive files. Hashing is not outcome interpretation."""
import hashlib,json,pathlib
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 with pathlib.Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify_files(base,manifest):
 base=pathlib.Path(base).resolve()
 for row in manifest['files']:
  rel=pathlib.PurePosixPath(row['relative'])
  assert not rel.is_absolute() and '..' not in rel.parts
  p=(base/pathlib.Path(*rel.parts)).resolve();assert base in p.parents
  assert p.is_file() and sha(p)==row['sha256'],str(p)
  if 'bytes' in row:assert p.stat().st_size==row['bytes']
def verify_partition(folder,expected_seal=None):
 folder=pathlib.Path(folder)
 if expected_seal:assert sha(folder/'seal.json')==expected_seal,'Partition seal changed'
 manifest=json.loads((folder/'seal.json').read_text());verify_files(folder,manifest)
 assert len(manifest['files'])==manifest['batches']
 assert {p.name for p in folder.iterdir() if p.is_file()}=={'seal.json'}|{r['relative'] for r in manifest['files']}
 return manifest
def heldout_guard(root,seed):
 root=pathlib.Path(root);release=json.loads((root/'analysis_release_v3.json').read_text())
 freeze=root/'analysis_freeze_v3.json';assert sha(freeze)==release['freeze_sha256']
 value=json.loads(freeze.read_text());assert value['both_development_runs_passed']
 verify_files(root,value)
 expected=value['generation_seal_sha256'][str(seed)]['heldout']
 return verify_partition(root/f'isolated_results_seed{seed}/heldout',expected)
