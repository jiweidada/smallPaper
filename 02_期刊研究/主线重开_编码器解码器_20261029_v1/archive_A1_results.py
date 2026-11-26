"""Package terminal A1 evidence,excluding linked source/assets; no experiment execution."""
from pathlib import Path
import hashlib
import json
import tarfile

BASE = Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1')
ROOT = BASE/'readout_probe_v1'
assert json.loads((ROOT/'analysis/completed.json').read_text())['status'] == 'ANALYSIS_COMPLETE'
assert json.loads((BASE/'receipts/cumulative_forward_budget.json').read_text())['attempted_full_model_forwards'] == 185
paths = [p for p in ROOT.iterdir() if p.is_file() and not p.is_symlink()]
for directory in ('features','analysis','receipts'):
    paths.extend(p for p in (ROOT/directory).iterdir() if p.is_file() and not p.is_symlink())
paths = sorted(p for p in paths if p.name != 'file_manifest.json')
manifest = dict(scope='A1 codes,frozen contracts,features,fitted readouts,receipts and analysis; excludes linked source/assets and Python caches',
                files=[dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
manifest_path = ROOT/'file_manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
archive = BASE/'A1_readout_probe_bounded.tar.gz'
assert not archive.exists(),'Do not silently replace an existing evidence archive'
with tarfile.open(archive,'w:gz') as target:
    for path in paths+[manifest_path]:
        target.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
receipt = dict(status='TERMINAL_A1_ARCHIVED',archive=str(archive),bytes=archive.stat().st_size,
               sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),individually_hashed_files=len(paths),
               full_model_forwards=185,main_model_training_updates=0,cpu_linear_readouts_fitted=32)
(BASE/'A1_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
