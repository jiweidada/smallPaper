"""Archive terminal diagnostic artifacts; no model execution."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile

r=Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1/window_update_probe_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((r/'summary.json').read_text())
c=json.loads((r/'completed.json').read_text())
assert c['summary_sha256']==sha(r/'summary.json') and s['status']=='COMPLETED'
assert s['global_full_detector_forwards']==200 and s['new_full_detector_forwards']==15
assert s['state_digest_before']==s['state_digest_after']
frozen=json.loads((r/'freeze_manifest.json').read_text())
for x in frozen['files']:assert sha(r/x['file'])==x['sha256']
b=r/'backup';b.mkdir(exist_ok=True)
files=[p for p in r.rglob('*') if p.is_file() and b not in p.parents and '__pycache__' not in p.parts]
rows=[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
(b/'file_manifest.json').write_text(json.dumps({'files':rows},indent=2)+'\n')
archive=b/'A8_window_memory_results.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for row in rows:t.add(r/row['path'],arcname=row['path'])
 t.add(b/'file_manifest.json',arcname='backup/file_manifest.json')
with tarfile.open(archive,'r:gz') as t:
 for row in rows:assert hashlib.sha256(t.extractfile(row['path']).read()).hexdigest()==row['sha256']
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'TERMINAL_RESULTS_FROZEN_HASHES_AND_ARCHIVE_MEMBERS_VERIFIED',
 'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'verified_result_files':len(rows),
 'full_forwards_total':200,'new_full_forwards':15,'new_fits':0,'detector_updates':0,
 'scope':'Artifact integrity only; scientific endpoint verification is separate.'}
(b/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
