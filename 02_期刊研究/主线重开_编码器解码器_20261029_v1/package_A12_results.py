from pathlib import Path
import hashlib
import json
import tarfile
root=Path(__file__).resolve().parent
completed=json.loads((root/'completed.json').read_text(encoding='utf-8'))
backup=root/'backup';backup.mkdir()
archive=backup/'A12_native_distribution_results.tar.gz'
items=[]
for r in completed['files']:
    path=root/r['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
    if r['path']=='prediction_blind_val_sources.tar.gz':continue
    items.append((path,r['path']))
items.append((root/'completed.json','completed.json'))
with tarfile.open(archive,'w:gz') as tar:
    for path,name in items:tar.add(path,arcname=name)
manifest={'files':[{'path':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path,name in items]}
(backup/'file_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
receipt={'status':'ARCHIVED_COMPLETED_NATIVE_CAPTURE','files':len(items),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'excluded_already_local_source_archive':True}
(backup/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
