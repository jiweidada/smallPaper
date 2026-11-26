"""Create an account-local immutable metadata view and verify every native frame."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import zipfile

storage=Path(CONFIG['storage']);base=storage/'datasets_verified_20261017'
root=base/'UAVDT_v1';native=base/'UAVDT_official/native_frames_v2'
assert not root.exists(),'Data view exists; inspect its receipt, never rerun or overwrite.'
packet=base/'UAVDT_metadata_v2.tar.gz'
assert hashlib.sha256(packet.read_bytes()).hexdigest()=='fee2a32aecccc701baf43a9687c0843f9ce5a84eb5c65d9aec74c2cc08fb230d'
root.mkdir()
with (root/'preparation_started.json').open('x') as f:json.dump({'account':CONFIG['account'],'pid':os.getpid(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
with tarfile.open(packet,'r:gz') as source:
    for item in source:
        path=root/item.name
        assert path.is_relative_to(root) and '..' not in Path(item.name).parts and item.isfile()
        path.parent.mkdir(parents=True,exist_ok=True)
        with source.extractfile(item) as src,path.open('xb') as dst:shutil.copyfileobj(src,dst,2*1024**2)
metadata=json.loads((root/'data_metadata.json').read_text())
for item in metadata['metadata_files']:
    path=root/item['relative'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
manifest=json.loads((root/'native_image_manifest.json').read_text());assert len(manifest)==40735
if not native.exists():
    archive=base/'UAVDT_official/raw/UAV-benchmark-M.zip'
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==metadata['image_archive_sha256'];native.mkdir()
    with zipfile.ZipFile(archive) as source:
        for index,item in enumerate(manifest,1):
            path=native/item['relative'];assert path.is_relative_to(native) and '..' not in Path(item['relative']).parts
            raw=source.read(item['archive_member']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(raw)
            if index%5000==0:print(json.dumps({'event':'extract_verified','images':index}),flush=True)
else:
    for index,item in enumerate(manifest,1):
        path=native/item['relative'];assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        if index%10000==0:print(json.dumps({'event':'existing_native_verified','images':index}),flush=True)
(root/'images').symlink_to(native,target_is_directory=True)
receipt={'state':'complete_data_view_verified','account':CONFIG['account'],'root':str(root),'images':40735,
         'all_native_image_sha_verified':True,'data_metadata_sha256':hashlib.sha256((root/'data_metadata.json').read_bytes()).hexdigest(),
         'annotations_sha256':{split:hashlib.sha256((root/'annotations'/(split+'.json')).read_bytes()).hexdigest() for split in ['train','val','test']},
         'split':metadata['split'],'counts':metadata['counts'],'training_ignore_treatment':metadata['training_ignore_treatment'],
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_training_started':False}
with (root/'data_ready.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
