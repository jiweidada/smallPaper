"""Persistent parallel native-frame verification/extraction after an interrupted SSH session."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import zipfile

base=Path(CONFIG['storage'])/'datasets_verified_20261017';root=base/'UAVDT_v1';native=base/'UAVDT_official/native_frames_v2'
try:
    assert not (root/'data_ready.json').exists()
    metadata=json.loads((root/'data_metadata.json').read_text())
    for item in metadata['metadata_files']:
        assert hashlib.sha256((root/item['relative']).read_bytes()).hexdigest()==item['sha256']
    manifest=json.loads((root/'native_image_manifest.json').read_text());assert len(manifest)==40735
    native.mkdir(exist_ok=True);archive=base/'UAVDT_official/raw/UAV-benchmark-M.zip';local=threading.local()
    def verify(item):
        path=native/item['relative'];assert path.is_relative_to(native) and '..' not in Path(item['relative']).parts
        if path.is_file():raw=path.read_bytes()
        else:
            assert CONFIG['account']=='A' and archive.stat().st_size==6790452113
            if not hasattr(local,'archive'):local.archive=zipfile.ZipFile(archive)
            raw=local.archive.read(item['archive_member']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(raw)
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        return len(raw)
    total=0;started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending=[pool.submit(verify,item) for item in manifest]
        for count,future in enumerate(concurrent.futures.as_completed(pending),1):
            total+=future.result()
            if count%1000==0 or count==len(manifest):
                progress={'verified_images':count,'total_images':len(manifest),'verified_bytes':total,'elapsed_seconds':time.monotonic()-started,
                          'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                p=root/'verification_progress_v2.json';tmp=p.with_suffix('.next');tmp.write_text(json.dumps(progress));os.replace(tmp,p)
                print(json.dumps(progress),flush=True)
    if not (root/'images').exists():(root/'images').symlink_to(native,target_is_directory=True)
    assert (root/'images').resolve()==native.resolve()
    receipt={'state':'complete_data_view_verified','account':CONFIG['account'],'root':str(root),'images':40735,
             'all_native_image_sha_verified':True,'data_metadata_sha256':hashlib.sha256((root/'data_metadata.json').read_bytes()).hexdigest(),
             'annotations_sha256':{s:hashlib.sha256((root/'annotations'/(s+'.json')).read_bytes()).hexdigest() for s in ['train','val','test']},
             'split':metadata['split'],'counts':metadata['counts'],'training_ignore_treatment':metadata['training_ignore_treatment'],
             'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_training_started':False,
             'persistent_parallel_verification':True}
    with (root/'data_ready.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)
except BaseException as exc:
    with (root/'verification_failed_v2.json').open('x') as f:json.dump({'error':type(exc).__name__+': '+str(exc)},f)
    raise
