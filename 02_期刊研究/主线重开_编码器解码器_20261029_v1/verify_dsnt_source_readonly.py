"""Verify Git blobs,including documentation links as link text;never follow/import them."""
from pathlib import Path,PurePosixPath
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parent
repo=ROOT/'comparison_sources_v1/dsntnn'
git=r'C:\Program Files\Git\cmd\git.exe'
commit=subprocess.check_output([git,'-C',str(repo),'rev-parse','HEAD'],text=True).strip()
tree=subprocess.check_output([git,'-C',str(repo),'ls-tree','-r','-z','HEAD'])
checked=[]
for entry in tree.split(b'\0'):
    if not entry: continue
    meta,name=entry.split(b'\t',1)
    mode,kind,oid=meta.split()
    assert kind==b'blob' and mode!=b'160000'
    rel=name.decode('utf-8')
    assert not PurePosixPath(rel).is_absolute() and '..' not in PurePosixPath(rel).parts
    path=repo/rel
    data=os.readlink(path).encode() if mode==b'120000' and path.is_symlink() else path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert blob==oid.decode(),rel
    checked.append(dict(file=rel,mode=mode.decode(),bytes=len(data),git_blob_sha=blob,sha256=hashlib.sha256(data).hexdigest()))
receipt=dict(status='ALL_TRACKED_BLOBS_VERIFIED_LINKS_AS_TEXT_NOT_FOLLOWED',commit=commit,
             repo='https://github.com/anibali/dsntnn',files=checked,
             original_failure='Previous generic verifier rejected mode120000 documentation symlinks before declaring success;not a missing implementation.',
             read_scope='dsntnn/__init__.py:linear_expectation,normalized_linspace,soft_argmax,dsnt,flat_softmax;no project import,build or model execution')
(ROOT/'comparison_sources_v1/dsnt_source_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}|dict(verified_files=len(checked),link_entries=sum(r['mode']=='120000' for r in checked))))
