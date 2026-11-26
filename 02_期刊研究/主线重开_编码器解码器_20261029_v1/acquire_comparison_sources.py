"""Read-only official Git source acquisition. No model/build/install execution."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'comparison_sources_v1'
DEST.mkdir(exist_ok=True)
GIT = r'C:\Program Files\Git\cmd\git.exe'
env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never', GIT_LFS_SKIP_SMUDGE='1')
options = [GIT, '-c', 'credential.helper=', '-c', 'http.sslVerify=true', '-c', 'http.sslBackend=openssl',
           '-c', 'http.version=HTTP/1.1', '-c', 'http.lowSpeedLimit=1', '-c', 'http.lowSpeedTime=15',
           '-c', 'core.autocrlf=false']
repos = [('Relation-DETR', 'https://github.com/xiuqhou/Relation-DETR.git'),
         ('DETRNN', 'https://github.com/nigelwoo2008-coder/DETRNN.git')]
receipts = []
for name, url in repos:
    dest = DEST / name
    began = time.monotonic()
    existed = dest.exists()
    if existed:
        origin = subprocess.check_output(options + ['-C',str(dest),'remote','get-url','origin'],
                                         env=env,text=True,encoding='utf-8').strip()
        assert origin == url
        result = subprocess.run(options + ['-C',str(dest),'rev-parse','--verify','HEAD'],
                                capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=15)
    else:
        result = subprocess.run(options + ['clone','--depth=1','--single-branch','--no-tags',url,str(dest)],
                                capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=45)
    row = {'repo':url,'local':str(dest),'exit_code':result.returncode,'seconds':time.monotonic()-began,
           'stderr':result.stderr[:1500], 'existing_clone_reused_without_mutation':existed}
    if result.returncode == 0:
        commit = subprocess.check_output(options + ['-C',str(dest),'rev-parse','HEAD'], env=env, text=True).strip()
        tree = subprocess.check_output(options + ['-C',str(dest),'ls-tree','-r','-z','HEAD'], env=env)
        checked = []
        for entry in tree.split(b'\0'):
            if not entry:
                continue
            meta, filename = entry.split(b'\t',1)
            mode, kind, oid = meta.split()
            assert kind == b'blob' and mode != b'120000', 'No implicit symlink/submodule execution'
            path = filename.decode('utf-8')
            data = (dest/path).read_bytes()
            digest = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert digest == oid.decode(), path
            checked.append({'path':path,'bytes':len(data),'git_blob_sha':digest,'sha256':hashlib.sha256(data).hexdigest()})
        row.update(status='ALL_SOURCE_BLOBS_VERIFIED',commit=commit,verified_files=checked)
    else:
        row['status'] = 'SOURCE_ACQUISITION_FAILED_NO_EXECUTION'
    receipts.append(row)
    (DEST / 'acquisition_receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k not in ['verified_files','stderr']}),flush=True)
