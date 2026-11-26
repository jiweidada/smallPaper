"""Bounded acquisition of the same public official commit, no training/install."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

root=Path(__file__).resolve().parent
dest=root/'official_assets_v1'/'TinyFormer_git_9075d9f_v1'
commit='9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1'
git=r'C:\Program Files\Git\cmd\git.exe'
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
common=[git,'-c','credential.helper=','-c','http.sslVerify=true','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','-c','http.lowSpeedLimit=1','-c','http.lowSpeedTime=15','-c','core.autocrlf=false']
receipt={'commit':commit,'source':'https://github.com/mmpmmpmmpjosh/TinyFormer.git','path':str(dest),'operations':[]}

def run(args):
    started=time.monotonic()
    process=subprocess.Popen(common+args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        stdout,stderr=process.communicate(timeout=35)
        result={'exit_code':process.returncode,'seconds':time.monotonic()-started,'stdout':stdout.decode(errors='replace')[-1800:],'stderr':stderr.decode(errors='replace')[-1800:]}
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=8)
        stdout,stderr=process.communicate(timeout=5)
        result={'exit_code':'TERMINATED_AT_35S','seconds':time.monotonic()-started,'stderr':stderr.decode(errors='replace')[-1000:]}
    receipt['operations'].append(result)
    print(json.dumps(result),flush=True)
    return result

if not dest.exists():
    result=run(['clone','--depth=1','--no-checkout','--single-branch','--no-tags',receipt['source'],str(dest)])
else:
    result=run(['-C',str(dest),'rev-parse','HEAD'])
if result['exit_code']==0:
    head=run(['-C',str(dest),'rev-parse','HEAD'])
    if head['exit_code']==0 and head['stdout'].strip()==commit:
        checked=run(['-C',str(dest),'checkout','--detach',commit])
        if checked['exit_code']==0:
            tree=json.loads((root/'source_audit_v1'/'tinyformer_tree.json').read_text(encoding='utf-8'))
            assert tree['sha']==commit
            verified=[]
            for item in tree['tree']:
                if item['type']!='blob':continue
                path=dest/item['path']
                assert path.is_file() and not path.is_symlink(),item['path']
                blob=path.read_bytes()
                digest=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
                assert digest==item['sha'], item['path']
                verified.append({'path':item['path'],'git_blob_sha':digest,'bytes':len(blob)})
            receipt.update(status='OFFICIAL_COMMIT_ALL_BLOBS_VERIFIED',verified_files=verified)
receipt.setdefault('status','SOURCE_NOT_ACQUIRED')
(root/'official_assets_v1'/'source_git_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'verified_files':len(receipt.get('verified_files',[]))}),flush=True)
