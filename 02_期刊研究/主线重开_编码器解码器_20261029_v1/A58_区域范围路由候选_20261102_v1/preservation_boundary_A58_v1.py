"""Resolve mutable current-state pointers through exact prior snapshots."""
from pathlib import Path
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
B=P.parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(4<<20),b''):h.update(v)
    return h.hexdigest()

def save_new(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)

def begin(tag,prior_name):
    prior_path=P/prior_name
    prior=json.loads(prior_path.read_text('utf-8'))
    for n,r in prior['files'].items():
        f=Path(n)
        assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],n
    snap=P/('before_'+tag);snap.mkdir(exist_ok=False)
    mapping={}
    for f in [B/'01_候选判别表.json',B/'15_当前状态与结果校验.json',P/'CURRENT_A58.json']:
        target=snap/f.name;shutil.copyfile(f,target)
        mapping[str(f)]=dict(snapshot=str(target),sha256=sha(target))
    for n,r in prior['files'].items():
        f=Path(mapping[n]['snapshot']) if n in mapping else Path(n)
        assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],n
    save_new(P/(tag+'_prior_resolution.json'),dict(prior_receipt=str(prior_path),
             prior_receipt_sha256=sha(prior_path),status='ALL_PRIOR_BYTES_VERIFIED',
             prior_files=len(prior['files']),files=mapping))

def finish(name,status,**extra):
    files=[f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    files += [B/'01_候选判别表.json',B/'15_当前状态与结果校验.json']
    save_new(P/name,dict(status=status,goal_status='active',mainline_validated=False,
             files={str(f):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(set(files))},**extra))
