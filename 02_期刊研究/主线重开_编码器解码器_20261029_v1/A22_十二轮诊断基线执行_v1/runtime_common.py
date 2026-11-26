"""Shared pinned paths and logging. No detector construction or forward at import."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,sys,tempfile
ROOT=Path(__file__).resolve().parent
AP=ROOT.parent
A20=AP/'a20_vssd_host_capacity_20261030_v1'
A21=AP/'a21_shared_state_path_20261030_v1'
TRAIN=AP.parent/'dataset/train'
sys.path.insert(0,str(A20));sys.path.insert(0,str(A21))
import a20_host as host
import torch

def utc():return datetime.now(timezone.utc).isoformat()
def rd(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return host.sha(p)
def atomic_json(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def event(name,value,durable=False):
    row={'utc':utc(),**value}
    with (ROOT/name).open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        if durable:os.fsync(f.fileno())
    return row
def process_identity(pid):
    p=Path('/proc')/str(pid)
    stat=(p/'stat').read_text().split()
    return {'pid':pid,'start_ticks':int(stat[21]),'state':stat[2],
            'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),
            'cwd':str((p/'cwd').resolve())}
def verify_execution_manifest():
    m=rd(ROOT/'execution_manifest_FROZEN.json')
    assert m['status']=='FROZEN_TRAINING_EXECUTION_ONLY'
    for x in m['files']:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    for x in m['dependencies']:assert sha(Path(x['path']))==x['sha256'],x['path']
    return m
def author_transform():
    p=AP/'dome_assets_v1/dome_probe_loader.py'
    spec=importlib.util.spec_from_file_location('a22_transform_definition',p)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    return loader.input_transform()[0]
def tensor_hash(x):
    x=x.detach().as_subclass(torch.Tensor).contiguous().cpu()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def finite_tensors(values):
    v=[torch.isfinite(x).all() for x in values]
    return bool(torch.stack(v).all()) if v else True
