"""Collect frozen input/results archives with exact SHA; relay the SAME inputs to B."""
import argparse,datetime,hashlib,json,pathlib,tarfile,time
from control_probe import ctl,ROOTS,LOCAL
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def get(sftp,remote,path,size,expected):
    if path.is_file():assert path.stat().st_size==size and sha(path)==expected;return
    part=path.with_suffix(path.suffix+'.part');offset=part.stat().st_size if part.exists() else 0;assert offset<=size;last=0
    with sftp.open(remote,'rb') as source,part.open('ab' if offset else 'xb') as target:
        while offset<size:
            chunks=[(pos,min(32768,size-pos)) for pos in range(offset,min(offset+32*32768,size),32768)]
            for (pos,n),block in zip(chunks,source.readv(chunks)):
                assert len(block)==n;target.write(block);offset+=len(block)
            if time.monotonic()-last>20:print(json.dumps({'transfer':remote,'bytes':offset,'total':size}),flush=True);last=time.monotonic()
    assert part.stat().st_size==size and sha(part)==expected;part.rename(path)
def execute(client,a,program):
    r=ctl.execute(client,a,'PROBE_ROOT='+repr(ROOTS[a])+'\n'+program,timeout=180)
    assert r['returncode']==0,r;return json.loads(r['stdout'])
def pack(client,a,kind):
    seed=1 if a=='A' else 2
    program='''import pathlib,json,tarfile,hashlib
root=pathlib.Path(PROBE_ROOT);kind=KIND;seed=SEED
folder=root/('inputs' if kind=='inputs' else f'results_seed{seed}')
assert json.loads((folder/'complete.json').read_text())['state']=='complete'
if kind!='inputs':
 assert not (folder/'failure.json').exists()
 assert json.loads((folder/'supplement_complete.json').read_text())['state']=='complete'
archive=root/(folder.name+'.tar.gz')
if not archive.exists():
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(folder.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
  for name in ['sample_plan.json','package_manifest.json']+([] if kind=='inputs' else ['deployment_receipt.json']):
   tar.add(root/name,arcname=name,recursive=False)
  if kind!='inputs' and (root/'task_records').is_dir():
   for p in (root/'task_records').glob('*'):
    if p.is_file():tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
  if kind!='inputs':
   for p in sorted((root/'extension_v2').glob('*')):
    if p.is_file():tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
   for name in ['probe_driver_v2.py',f'preflight_seed{seed}.json',f'extension_preflight_seed{seed}.json']:
    tar.add(root/name,arcname=name,recursive=False)
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest()}),flush=True)
'''
    return execute(client,a,'KIND='+repr(kind)+'\nSEED='+repr(seed)+'\n'+program)
def safe_extract(archive,dest):
    dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            rel=pathlib.PurePosixPath(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            target=dest/pathlib.Path(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as f:content=f.read()
            if target.exists():assert target.read_bytes()==content
            else:target.write_bytes(content)
def main(action,account):
    passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes());clients=[]
    try:
        a='A' if action=='inputs' else account;client=ctl.connect(ctl.ACCOUNTS[a]['port'],passwords);clients.append(client);meta=pack(client,a,'inputs' if action=='inputs' else 'results')
        local=LOCAL/(('shared_inputs' if action=='inputs' else a+'_results')+'.tar.gz');get(client.open_sftp(),meta['path'],local,meta['bytes'],meta['sha256'])
        dest=LOCAL/('shared_inputs' if action=='inputs' else a);safe_extract(local,dest)
        receipt={'action':action,'account':a,'remote':meta,'local':str(local),'local_sha256_verified':True}
        if action=='inputs':
            b=ctl.connect(ctl.ACCOUNTS['B']['port'],passwords);clients.append(b);sftp=b.open_sftp();remote=ROOTS['B']+'/shared_inputs.tar.gz'
            with sftp.open(remote,'wx') as target,local.open('rb') as source:
                target.set_pipelined(True)
                for block in iter(lambda:source.read(1024*1024),b''):target.write(block)
            program='''import pathlib,json,tarfile,hashlib
root=pathlib.Path(PROBE_ROOT);archive=root/'shared_inputs.tar.gz'
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert h.hexdigest()==EXPECTED
assert not (root/'inputs').exists()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  rel=pathlib.PurePosixPath(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
  p=root/pathlib.Path(*rel.parts)
  with tar.extractfile(member) as f:content=f.read()
  if p.exists():assert p.read_bytes()==content
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
m=json.loads((root/'inputs/manifest.json').read_text())
for item in m['batches']:assert hashlib.sha256((root/'inputs'/item['file']).read_bytes()).hexdigest()==item['sha256']
print(json.dumps({'state':'identical_inputs_verified','inputs':m['formal_inputs'],'archive_sha256':EXPECTED}),flush=True)
'''
            # Per-account deployment receipts differ; only inputs plus the two
            # common frozen manifests should be relayed, not account metadata.
            result=execute(b,'B','EXPECTED='+repr(meta['sha256'])+'\n'+program);receipt['B']=result
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(LOCAL/(stamp+'_'+action+'_'+a+'_transfer.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(receipt,ensure_ascii=False,indent=2))
    finally:
        passwords.clear()
        for c in clients:c.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inputs','results']);p.add_argument('--account',choices=['A','B'],default='A');a=p.parse_args();main(a.action,a.account)
