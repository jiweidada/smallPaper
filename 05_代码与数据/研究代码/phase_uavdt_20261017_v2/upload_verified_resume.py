"""Flow-controlled SSH stream; verifies an existing partial prefix before resuming data only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import time

from control_uavdt import ACCOUNTS, connect, scheduled_monitor_v4

RECEIVER=r'''
import hashlib,json,os,pathlib,sys
p=pathlib.Path(TARGET);part=pathlib.Path(TARGET+'.part')
assert not p.exists(),'Final target exists; verify it rather than replace it'
p.parent.mkdir(parents=True,exist_ok=True)
size=part.stat().st_size if part.exists() else 0;assert 0<=size<=TOTAL
h=hashlib.sha256()
if size:
 with part.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
print(json.dumps({'resume_offset':size,'prefix_sha256':h.hexdigest()}),flush=True)
with part.open('ab') as out:
 while size<TOTAL:
  b=sys.stdin.buffer.read(min(2*1024**2,TOTAL-size))
  if not b:raise EOFError('Transfer interrupted, verified partial prefix preserved')
  out.write(b);h.update(b);size+=len(b)
 out.flush();os.fsync(out.fileno())
assert size==TOTAL and h.hexdigest()==EXPECTED,'Final transferred SHA mismatch'
assert not p.exists();part.rename(p)
print(json.dumps({'bytes':size,'sha256':h.hexdigest(),'remote_sha_verified':True}),flush=True)
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--account',choices=['A','B'],required=True)
    parser.add_argument('--local',type=Path,required=True);parser.add_argument('--remote',required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();cfg=ACCOUNTS[args.account]
    assert PurePosixPath(args.remote).is_relative_to(PurePosixPath(cfg['storage']+'/datasets_verified_20261017'))
    assert not args.receipt.exists()
    size=args.local.stat().st_size
    program='TARGET='+repr(args.remote)+'\nTOTAL='+str(size)+'\nEXPECTED='+repr(args.sha256)+'\n'+RECEIVER
    passwords=json.loads(scheduled_monitor_v4.credentials_bytes());client=None;started=time.monotonic()
    try:
        client=connect(cfg['port'],passwords);passwords.clear()
        _,stdout,stderr=client.exec_command(shlex.quote(cfg['python'])+' -u -c '+shlex.quote(program),timeout=600)
        hello=json.loads(stdout.readline());offset=hello['resume_offset'];digest=hashlib.sha256()
        with args.local.open('rb') as source:
            remaining=offset
            while remaining:
                block=source.read(min(4*1024**2,remaining));assert block;digest.update(block);remaining-=len(block)
            assert digest.hexdigest()==hello['prefix_sha256'],'Existing remote partial does not match source; preserve and investigate'
            print(json.dumps({'prefix_verified':True,'resume_offset':offset,'total':size}),flush=True)
            last=time.monotonic();sent=offset
            while True:
                block=source.read(2*1024**2)
                if not block:break
                stdout.channel.sendall(block);digest.update(block);sent+=len(block)
                if time.monotonic()-last>=20:
                    print(json.dumps({'sent':sent,'total':size,'percent':round(sent/size*100,2)}),flush=True);last=time.monotonic()
            assert sent==size and digest.hexdigest()==args.sha256
            stdout.channel.shutdown_write()
        output=stdout.read().decode();error=stderr.read().decode();code=stdout.channel.recv_exit_status()
        assert code==0,(code,error[-1500:])
        result=json.loads(output);assert result['sha256']==args.sha256 and result['bytes']==size
        result.update(source=str(args.local.resolve()),remote=args.remote,account=args.account,
                      resumed_verified_bytes=offset,elapsed_seconds=time.monotonic()-started)
        with args.receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True)
    finally:
        passwords.clear()
        if client:client.close()

if __name__=='__main__':main()
