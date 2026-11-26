"""Copy only frozen reviewed inputs to a new isolated directory on an authorized host."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import time
from ssh_control import connect

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ITEMS=[('secondary_frozen_inputs.tar.gz','e127bf14dacd999ca3794c90dc16d778f34f22ff135c227d599ad43ac243c36d'),
       ('verified_dataset.zip','fdde141711867fa1bc2de3a1e6342cca257eb056e2a1700795eb04068aa90266'),
       ('dataset_bundle_manifest.json','ba463a04fe688d5c3005cb3cf9ca6e13aa699b0b11016e15e7fa9b76b52e2a56')]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(4*1024**2):h.update(chunk)
    return h.hexdigest()

def mkdirs(sftp,path):
    for p in [*reversed(path.parents),path]:
        if str(p)=='/':continue
        try:sftp.stat(str(p))
        except FileNotFoundError:sftp.mkdir(str(p))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,choices=[30048],required=True);args=parser.parse_args()
    remote=PurePosixPath('/24085403014/mpsma_fresh_pair_20261112_v1/inputs')
    result={'port':args.port,'remote_root':str(remote),'files':[],'training_started':False}
    client=connect(args.port)
    try:
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(120)
            mkdirs(sftp,remote)
            for name,expected in ITEMS:
                local=PARENT/name
                if not local.exists() and name=='secondary_frozen_inputs.tar.gz':
                    candidates=list(PARENT.glob('**/secondary_frozen_inputs.tar.gz'))
                    assert len(candidates)==1;local=candidates[0]
                assert local.is_file() and digest(local)==expected
                final=remote/name;temporary=remote/(name+'.uploading')
                try:sftp.stat(str(final))
                except FileNotFoundError:pass
                else:raise RuntimeError('Refusing to overwrite existing frozen input: '+str(final))
                offset=0
                try:offset=sftp.stat(str(temporary)).st_size
                except FileNotFoundError:pass
                assert 0<=offset<=local.stat().st_size
                started=time.monotonic();last=started
                with local.open('rb') as src,sftp.open(str(temporary),'ab' if offset else 'wx') as dst:
                    src.seek(offset)
                    while chunk:=src.read(1024**2):
                        dst.write(chunk);offset+=len(chunk)
                        if time.monotonic()-last>20:
                            print(json.dumps({'file':name,'transferred_bytes':offset,'total_bytes':local.stat().st_size,'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True);last=time.monotonic()
                    dst.flush()
                assert sftp.stat(str(temporary)).st_size==local.stat().st_size
                sftp.posix_rename(str(temporary),str(final))
                result['files'].append({'name':name,'bytes':local.stat().st_size,'expected_sha256':expected,'remote_sha_pending':True})
                print(json.dumps({'file_uploaded':name,'bytes':local.stat().st_size}),flush=True)
        (HERE/'input_transfer_30048.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    finally:client.close()

if __name__=='__main__':main()
