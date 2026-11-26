"""Copy only the verified environment archive between user-authorized hosts.

Passwords are prompted with echo disabled and remain only in process memory.
The archive is relayed through memory, never through a public service.
"""
from pathlib import Path
import datetime, getpass, hashlib, json, time, warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
import paramiko

LOCAL = Path(__file__).resolve().parent
ROOT = '/25085400120/mpsma_fair_20261103_v1'
def connect(port, key_file, label):
    password = getpass.getpass(label + ' SSH password (not saved): ')
    client = paramiko.SSHClient()
    client.load_host_keys(str(LOCAL/key_file))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('172.17.120.160', port=port, username='root', password=password,
                   look_for_keys=False, allow_agent=False, timeout=20, auth_timeout=20)
    client.get_transport().set_keepalive(20)
    return client

first=connect(30166,'known_hosts','Primary')
second=connect(30989,'known_hosts_secondary','Secondary')
started=time.monotonic()
state={'status':'COPYING','started_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'source_port':30166,'destination_port':30989,'credentials_saved':False}
def record():
    temp=LOCAL/'secondary_environment_transfer.json.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    temp.replace(LOCAL/'secondary_environment_transfer.json')
record()
try:
    a,b=first.open_sftp(),second.open_sftp()
    while True:
        with a.open(ROOT+'/secondary_transfer/environment_archive.json','rb') as stream:
            source=json.load(stream)
        if source['status']=='COMPLETED':break
        assert source['status']=='RUNNING', source['status']
        pid=int(source['pid'])
        with a.open('/proc/'+str(pid)+'/stat','rb') as stream:
            proc=stream.read().decode().rsplit(')',1)[1].split()
        assert proc[19]==source['start_ticks'] and proc[0] not in ['Z','X']
        assert time.monotonic()-started<3600, 'Archive observation deadline reached; do not restart it'
        state.update(status='WAITING_ON_VERIFIED_ARCHIVE_PROCESS',source_pid=pid,source_start_ticks=source['start_ticks'],
                     source_partial_bytes=a.stat(source['archive']+'.partial').st_size,elapsed_seconds=time.monotonic()-started)
        record();print(json.dumps(state),flush=True);time.sleep(15)
    assert source['status']=='COMPLETED'
    state.update(status='COPYING');record()
    target=ROOT+'/deim_environment.tar.gz'
    for candidate in [target,target+'.partial']:
        try:b.stat(candidate)
        except FileNotFoundError:pass
        else:raise RuntimeError('Destination already exists; do not duplicate transfer')
    digest=hashlib.sha256();copied=0;last_report=time.monotonic()
    with a.open(source['archive'],'rb') as src, b.open(target+'.partial','wx') as dest:
        dest.set_pipelined(True)
        while copied < source['bytes']:
            size=min(2*1024*1024,source['bytes']-copied)
            # Paramiko 2.8: a bounded readv request instead of unbounded prefetch.
            chunk=next(iter(src.readv([(copied,size)])))
            assert len(chunk)==size
            dest.write(chunk);digest.update(chunk);copied+=len(chunk)
            if time.monotonic()-last_report>=15:
                state.update(bytes=copied,total_bytes=source['bytes'],elapsed_seconds=time.monotonic()-started)
                record();print(json.dumps(state),flush=True);last_report=time.monotonic()
    assert copied==source['bytes'] and digest.hexdigest()==source['sha256']
    b.rename(target+'.partial',target)
    state.update(status='TRANSFERRED_DESTINATION_REHASH_REQUIRED',bytes=copied,sha256=digest.hexdigest(),
                 archive_source=source,destination=target,elapsed_seconds=time.monotonic()-started)
    record();print(json.dumps(state),flush=True)
except Exception as error:
    state.update(status='FAILED_NO_AUTOMATIC_RETRY',error=repr(error))
    record();raise
finally:
    first.close();second.close()
