from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
assert hashlib.sha256((D/'execution_inputs_manifest.json').read_bytes()).hexdigest()==a.manifest_sha256
assert not (D/'worker_exit.json').exists() and not (D/'forward_entries.jsonl').exists()
lock=os.open(str(D/'launch.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
os.write(lock,str(os.getpid()).encode());os.close(lock)
child=subprocess.Popen([sys.executable,str(D/'run_and_record.py'),'--manifest-sha256',a.manifest_sha256],
    cwd=str(D),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
    start_new_session=True,close_fds=True)
stat=Path('/proc')/str(child.pid)/'stat'
ticks=stat.read_text().rsplit(')',1)[1].split()[19] if stat.exists() else None
receipt=dict(wrapper_pid=child.pid,wrapper_start_ticks=ticks,launched_unix=time.time(),
    manifest_sha256=a.manifest_sha256,launch_only_not_worker_exit=True,
    authoritative_exit_record=str(D/'worker_exit.json'),automatic_followups=False)
with (D/'launch_receipt.json').open('x',encoding='utf-8') as output:json.dump(receipt,output,indent=2)
print(json.dumps(receipt))
