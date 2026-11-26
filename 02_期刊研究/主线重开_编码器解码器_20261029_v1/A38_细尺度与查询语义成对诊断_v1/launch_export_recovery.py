from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--manifest-sha256',required=True);p.add_argument('--recovery-manifest-sha256',required=True);a=p.parse_args()
assert hashlib.sha256((D/'execution_inputs_manifest.json').read_bytes()).hexdigest()==a.manifest_sha256
assert hashlib.sha256((D/'export_recovery_execution_manifest.json').read_bytes()).hexdigest()==a.recovery_manifest_sha256
old=json.loads((D/'worker_exit.json').read_text());assert old['author_entries']==old['author_completed']==2 and old['worker_exit_code']==1
pid=int((D/'collector.lock').read_text());assert not (Path('/proc')/str(pid)).exists()
assert not (D/'recovery_worker_exit.json').exists() and not (D/'collection_result.json').exists()
fd=os.open(str(D/'export_recovery_launch.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
child=subprocess.Popen([sys.executable,str(D/'run_recovery_and_record.py'),'--manifest-sha256',a.manifest_sha256,
    '--recovery-manifest-sha256',a.recovery_manifest_sha256],cwd=str(D),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,start_new_session=True,close_fds=True)
stat=Path('/proc')/str(child.pid)/'stat';ticks=stat.read_text().rsplit(')',1)[1].split()[19] if stat.exists() else None
receipt=dict(wrapper_pid=child.pid,wrapper_start_ticks=ticks,launched_unix=time.time(),remaining_native_entries=112,
    combined_cap=114,launch_only_not_worker_exit=True,authoritative_exit_record=str(D/'recovery_worker_exit.json'))
with (D/'recovery_launch_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
