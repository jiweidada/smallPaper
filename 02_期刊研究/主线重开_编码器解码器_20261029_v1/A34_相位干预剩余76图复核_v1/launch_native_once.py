from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((R/'execution_inputs_manifest.json').read_text())
assert sha(R/'native_phase_worker.py')==m['worker_sha256'] and sha(R/'phase_protocol_FROZEN.json')==m['protocol_sha256']
with (R/'launch_once.lock').open('x') as f:f.write(str(os.getpid()))
assert not (R/'worker_once.lock').exists()
with (R/'stdout.log').open('xb') as out,(R/'stderr.log').open('xb') as err:
 p=subprocess.Popen([sys.executable,'-u',str(R/'native_phase_worker.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
pidfile=Path('/proc')/str(p.pid)/'stat'
receipt=dict(pid=p.pid,start_ticks=pidfile.read_text().split()[21] if pidfile.exists() else None,protocol_sha256=m['protocol_sha256'],worker_sha256=m['worker_sha256'],automatic_followups=False)
(R/'launch_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
