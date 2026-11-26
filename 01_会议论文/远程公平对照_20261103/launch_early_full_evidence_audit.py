from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
root=Path('/25085400120/mpsma_fair_20261103_v1')
run=root/'runs/FULL'
row=json.loads((run/'best.metrics.json').read_text())
assert row['epoch']==0,'This launcher is only for an early saved-checkpoint integrity check'
assert len((run/'epochs.jsonl').read_text().splitlines())>=1
assert hashlib.sha256((root/'audit_saved_endpoint.py').read_bytes()).hexdigest()=='4aae6d69421280bf99690e7030a82847a5423a8e1b80b023f78bbfb1daf4a3f3'
assert not (root/'early_full_evidence_audit_launch.json').exists()
with (root/'early_full_evidence_audit_once.lock').open('x') as lock:lock.write(str(row['epoch']))
cmd=['/25085400120/envs/deim/bin/python','-B','-u',str(root/'audit_saved_endpoint.py'),'--arm','FULL','--endpoint','best']
with (root/'early_full_evidence_audit.log').open('x') as log:
    proc=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                          env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
receipt={'pid':proc.pid,'start_ticks':Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
         'launched_remote_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,
         'script_sha256':hashlib.sha256((root/'audit_saved_endpoint.py').read_bytes()).hexdigest(),
         'actual_secondary_runtime_sha256':hashlib.sha256((root/'secondary_runtime.json').read_bytes()).hexdigest(),
         'model_execution':False,'optimizer_updates':0,'evaluator_rerun':False,
         'scope':'Read and preserve a saved endpoint; verify hashes and summarize stored COCO arrays.'}
(root/'early_full_evidence_audit_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
