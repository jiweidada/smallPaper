from pathlib import Path
import argparse,json,math,os,subprocess,sys,time
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--manifest-sha256',required=True);p.add_argument('--recovery-manifest-sha256',required=True);a=p.parse_args()
old=json.loads((D/'worker_exit.json').read_text());assert old['author_entries']==old['author_completed']==2 and old['worker_exit_code']==1
started=time.time();code=None;timed_out=False
with (D/'recovery_collector_stdout.log').open('xb') as output:
    try:
        result=subprocess.run([sys.executable,str(D/'collect_features_export_recovery.py'),
            '--manifest-sha256',a.manifest_sha256,'--recovery-manifest-sha256',a.recovery_manifest_sha256],
            cwd=str(D),env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONUTF8='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
            stdout=output,stderr=subprocess.STDOUT,timeout=900-math.ceil(old['elapsed_seconds']))
        code=result.returncode
    except subprocess.TimeoutExpired:code=-999;timed_out=True
entries=completed=0
for name in ['forward_entries.jsonl','recovery_forward_entries.jsonl']:
    path=D/name
    if not path.exists():continue
    for line in path.open():
        row=json.loads(line)
        entries+=row['event']=='author_detector_entry';completed+=row['event']=='author_detector_completed'
success=False
if code==0 and (D/'collection_result.json').exists():
    result=json.loads((D/'collection_result.json').read_text());success=result['entries']==result['completed']==entries==completed==114
record=dict(status='RECOVERY_WORKER_TERMINAL_SUCCESS' if success else 'RECOVERY_TERMINAL_FAILURE_NO_FURTHER_RESTART',
    worker_exit_code=code,worker_timed_out=timed_out,success=success,combined_author_entries=entries,
    combined_author_completed=completed,wrapper_pid=os.getpid(),elapsed_seconds=time.time()-started,
    original_failed_worker_elapsed_seconds=old['elapsed_seconds'])
with (D/'recovery_worker_exit.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
sys.exit(0 if success else 1)
