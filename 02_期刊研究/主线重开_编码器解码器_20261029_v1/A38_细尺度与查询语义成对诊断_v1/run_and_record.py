from pathlib import Path
import argparse,json,os,subprocess,sys,time

D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
started=time.time(); code=None; timed_out=False
environment=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONUTF8='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
with (D/'collector_stdout.log').open('xb') as output:
    try:
        result=subprocess.run([sys.executable,str(D/'collect_features.py'),'--manifest-sha256',a.manifest_sha256],
            cwd=str(D),env=environment,stdout=output,stderr=subprocess.STDOUT,timeout=900)
        code=result.returncode
    except subprocess.TimeoutExpired:
        timed_out=True;code=-999
entries=completed=0
ledger=D/'forward_entries.jsonl'
if ledger.exists():
    for line in ledger.open(encoding='utf-8'):
        row=json.loads(line)
        if row['event']=='author_detector_entry':entries+=1
        if row['event']=='author_detector_completed':completed+=1
success=False
if code==0 and (D/'collection_result.json').exists():
    result=json.loads((D/'collection_result.json').read_text(encoding='utf-8'))
    success=(result['entries']==result['completed']==entries==completed==114)
record=dict(worker_exit_code=code,worker_timed_out=timed_out,author_entries=entries,
    author_completed=completed,success=success,wrapper_pid=os.getpid(),elapsed_seconds=time.time()-started,
    status='WORKER_TERMINAL_SUCCESS' if success else 'WORKER_TERMINAL_FAILURE_NO_RESTART')
with (D/'worker_exit.json').open('x',encoding='utf-8') as output:json.dump(record,output,indent=2)
sys.exit(0 if success else 1)
